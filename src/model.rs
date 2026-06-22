// ============================================================================
// model.rs — Model Loading and High-Level Interface (Pure Rust)
// ============================================================================
//
// This module:
//   1. Reads GGUF file metadata to determine model architecture
//   2. Loads and dequantizes all tensor weights
//   3. Constructs the Transformer model
//   4. Provides a high-level generate() interface
//
// Supported architectures: LLaMA, Llama-2, Llama-3, Mistral, Qwen2
// (all share the same GGUF weight naming convention)

use std::collections::HashMap;
use std::path::Path;

use crate::attention::Attention;
use crate::gguf::{GgufFile, GgufDataType, MetadataValue};
use crate::kv_cache::KVCache;
use crate::mlp::Mlp;
use crate::quant;
use crate::rmsnorm::RmsNorm;
use crate::rope::RoPE;
use crate::sampler::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;
use crate::transformer::{Transformer, TransformerLayer};

// ============================================================================
// Model Configuration (parsed from GGUF metadata)
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub embed_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub architecture: String,
}

impl ModelConfig {
    /// Parse model configuration from GGUF metadata
    pub fn from_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        let get_u32 = |key: &str| -> usize {
            metadata
                .get(key)
                .and_then(|v| v.to_u64())
                .unwrap_or(0) as usize
        };

        let get_f32 = |key: &str| -> f32 {
            metadata
                .get(key)
                .and_then(|v| v.to_f32())
                .unwrap_or(0.0)
        };

        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.to_string_ref())
            .unwrap_or("llama")
            .to_string();

        let prefix = format!("{}.", architecture);

        let embed_dim = get_u32(&format!("{}embedding_length", prefix));
        let n_layers = get_u32(&format!("{}block_count", prefix));
        let n_heads = get_u32(&format!("{}attention.head_count", prefix));
        let n_kv_heads = get_u32(&format!("{}attention.head_count_kv", prefix));
        let hidden_dim = get_u32(&format!("{}feed_forward_length", prefix));
        let vocab_size = get_u32(&format!("{}vocab_size", prefix));
        let max_seq_len = get_u32(&format!("{}context_length", prefix));
        let rope_base = get_f32(&format!("{}rope.freq_base", prefix));

        let norm_eps = metadata
            .get(&format!("{}attention.layer_norm_rms_epsilon", prefix))
            .or_else(|| metadata.get(&format!("{}attention.layer_norm_epsilon", prefix)))
            .and_then(|v| v.to_f32())
            .unwrap_or(1e-5);

        // Architecture-specific defaults
        let (default_embed, default_layers, default_heads, default_hidden, default_vocab, default_rope_base) =
            match architecture.as_str() {
                "qwen2" => (3584, 28, 28, 18944, 151936, 1000000.0),  // Qwen2-7B defaults
                "llama" => (4096, 32, 32, 11008, 32000, 10000.0),     // LLaMA-2-7B defaults
                "mistral" => (4096, 32, 32, 14336, 32000, 10000.0),   // Mistral-7B defaults
                _ => (4096, 32, 32, 11008, 32000, 10000.0),           // Generic defaults
            };

        let embed_dim = if embed_dim == 0 { default_embed } else { embed_dim };
        let n_layers = if n_layers == 0 { default_layers } else { n_layers };
        let n_heads = if n_heads == 0 { default_heads } else { n_heads };
        let n_kv_heads = if n_kv_heads == 0 { n_heads } else { n_kv_heads };
        let hidden_dim = if hidden_dim == 0 { default_hidden } else { hidden_dim };
        let vocab_size = if vocab_size == 0 { default_vocab } else { vocab_size };
        let max_seq_len = if max_seq_len == 0 { 32768 } else { max_seq_len };
        let rope_base = if rope_base == 0.0 { default_rope_base } else { rope_base };

        ModelConfig {
            embed_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            vocab_size,
            max_seq_len,
            rope_base,
            norm_eps,
            architecture,
        }
    }
}

// ============================================================================
// Model
// ============================================================================

pub struct Model {
    pub transformer: Transformer,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
    pub rope: RoPE,
}

impl Model {
    /// Load model from a GGUF file
    pub fn load(path: &Path, max_seq_len: Option<usize>) -> Result<Self, String> {
        // --- Parse GGUF header ---
        let gguf = GgufFile::load(path).map_err(|e| format!("Failed to parse GGUF: {}", e))?;

        // --- Parse model config ---
        let mut config = ModelConfig::from_metadata(&gguf.metadata);
        if let Some(max_len) = max_seq_len {
            config.max_seq_len = max_len;
        }

        eprintln!("\nModel configuration:");
        eprintln!("  Architecture:  {}", config.architecture);
        eprintln!("  Embed dim:     {}", config.embed_dim);
        eprintln!("  Layers:        {}", config.n_layers);
        eprintln!("  Heads:         {}", config.n_heads);
        eprintln!("  KV heads:      {}", config.n_kv_heads);
        eprintln!("  Hidden dim:    {}", config.hidden_dim);
        eprintln!("  Vocab size:    {}", config.vocab_size);
        eprintln!("  Max seq len:   {}", config.max_seq_len);
        eprintln!("  Rope base:     {}", config.rope_base);
        eprintln!("  Norm eps:      {}", config.norm_eps);

        // --- Load tokenizer ---
        let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata);

        // --- Precompute RoPE frequencies ---
        let head_dim = config.embed_dim / config.n_heads;
        let rope = RoPE::new(head_dim, config.rope_base, config.max_seq_len);

        // --- Load tensor weights ---
        eprintln!("\nLoading tensor weights...");
        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();

        for tensor_info in &gguf.tensors {
            let n_elements = tensor_info.n_elements();
            let byte_size = match tensor_info.data_type {
                GgufDataType::F32 => n_elements * 4,
                GgufDataType::F16 => n_elements * 2,
                GgufDataType::Q4_0 => {
                    let n_blocks = (n_elements + 31) / 32;
                    n_blocks * 18
                }
                GgufDataType::Q4_K => {
                    let n_blocks = (n_elements + 255) / 256;
                    n_blocks * 144
                }
                GgufDataType::Q5_K => {
                    let n_blocks = (n_elements + 255) / 256;
                    n_blocks * 176
                }
                GgufDataType::Q6_K => {
                    let n_blocks = (n_elements + 255) / 256;
                    n_blocks * 210
                }
                GgufDataType::Q8_0 => {
                    let n_blocks = (n_elements + 31) / 32;
                    n_blocks * 34
                }
                _ => {
                    eprintln!(
                        "  WARNING: Unsupported type {:?} for tensor '{}', skipping",
                        tensor_info.data_type, tensor_info.name
                    );
                    continue;
                }
            };

            let file_offset = gguf.data_offset + tensor_info.offset;

            // Read raw bytes from file
            let raw_data = gg::read_tensor_data(path, file_offset, byte_size)
                .map_err(|e| format!("Failed to read tensor '{}': {}", tensor_info.name, e))?;

            // Dequantize to f32
            let f32_data =
                quant::dequantize(&raw_data, tensor_info.data_type, n_elements)
                    .map_err(|e| format!("Failed to dequantize '{}': {}", tensor_info.name, e))?;

            eprintln!(
                "  Loaded: {:<50} shape={:?} type={:?} ({:.1} MB)",
                tensor_info.name,
                tensor_info.dims,
                tensor_info.data_type,
                f32_data.len() as f64 * 4.0 / (1024.0 * 1024.0)
            );

            tensors.insert(tensor_info.name.clone(), f32_data);
        }

        // --- Build transformer model ---
        eprintln!("\nBuilding transformer model...");
        let head_dim = config.embed_dim / config.n_heads;

        // Load token embeddings
        let token_embd = tensors
            .remove("token_embd.weight")
            .ok_or("Missing token_embd.weight")?;

        // Load final norm
        let final_norm_weight = tensors
            .remove("output_norm.weight")
            .ok_or("Missing output_norm.weight")?;
        let final_norm = RmsNorm::new(final_norm_weight, config.norm_eps);

        // Load LM head (may be tied with token embeddings)
        let lm_head = if let Some(w) = tensors.remove("output.weight") {
            w
        } else {
            // Weight tying: lm_head shares weights with token_embd
            eprintln!("  Using weight tying: lm_head = token_embd");
            token_embd.clone()
        };

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            eprintln!("  Loading layer {}...", i);

            let norm_w = tensors
                .remove(&format!("blk.{}.attn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_norm.weight", i))?;
            let attention_norm = RmsNorm::new(norm_w, config.norm_eps);

            let wq = tensors
                .remove(&format!("blk.{}.attn_q.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_q.weight", i))?;
            let wk = tensors
                .remove(&format!("blk.{}.attn_k.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_k.weight", i))?;
            let wv = tensors
                .remove(&format!("blk.{}.attn_v.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_v.weight", i))?;
            let wo = tensors
                .remove(&format!("blk.{}.attn_output.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_output.weight", i))?;

            let attention = Attention::new(
                i,
                config.n_heads,
                config.n_kv_heads,
                head_dim,
                wq,
                wk,
                wv,
                wo,
            );

            let ffn_norm_w = tensors
                .remove(&format!("blk.{}.ffn_norm.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_norm.weight", i))?;
            let ffn_norm = RmsNorm::new(ffn_norm_w, config.norm_eps);

            let gate_proj = tensors
                .remove(&format!("blk.{}.ffn_gate.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_gate.weight", i))?;
            let up_proj = tensors
                .remove(&format!("blk.{}.ffn_up.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_up.weight", i))?;
            let down_proj = tensors
                .remove(&format!("blk.{}.ffn_down.weight", i))
                .ok_or_else(|| format!("Missing blk.{}.ffn_down.weight", i))?;

            let mlp = Mlp::new(gate_proj, up_proj, down_proj, config.hidden_dim, config.embed_dim);

            layers.push(TransformerLayer {
                layer_idx: i,
                attention_norm,
                attention,
                ffn_norm,
                mlp,
            });
        }

        let transformer = Transformer {
            token_embd,
            embed_dim: config.embed_dim,
            vocab_size: config.vocab_size,
            layers,
            final_norm,
            lm_head,
            max_seq_len: config.max_seq_len,
        };

        eprintln!("\nModel loaded successfully!\n");

        Ok(Model {
            transformer,
            tokenizer,
            config,
            rope,
        })
    }

    /// Generate text given a prompt
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler_config: SamplerConfig,
    ) -> String {
        let mut sampler = Sampler::new(sampler_config);
        let mut kv_caches: Vec<KVCache> = (0..self.config.n_layers)
            .map(|_| {
                let kv_dim = self.config.n_kv_heads * (self.config.embed_dim / self.config.n_heads);
                KVCache::new(kv_dim, self.config.max_seq_len)
            })
            .collect();

        // Encode prompt
        let prompt_ids = self.tokenizer.encode(prompt);
        eprintln!("Prompt tokens: {:?}", prompt_ids);
        eprintln!("Prompt: \"{}\"", prompt);
        eprintln!("---");

        let mut all_token_ids: Vec<usize> = Vec::new();
        let mut logits = vec![0.0f32; self.config.vocab_size];

        // --- Process prompt tokens (prefill) ---
        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            self.transformer
                .forward(token_id, pos, &self.rope, &mut kv_caches, &mut logits);
            all_token_ids.push(token_id);
        }

        // --- Generate new tokens ---
        let mut current_pos = prompt_ids.len();
        let mut generated_text = String::new();

        for step in 0..max_tokens {
            // Sample next token
            let recent = if all_token_ids.len() > 64 {
                &all_token_ids[all_token_ids.len() - 64..]
            } else {
                &all_token_ids
            };
            let next_token = sampler.sample(&mut logits, recent);

            // Check for end-of-sequence
            if let Some(eos_id) = self.tokenizer.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }

            // Decode token
            if let Some(text) = self.tokenizer.id_to_token(next_token) {
                generated_text.push_str(text);
            }

            // Forward pass for the new token
            self.transformer.forward(
                next_token,
                current_pos,
                &self.rope,
                &mut kv_caches,
                &mut logits,
            );

            all_token_ids.push(next_token);
            current_pos += 1;
        }

        generated_text
    }

    /// Get model config
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

// Helper module for reading tensor data from file
mod gg {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::Path;

    pub fn read_tensor_data(path: &Path, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        let mut file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Cannot seek: {}", e))?;
        let mut buf = vec![0u8; size];
        file.read_exact(&mut buf)
            .map_err(|e| format!("Cannot read: {}", e))?;
        Ok(buf)
    }
}
