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

        // --- Load tokenizer ---
        let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata);

        // --- Precompute RoPE frequencies ---
        let head_dim = config.embed_dim / config.n_heads;
        let rope = RoPE::new(head_dim, config.rope_base, config.max_seq_len);

        // --- Load tensor weights ---
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
                GgufDataType::Q5_0 => {
                    let n_blocks = (n_elements + 31) / 32;
                    n_blocks * 22
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

            tensors.insert(tensor_info.name.clone(), f32_data);
        }

        // --- Build transformer model ---
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
            token_embd.clone()
        };


        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {

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
            let bq = tensors
                .remove(&format!("blk.{}.attn_q.bias", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_q.bias", i))?;
            let bk = tensors
                .remove(&format!("blk.{}.attn_k.bias", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_k.bias", i))?;
            let bv = tensors
                .remove(&format!("blk.{}.attn_v.bias", i))
                .ok_or_else(|| format!("Missing blk.{}.attn_v.bias", i))?;

            let attention = Attention::new(
                i,
                config.n_heads,
                config.n_kv_heads,
                head_dim,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
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
            
            // Verify tensor dimensions
            let expected_gate = config.hidden_dim * config.embed_dim;
            let expected_down = config.embed_dim * config.hidden_dim;
            if gate_proj.len() != expected_gate {
                eprintln!("  WARNING: blk.{}.ffn_gate.weight: expected {} elements, got {}", i, expected_gate, gate_proj.len());
            }
            if up_proj.len() != expected_gate {
                eprintln!("  WARNING: blk.{}.ffn_up.weight: expected {} elements, got {}", i, expected_gate, up_proj.len());
            }
            if down_proj.len() != expected_down {
                eprintln!("  WARNING: blk.{}.ffn_down.weight: expected {} elements, got {}", i, expected_down, down_proj.len());
            }
            
            // Check for NaN/Inf in gate_proj and up_proj for all layers
            let gate_w = &gate_proj;
            let has_nan = gate_w.iter().any(|&v| v.is_nan());
            let has_inf = gate_w.iter().any(|&v| v.is_infinite());
            if has_nan || has_inf {
                eprintln!("  ERROR: blk.{}.ffn_gate.weight has NaN or Inf!", i);
            }
            let up_w = &up_proj;
            let has_nan = up_w.iter().any(|&v| v.is_nan());
            let has_inf = up_w.iter().any(|&v| v.is_infinite());
            if has_nan || has_inf {
                eprintln!("  ERROR: blk.{}.ffn_up.weight has NaN or Inf!", i);
            }

            let mlp = Mlp::new(gate_proj, up_proj, down_proj, config.hidden_dim, config.embed_dim);

            if cfg!(debug_assertions) {
                let w = &mlp.down_proj;
                let min = w.iter().cloned().fold(f32::MAX, f32::min);
                let max = w.iter().cloned().fold(f32::MIN, f32::max);
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                eprintln!("  blk.{}.ffn_down: min={:+.4} max={:+.4} mean={:+.6}", i, min, max, mean);
            }

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

        Ok(Model {
            transformer,
            tokenizer,
            config,
            rope,
        })
    }

    /// Generate text given a prompt, streaming output to stdout with TPS display
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler_config: SamplerConfig,
    ) -> String {
        use std::io::Write;
        let mut sampler = Sampler::new(sampler_config);
        let mut kv_caches: Vec<KVCache> = (0..self.config.n_layers)
            .map(|_| {
                let kv_dim = self.config.n_kv_heads * (self.config.embed_dim / self.config.n_heads);
                KVCache::new(kv_dim, self.config.max_seq_len)
            })
            .collect();

        // Encode prompt (Qwen2 does NOT use BOS token, despite GGUF metadata having it)
        let prompt_ids = self.tokenizer.encode(prompt);

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
        let mut byte_buf: Vec<u8> = Vec::new();
        let mut consumed = 0;

        for _step in 0..max_tokens {
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

            // Decode token to raw bytes and buffer them for UTF-8 streaming
            let bytes = self.tokenizer.decode_token_bytes(next_token);
            byte_buf.extend_from_slice(&bytes);
            // Flush as much valid UTF-8 as possible, replacing invalid bytes
            loop {
                if consumed >= byte_buf.len() {
                    break;
                }
                match std::str::from_utf8(&byte_buf[consumed..]) {
                    Ok(s) => {
                        if !s.is_empty() {
                            print!("{}", s);
                            generated_text.push_str(s);
                        }
                        consumed = byte_buf.len();
                        break;
                    }
                    Err(e) => {
                        let valid_len = e.valid_up_to();
                        if valid_len > 0 {
                            if let Ok(s) = std::str::from_utf8(&byte_buf[consumed..consumed + valid_len]) {
                                print!("{}", s);
                                generated_text.push_str(s);
                            }
                            consumed += valid_len;
                        }
                        let error_len = e.error_len().unwrap_or(1);
                        print!("\u{FFFD}");
                        generated_text.push('\u{FFFD}');
                        consumed += error_len;
                    }
                }
            }
            std::io::stdout().flush().ok();

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

        // Flush remaining bytes in the buffer
        while consumed < byte_buf.len() {
            match std::str::from_utf8(&byte_buf[consumed..]) {
                Ok(s) => {
                    if !s.is_empty() {
                        print!("{}", s);
                        generated_text.push_str(s);
                    }
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        if let Ok(s) = std::str::from_utf8(&byte_buf[consumed..consumed + valid_len]) {
                            print!("{}", s);
                            generated_text.push_str(s);
                        }
                        consumed += valid_len;
                    }
                    let error_len = e.error_len().unwrap_or(1);
                    print!("\u{FFFD}");
                    generated_text.push('\u{FFFD}');
                    consumed += error_len;
                }
            }
        }
        std::io::stdout().flush().ok();
        
        println!();
        
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
