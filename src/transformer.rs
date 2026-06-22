// ============================================================================
// transformer.rs — Transformer Block Stack (Pure Rust)
// ============================================================================
//
// LLaMA Transformer Architecture (per layer):
//
//   x_norm = RMSNorm(x)                    // Pre-norm
//   attn_out = Attention(x_norm)           // Self-attention
//   x = x + attn_out                       // Residual connection
//
//   x_norm = RMSNorm(x)                    // Pre-norm
//   ffn_out = SwiGLU(x_norm)              // Feed-forward network
//   x = x + ffn_out                        // Residual connection
//
//   [repeat for each layer]
//
//   x = RMSNorm(x)                         // Final normalization
//   logits = x @ W_lm_head^T              // Language model head
//
// Note: LLaMA uses pre-norm (RMSNorm before attention/FFN), not post-norm.

use crate::attention::Attention;
use crate::kv_cache::KVCache;
use crate::math;
use crate::mlp::Mlp;
use crate::rmsnorm::RmsNorm;
use crate::rope::RoPE;

// ============================================================================
// Transformer Layer
// ============================================================================

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    /// Layer index
    pub layer_idx: usize,
    /// Attention norm (RMSNorm)
    pub attention_norm: RmsNorm,
    /// Self-attention
    pub attention: Attention,
    /// FFN norm (RMSNorm)
    pub ffn_norm: RmsNorm,
    /// Feed-forward network (SwiGLU)
    pub mlp: Mlp,
}

impl TransformerLayer {
    /// Forward pass for a single token
    pub fn forward(
        &self,
        x: &[f32],
        position: usize,
        rope: &RoPE,
        kv_cache: &mut KVCache,
        output: &mut [f32],
    ) {
        let embed_dim = x.len();

        // --- Attention block ---
        // x_norm = RMSNorm(x)
        let mut x_norm = vec![0.0f32; embed_dim];
        self.attention_norm.forward(x, &mut x_norm);

        // attn_out = Attention(x_norm)
        let mut attn_out = vec![0.0f32; embed_dim];
        self.attention
            .forward(&x_norm, position, rope, kv_cache, &mut attn_out);

        // x = x + attn_out (residual)
        output.copy_from_slice(x);
        math::vec_add_inplace(output, &attn_out);

        // --- FFN block ---
        // x_norm = RMSNorm(x)
        let mut x_norm2 = vec![0.0f32; embed_dim];
        self.ffn_norm.forward(output, &mut x_norm2);

        // ffn_out = SwiGLU(x_norm)
        let mut ffn_out = vec![0.0f32; embed_dim];
        self.mlp.forward(&x_norm2, &mut ffn_out);

        // x = x + ffn_out (residual)
        math::vec_add_inplace(output, &ffn_out);
    }
}

// ============================================================================
// Full Transformer Model
// ============================================================================

#[derive(Debug, Clone)]
pub struct Transformer {
    /// Vocabulary embedding table: (vocab_size, embed_dim)
    pub token_embd: Vec<f32>,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Transformer layers
    pub layers: Vec<TransformerLayer>,
    /// Final RMSNorm weight
    pub final_norm: RmsNorm,
    /// Language model head (output projection): (vocab_size, embed_dim)
    pub lm_head: Vec<f32>,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Transformer {
    /// Forward pass for a single token position.
    ///
    /// Parameters:
    ///   `token_id`: input token ID
    ///   `position`: position index in the sequence
    ///   `rope`: precomputed RoPE frequencies
    ///   `kv_caches`: one KV cache per layer
    ///   `logits`: output logits buffer [vocab_size]
    pub fn forward(
        &self,
        token_id: usize,
        position: usize,
        rope: &RoPE,
        kv_caches: &mut [KVCache],
        logits: &mut [f32],
    ) {
        assert_eq!(kv_caches.len(), self.layers.len());

        // --- Token embedding lookup ---
        let emb_start = token_id * self.embed_dim;
        let mut hidden = vec![0.0f32; self.embed_dim];
        hidden.copy_from_slice(&self.token_embd[emb_start..emb_start + self.embed_dim]);

        // --- Process through transformer layers ---
        let mut layer_output = vec![0.0f32; self.embed_dim];
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(&hidden, position, rope, &mut kv_caches[i], &mut layer_output);
            std::mem::swap(&mut hidden, &mut layer_output);
        }

        // --- Final RMSNorm ---
        self.final_norm.forward_inplace(&mut hidden);

        // --- LM head: logits = hidden @ W_lm_head^T ---
        math::mat_vec_mul_transposed(&self.lm_head, &hidden, logits, self.vocab_size, self.embed_dim);
    }
}
