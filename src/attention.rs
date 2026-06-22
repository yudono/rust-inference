// ============================================================================
// attention.rs — Multi-Head Self-Attention (Pure Rust)
// ============================================================================
//
// Transformer attention:
//   Q = x @ W_q  (query projection)
//   K = x @ W_k  (key projection)
//   V = x @ W_v  (value projection)
//
//   Q = reshape(Q, [n_heads, head_dim])
//   K = reshape(K, [n_kv_heads, head_dim])
//   V = reshape(V, [n_kv_heads, head_dim])
//
//   Apply RoPE to Q and K
//   Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
//   Apply causal mask (upper triangular -inf)
//   Apply softmax
//   Compute output: out = scores @ V
//   Concatenate heads
//   Output projection: out = concat_heads @ W_o

use crate::kv_cache::KVCache;
use crate::math;
use crate::rope::RoPE;

// ============================================================================
// Attention Layer
// ============================================================================

#[derive(Debug, Clone)]
pub struct Attention {
    /// Layer index
    pub layer_idx: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key/value heads (for GQA, may be < n_heads)
    pub n_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// GQA repetition factor
    pub n_rep: usize,
    /// Weight matrices (stored as flat f32 arrays)
    pub wq: Vec<f32>, // (n_heads * head_dim, embed_dim)
    pub wk: Vec<f32>, // (n_kv_heads * head_dim, embed_dim)
    pub wv: Vec<f32>, // (n_kv_heads * head_dim, embed_dim)
    pub wo: Vec<f32>, // (embed_dim, n_heads * head_dim)
}

impl Attention {
    /// Create a new attention layer
    pub fn new(
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
    ) -> Self {
        let n_rep = n_heads / n_kv_heads;
        Attention {
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep,
            wq,
            wk,
            wv,
            wo,
        }
    }

    /// Compute attention for a single token position.
    ///
    /// Parameters:
    ///   `x`: input hidden state for the current token [embed_dim]
    ///   `position`: position index (for RoPE)
    ///   `rope`: precomputed RoPE frequencies
    ///   `cache`: KV cache for this layer
    ///   `output`: output buffer [embed_dim]
    pub fn forward(
        &self,
        x: &[f32],
        position: usize,
        rope: &RoPE,
        cache: &mut KVCache,
        output: &mut [f32],
    ) {
        let embed_dim = x.len();
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q_dim = self.n_heads * self.head_dim;

        // --- Project Q, K, V ---
        // Q = x @ W_q^T
        let mut q = vec![0.0f32; q_dim];
        math::mat_vec_mul_transposed(&self.wq, x, &mut q, q_dim, embed_dim);

        // K = x @ W_k^T
        let mut k = vec![0.0f32; kv_dim];
        math::mat_vec_mul_transposed(&self.wk, x, &mut k, kv_dim, embed_dim);

        // V = x @ W_v^T
        let mut v = vec![0.0f32; kv_dim];
        math::mat_vec_mul_transposed(&self.wv, x, &mut v, kv_dim, embed_dim);

        // --- Apply RoPE to Q and K ---
        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            rope.apply(&mut q[start..end], position);
        }
        for h in 0..self.n_kv_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            rope.apply(&mut k[start..end], position);
        }

        // --- Store K and V in cache ---
        cache.append_k(&k);
        cache.append_v(&v);

        // --- Compute attention ---
        // For each head, compute: out_h = softmax(Q_h @ K_h^T / sqrt(d)) @ V_h
        let seq_len = cache.len();
        let mut attn_output = vec![0.0f32; q_dim];

        for h in 0..self.n_heads {
            let q_head_start = h * self.head_dim;
            let kv_head = h / self.n_rep; // GQA: map query head to KV head
            let kv_head_start = kv_head * self.head_dim;

            // Compute attention scores: scores[j] = dot(Q_h, K_j_h) / sqrt(head_dim)
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let mut scores = vec![0.0f32; seq_len];

            for j in 0..seq_len {
                let k_vec = cache.get_k_slice(j, kv_head_start, self.head_dim);
                let score = math::dot_product(
                    &q[q_head_start..q_head_start + self.head_dim],
                    k_vec,
                ) * scale;
                scores[j] = score;
            }

            // Softmax over scores
            math::softmax_inplace(&mut scores);

            // Weighted sum of V vectors
            let out_head_start = h * self.head_dim;
            for j in 0..seq_len {
                let v_vec = cache.get_v_slice(j, kv_head_start, self.head_dim);
                for d in 0..self.head_dim {
                    attn_output[out_head_start + d] += scores[j] * v_vec[d];
                }
            }
        }

        // --- Output projection: out = attn_output @ W_o^T ---
        math::mat_vec_mul_transposed(&self.wo, &attn_output, output, embed_dim, q_dim);
    }

    /// Get the GQA n_rep factor
    pub fn n_rep(&self) -> usize {
        self.n_rep
    }
}
