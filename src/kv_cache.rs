// ============================================================================
// kv_cache.rs — Key-Value Cache (Pure Rust)
// ============================================================================
//
// During autoregressive generation, we only need to compute attention for
// the new token. Previous K and V vectors are cached to avoid recomputation.
//
// Design:
//   - Keys and Vectors are stored in a flat buffer: [head_0_pos_0, head_0_pos_1, ...]
//   - For each layer, we store keys and values separately
//   - The cache grows by appending new token states
//   - Preallocated to max_seq_len for efficiency
//
// Memory layout per layer (for n_kv_heads × head_dim dimensions):
//   keys: [seq_0, seq_1, ..., seq_t] where each seq_i is a flat vector
//   values: same layout



// ============================================================================
// KV Cache
// ============================================================================

#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached keys: flat buffer of shape [max_seq_len, kv_dim]
    keys: Vec<f32>,
    /// Cached values: flat buffer of shape [max_seq_len, kv_dim]
    values: Vec<f32>,
    /// Current sequence length (number of tokens cached)
    seq_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// KV dimension (n_kv_heads × head_dim)
    kv_dim: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(kv_dim: usize, max_seq_len: usize) -> Self {
        let total_size = max_seq_len * kv_dim;
        KVCache {
            keys: vec![0.0f32; total_size],
            values: vec![0.0f32; total_size],
            seq_len: 0,
            max_seq_len,
            kv_dim,
        }
    }

    /// Append a new key vector to the cache
    pub fn append_k(&mut self, k: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert!(
            self.seq_len < self.max_seq_len,
            "KV cache overflow: seq_len={} >= max_seq_len={}",
            self.seq_len,
            self.max_seq_len
        );

        let offset = self.seq_len * self.kv_dim;
        self.keys[offset..offset + self.kv_dim].copy_from_slice(k);
    }

    /// Append a new value vector to the cache
    pub fn append_v(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.kv_dim);

        let offset = self.seq_len * self.kv_dim;
        self.values[offset..offset + self.kv_dim].copy_from_slice(v);

        // Increment seq_len after both K and V are stored
        self.seq_len += 1;
    }

    /// Get a slice of keys for a specific position and head dimension range
    /// Returns &[f32] of length `len` starting at `head_start`
    pub fn get_k_slice(&self, position: usize, head_start: usize, len: usize) -> &[f32] {
        debug_assert!(position < self.seq_len);
        let offset = position * self.kv_dim + head_start;
        &self.keys[offset..offset + len]
    }

    /// Get a slice of values for a specific position and head dimension range
    pub fn get_v_slice(&self, position: usize, head_start: usize, len: usize) -> &[f32] {
        debug_assert!(position < self.seq_len);
        let offset = position * self.kv_dim + head_start;
        &self.values[offset..offset + len]
    }

    /// Get the full key buffer for all positions
    pub fn all_keys(&self) -> &[f32] {
        &self.keys[..self.seq_len * self.kv_dim]
    }

    /// Get the full value buffer for all positions
    pub fn all_values(&self) -> &[f32] {
        &self.values[..self.seq_len * self.kv_dim]
    }

    /// Current sequence length (number of tokens stored)
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// KV dimension
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Maximum sequence length
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }

    /// Reset cache (clear all stored K/V)
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}
