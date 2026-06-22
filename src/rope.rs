// ============================================================================
// rope.rs — Rotary Position Embedding (Pure Rust)
// ============================================================================
//
// RoPE (Su et al., 2021) encodes positional information by rotating
// query and key vectors in the attention mechanism.
//
// Given dimension index i (0..dim/2):
//   theta_i = 1 / (base^(2*i / dim))
//   angle = position * theta_i
//
// For vector pair (x[2i], x[2i+1]):
//   x'[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
//   x'[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)
//
// The `base` parameter varies by model:
//   Llama 2: base = 10000.0
//   Llama 3: base = 500000.0
//   Mistral: base = 10000.0

// ============================================================================
// RoPE State
// ============================================================================

#[derive(Debug, Clone)]
pub struct RoPE {
    /// Precomputed frequency table: freqs[pos * half_dim + i] = theta_i
    /// where theta_i = 1 / (base^(2*i/dim))
    freqs_cos: Vec<f32>,
    freqs_sin: Vec<f32>,
    /// Number of positions precomputed
    max_pos: usize,
    /// Half dimension (dim/2)
    half_dim: usize,
}

impl RoPE {
    /// Create a new RoPE instance with precomputed frequencies.
    /// `dim`: head dimension (e.g., 128)
    /// `base`: frequency base (e.g., 10000.0)
    /// `max_seq_len`: maximum sequence length to support
    pub fn new(dim: usize, base: f32, max_seq_len: usize) -> Self {
        let half_dim = dim / 2;
        let mut freqs_cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut freqs_sin = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let exponent = (2 * i) as f32 / dim as f32;
                let theta = 1.0 / base.powf(exponent);
                let angle = pos as f32 * theta;
                freqs_cos.push(angle.cos());
                freqs_sin.push(angle.sin());
            }
        }

        RoPE {
            freqs_cos,
            freqs_sin,
            max_pos: max_seq_len,
            half_dim,
        }
    }

    /// Apply RoPE to a query or key vector in-place.
    /// `x`: vector of length `dim`
    /// `position`: token position
    pub fn apply(&self, x: &mut [f32], position: usize) {
        debug_assert_eq!(x.len(), self.half_dim * 2);

        if position >= self.max_pos {
            // Out of range: compute on the fly (fallback)
            let base = 10000.0f32;
            for i in 0..self.half_dim {
                let exponent = (2 * i) as f32 / (self.half_dim * 2) as f32;
                let theta = 1.0 / base.powf(exponent);
                let angle = position as f32 * theta;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = x[2 * i];
                let x1 = x[2 * i + 1];
                x[2 * i] = x0 * cos_a - x1 * sin_a;
                x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
            return;
        }

        let base_idx = position * self.half_dim;
        for i in 0..self.half_dim {
            let cos_a = self.freqs_cos[base_idx + i];
            let sin_a = self.freqs_sin[base_idx + i];
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos_a - x1 * sin_a;
            x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }

    /// Get cosine and sine values for a position (for batch application)
    pub fn get_freqs(&self, position: usize) -> Option<(&[f32], &[f32])> {
        if position >= self.max_pos {
            return None;
        }
        let base = position * self.half_dim;
        Some((
            &self.freqs_cos[base..base + self.half_dim],
            &self.freqs_sin[base..base + self.half_dim],
        ))
    }
}
