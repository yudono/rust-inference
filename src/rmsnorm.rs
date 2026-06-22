// ============================================================================
// rmsnorm.rs — Root Mean Square Layer Normalization (Pure Rust)
// ============================================================================
//
// RMSNorm formula:
//   y = x * (w / sqrt(mean(x^2) + eps))
//
// Where:
//   x = input vector
//   w = learned weight vector
//   eps = small constant for numerical stability (typically 1e-6)
//   mean(x^2) = (1/n) * sum(x_i^2)
//
// RMSNorm is used in LLaMA/LLaMA-2/LLaMA-3 instead of LayerNorm.
// It normalizes by the root mean square of the input, which is faster
// than LayerNorm (no mean subtraction needed).

use crate::math::rmsnorm;

// ============================================================================
// RMSNorm Layer
// ============================================================================

#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Learned weight vector (same dimension as input)
    pub weight: Vec<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
}

impl RmsNorm {
    /// Create a new RMSNorm layer
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        RmsNorm { weight, eps }
    }

    /// Apply RMSNorm to an input vector
    /// `input`: slice of length `dim`
    /// `output`: slice of length `dim` (must be pre-allocated)
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        rmsnorm(input, &self.weight, output, self.eps);
    }

    /// Apply RMSNorm in-place (modifies input directly)
    pub fn forward_inplace(&self, input: &mut [f32]) {
        let dim = input.len();
        let mut ss = 0.0f32;
        for i in 0..dim {
            ss += input[i] * input[i];
        }
        let rms = (ss / dim as f32 + self.eps).sqrt();
        let scale = 1.0 / rms;
        for i in 0..dim {
            input[i] = input[i] * scale * self.weight[i];
        }
    }
}
