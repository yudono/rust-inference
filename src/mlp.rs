// ============================================================================
// mlp.rs — Feed-Forward Network (SwiGLU) (Pure Rust)
// ============================================================================
//
// LLaMA/LLaMA-2/LLaMA-3 use a SwiGLU FFN:
//
//   gate = x @ W_gate^T
//   up   = x @ W_up^T
//
//   gate = SiLU(gate)        // SiLU(x) = x * sigmoid(x)
//   hidden = gate * up       // element-wise
//
//   down = hidden @ W_down^T // project back to embed_dim
//
// This is more expressive than a simple two-layer MLP with ReLU.
// The gate mechanism allows the network to selectively suppress
// or amplify different dimensions.

use crate::math;

// ============================================================================
// MLP Layer
// ============================================================================

#[derive(Debug, Clone)]
pub struct Mlp {
    /// Gate projection weight: (hidden_dim, embed_dim)
    pub gate_proj: Vec<f32>,
    /// Up projection weight: (hidden_dim, embed_dim)
    pub up_proj: Vec<f32>,
    /// Down projection weight: (embed_dim, hidden_dim)
    pub down_proj: Vec<f32>,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Embed dimension
    pub embed_dim: usize,
}

impl Mlp {
    /// Create a new MLP layer
    pub fn new(
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        hidden_dim: usize,
        embed_dim: usize,
    ) -> Self {
        Mlp {
            gate_proj,
            up_proj,
            down_proj,
            hidden_dim,
            embed_dim,
        }
    }

    /// Forward pass: output = down(SiLU(gate(x)) * up(x))
    ///
    /// Parameters:
    ///   `x`: input hidden state [embed_dim]
    ///   `output`: output buffer [embed_dim] (must be pre-allocated)
    pub fn forward(&self, x: &[f32], output: &mut [f32]) {
        let gate_dim = self.hidden_dim;
        let up_dim = self.hidden_dim;

        // --- Gate projection: gate = x @ W_gate^T ---
        let mut gate = vec![0.0f32; gate_dim];
        math::mat_vec_mul_transposed(&self.gate_proj, x, &mut gate, gate_dim, self.embed_dim);

        // --- Apply SiLU activation to gate ---
        math::silu_inplace(&mut gate);

        // --- Up projection: up = x @ W_up^T ---
        let mut up = vec![0.0f32; up_dim];
        math::mat_vec_mul_transposed(&self.up_proj, x, &mut up, up_dim, self.embed_dim);

        // --- Element-wise multiplication: hidden = gate * up ---
        let mut hidden = vec![0.0f32; self.hidden_dim];
        math::vec_mul(&gate, &up, &mut hidden);

        // --- Down projection: output = hidden @ W_down^T ---
        math::mat_vec_mul_transposed(&self.down_proj, &hidden, output, self.embed_dim, self.hidden_dim);
    }
}
