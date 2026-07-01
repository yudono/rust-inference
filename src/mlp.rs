use crate::math;
use crate::quant::QuantizedMatrix;

#[derive(Debug, Clone)]
pub struct Mlp {
    pub gate_proj: QuantizedMatrix,
    pub up_proj: QuantizedMatrix,
    pub down_proj: QuantizedMatrix,
    pub hidden_dim: usize,
    pub embed_dim: usize,
}

impl Mlp {
    pub fn new(
        gate_proj: QuantizedMatrix,
        up_proj: QuantizedMatrix,
        down_proj: QuantizedMatrix,
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

    pub fn forward(&self, x: &[f32], output: &mut [f32]) {
        let gate_dim = self.hidden_dim;
        let up_dim = self.hidden_dim;

        let mut gate = vec![0.0f32; gate_dim];
        self.gate_proj.mat_vec_mul(x, &mut gate);
        math::silu_inplace(&mut gate);

        let mut up = vec![0.0f32; up_dim];
        self.up_proj.mat_vec_mul(x, &mut up);

        let mut hidden = vec![0.0f32; self.hidden_dim];
        math::vec_mul(&gate, &up, &mut hidden);

        self.down_proj.mat_vec_mul(&hidden, output);
    }
}
