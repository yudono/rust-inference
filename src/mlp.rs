use crate::math;

#[derive(Debug, Clone)]
pub struct Mlp {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub hidden_dim: usize,
    pub embed_dim: usize,
}

impl Mlp {
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

    pub fn forward(&self, x: &[f32], output: &mut [f32]) {
        let gate_dim = self.hidden_dim;
        let up_dim = self.hidden_dim;

        let mut gate = vec![0.0f32; gate_dim];
        math::mat_vec_mul_transposed(&self.gate_proj, x, &mut gate, gate_dim, self.embed_dim);
        math::silu_inplace(&mut gate);

        let mut up = vec![0.0f32; up_dim];
        math::mat_vec_mul_transposed(&self.up_proj, x, &mut up, up_dim, self.embed_dim);

        let mut hidden = vec![0.0f32; self.hidden_dim];
        math::vec_mul(&gate, &up, &mut hidden);

        math::mat_vec_mul_transposed(&self.down_proj, &hidden, output, self.embed_dim, self.hidden_dim);
    }
}
