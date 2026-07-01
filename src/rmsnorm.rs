use crate::math::rmsnorm;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        RmsNorm { weight, eps }
    }

    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        rmsnorm(input, &self.weight, output, self.eps);
    }

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
