use crate::attention::Attention;
use crate::kv_cache::KVCache;
use crate::math;
use crate::mlp::Mlp;
use crate::rmsnorm::RmsNorm;
use crate::rope::RoPE;

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub layer_idx: usize,
    pub attention_norm: RmsNorm,
    pub attention: Attention,
    pub ffn_norm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerLayer {
    pub fn forward(
        &self,
        x: &[f32],
        position: usize,
        rope: &RoPE,
        kv_cache: &mut KVCache,
        output: &mut [f32],
    ) {
        let embed_dim = x.len();

        let mut x_norm = vec![0.0f32; embed_dim];
        self.attention_norm.forward(x, &mut x_norm);

        let mut attn_out = vec![0.0f32; embed_dim];
        self.attention
            .forward(&x_norm, position, rope, kv_cache, &mut attn_out);

        output.copy_from_slice(x);
        math::vec_add_inplace(output, &attn_out);

        let mut x_norm2 = vec![0.0f32; embed_dim];
        self.ffn_norm.forward(output, &mut x_norm2);

        let mut ffn_out = vec![0.0f32; embed_dim];
        self.mlp.forward(&x_norm2, &mut ffn_out);

        math::vec_add_inplace(output, &ffn_out);
    }
}

#[derive(Debug, Clone)]
pub struct Transformer {
    pub token_embd: Vec<f32>,
    pub embed_dim: usize,
    pub vocab_size: usize,
    pub layers: Vec<TransformerLayer>,
    pub final_norm: RmsNorm,
    pub lm_head: Vec<f32>,
    pub max_seq_len: usize,
}

impl Transformer {
    pub fn forward(
        &self,
        token_id: usize,
        position: usize,
        rope: &RoPE,
        kv_caches: &mut [KVCache],
        logits: &mut [f32],
    ) {
        assert_eq!(kv_caches.len(), self.layers.len());

        let mut hidden = vec![0.0f32; self.embed_dim];
        for dim in 0..self.embed_dim {
            hidden[dim] = self.token_embd[self.embed_dim * token_id + dim];
        }

        let mut layer_output = vec![0.0f32; self.embed_dim];
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(&hidden, position, rope, &mut kv_caches[i], &mut layer_output);
            if position == 0 && cfg!(debug_assertions) {
                let hmin = layer_output.iter().cloned().fold(f32::MAX, f32::min);
                let hmax = layer_output.iter().cloned().fold(f32::MIN, f32::max);
                let mean = layer_output.iter().sum::<f32>() / layer_output.len() as f32;
                eprintln!("  [L{}] min={:9.4} max={:9.4} mean={:9.4} h0..3={:.4},{:.4},{:.4},{:.4}",
                    i, hmin, hmax, mean, layer_output[0], layer_output[1], layer_output[2], layer_output[3]);
            }
            std::mem::swap(&mut hidden, &mut layer_output);
        }
        self.final_norm.forward_inplace(&mut hidden);
        if position == 0 && cfg!(debug_assertions) {
            let hmin = hidden.iter().cloned().fold(f32::MAX, f32::min);
            let hmax = hidden.iter().cloned().fold(f32::MIN, f32::max);
            let mean = hidden.iter().sum::<f32>() / hidden.len() as f32;
            eprintln!("  [FNL] min={:9.4} max={:9.4} mean={:9.4} h0..3={:.4},{:.4},{:.4},{:.4}",
                hmin, hmax, mean, hidden[0], hidden[1], hidden[2], hidden[3]);
        }
        math::mat_vec_mul_transposed(&self.lm_head, &hidden, logits, self.vocab_size, self.embed_dim);
    }
}
