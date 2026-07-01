use crate::kv_cache::KVCache;
use crate::math;
use crate::rope::RoPE;

#[derive(Debug, Clone)]
pub struct Attention {
    pub layer_idx: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rep: usize,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
}

impl Attention {
    pub fn new(
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        bq: Vec<f32>,
        bk: Vec<f32>,
        bv: Vec<f32>,
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
            bq,
            bk,
            bv,
        }
    }

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

        let mut q = vec![0.0f32; q_dim];
        math::mat_vec_mul_transposed(&self.wq, x, &mut q, q_dim, embed_dim);
        math::vec_add_inplace(&mut q, &self.bq);

        let mut k = vec![0.0f32; kv_dim];
        math::mat_vec_mul_transposed(&self.wk, x, &mut k, kv_dim, embed_dim);
        math::vec_add_inplace(&mut k, &self.bk);

        let mut v = vec![0.0f32; kv_dim];
        math::mat_vec_mul_transposed(&self.wv, x, &mut v, kv_dim, embed_dim);
        math::vec_add_inplace(&mut v, &self.bv);

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

        cache.append_k(&k);
        cache.append_v(&v);

        let seq_len = cache.len();
        let mut attn_output = vec![0.0f32; q_dim];

        for h in 0..self.n_heads {
            let q_head_start = h * self.head_dim;
            let kv_head = h / self.n_rep;
            let kv_head_start = kv_head * self.head_dim;

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

            math::softmax_inplace(&mut scores);

            let out_head_start = h * self.head_dim;
            for j in 0..seq_len {
                let v_vec = cache.get_v_slice(j, kv_head_start, self.head_dim);
                for d in 0..self.head_dim {
                    attn_output[out_head_start + d] += scores[j] * v_vec[d];
                }
            }
        }

        math::mat_vec_mul_transposed(&self.wo, &attn_output, output, embed_dim, q_dim);
    }

    pub fn n_rep(&self) -> usize {
        self.n_rep
    }
}
