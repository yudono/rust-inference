// ============================================================================
// math.rs — Vector and Matrix Math Operations (Pure Rust)
// ============================================================================
//
// All operations work on f32 slices. No SIMD intrinsics — uses scalar code
// that the compiler may auto-vectorize. Design is straightforward and
// cache-friendly.

// ============================================================================
// Vector Operations
// ============================================================================

/// Element-wise addition: out[i] = a[i] + b[i]
pub fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "vec_add: length mismatch");
    assert_eq!(a.len(), out.len(), "vec_add: output length mismatch");
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// In-place element-wise addition: a[i] += b[i]
pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "vec_add_inplace: length mismatch");
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Element-wise multiplication: out[i] = a[i] * b[i]
pub fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "vec_mul: length mismatch");
    assert_eq!(a.len(), out.len(), "vec_mul: output length mismatch");
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

/// In-place element-wise multiplication: a[i] *= b[i]
pub fn vec_mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "vec_mul_inplace: length mismatch");
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Scalar-vector multiplication: out[i] = a[i] * s
pub fn vec_scale(a: &[f32], s: f32, out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * s;
    }
}

/// In-place scalar multiplication: a[i] *= s
pub fn vec_scale_inplace(a: &mut [f32], s: f32) {
    for val in a.iter_mut() {
        *val *= s;
    }
}

/// Dot product: sum(a[i] * b[i])
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: length mismatch");
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Euclidean norm (L2 norm)
pub fn vec_norm(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt()
}

/// Sum of elements
pub fn vec_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

/// Softmax over a slice: out[i] = exp(a[i]) / sum(exp(a[j]))
pub fn softmax(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len(), "softmax: length mismatch");
    if a.is_empty() {
        return;
    }

    // For numerical stability, subtract the max
    let max_val = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut exp_sum = 0.0f32;
    for i in 0..a.len() {
        out[i] = (a[i] - max_val).exp();
        exp_sum += out[i];
    }
    let inv_sum = 1.0 / exp_sum;
    for val in out.iter_mut() {
        *val *= inv_sum;
    }
}

/// Softmax in-place
pub fn softmax_inplace(a: &mut [f32]) {
    if a.is_empty() {
        return;
    }
    let max_val = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for val in a.iter_mut() {
        *val = (*val - max_val).exp();
        exp_sum += *val;
    }
    let inv_sum = 1.0 / exp_sum;
    for val in a.iter_mut() {
        *val *= inv_sum;
    }
}

/// SiLU activation: out[i] = a[i] * sigmoid(a[i]) = a[i] / (1 + exp(-a[i]))
pub fn silu(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = silu_single(a[i]);
    }
}

/// In-place SiLU
pub fn silu_inplace(a: &mut [f32]) {
    for val in a.iter_mut() {
        *val = silu_single(*val);
    }
}

/// Single element SiLU
#[inline]
pub fn silu_single(x: f32) -> f32 {
    // Numerically stable: x / (1 + exp(-x))
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        x / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        x * exp_pos / (1.0 + exp_pos)
    }
}

/// ReLU activation: out[i] = max(0, a[i])
pub fn relu(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].max(0.0);
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C = A × B
/// A: (M, K), B: (K, N), C: (M, N)
/// All stored in row-major order.
pub fn mat_mul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    // Zero output
    c.fill(0.0);

    // Standard ijk loop — compiler may vectorize the inner k-loop
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            let a_row = &a[i * k..(i + 1) * k];
            for l in 0..k {
                sum += a_row[l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Matrix-vector multiply for GGUF weight layout
/// GGUF stores 2D tensors with dims [ne0, ne1] in column-major order:
///   data[i0 + ne0 * i1] = element at (i0, i1)
/// For a weight matrix mapping in_dim -> out_dim:
///   ne0 = in_dim (fast/inner dimension), ne1 = out_dim (slow/outer dimension)
///   weight[i_in + in_dim * i_out] = W[i_out, i_in]
/// This computes: out[i] = Σ_j x[j] * weight[j + in_dim * i]
pub fn mat_vec_mul_transposed(
    weight: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert_eq!(weight.len(), in_dim * out_dim);
    assert_eq!(x.len(), in_dim);
    assert_eq!(out.len(), out_dim);

    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += weight[j + in_dim * i] * x[j];
        }
        out[i] = sum;
    }
}

/// Matrix-vector multiply for standard row-major storage W[out_dim, in_dim]
pub fn mat_vec_mul(
    weight: &[f32],
    x: &[f32],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert_eq!(weight.len(), out_dim * in_dim);
    assert_eq!(x.len(), in_dim);
    assert_eq!(y.len(), out_dim);
    for i in 0..out_dim {
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += weight[i * in_dim + j] * x[j];
        }
        y[i] = sum;
    }
}

// ============================================================================
// RMSNorm
// ============================================================================

/// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
/// Normalizes across the entire input vector.
pub fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(w.len(), n);
    assert_eq!(out.len(), n);

    // Compute sum of squares
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }

    // Root mean square
    let rms = (ss / n as f32 + eps).sqrt();
    let scale = 1.0 / rms;

    // Normalize and apply weights
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

/// RMSNorm in-place (modifies x directly)
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(w.len(), n);

    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    let rms = (ss / n as f32 + eps).sqrt();
    let scale = 1.0 / rms;

    for i in 0..n {
        x[i] = x[i] * scale * w[i];
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

/// Apply RoPE to a query or key vector at a given position.
/// `freqs` should be precomputed as:
///   freqs[i] = 1 / (base^(2i/dim)) for i in 0..dim/2
///
/// For each pair (x[2i], x[2i+1]) at position pos:
///   angle = pos * freqs[i]
///   x'[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
///   x'[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)
pub fn apply_rope(x: &mut [f32], position: usize, freqs: &[f32]) {
    let dim = x.len();
    assert_eq!(
        freqs.len(),
        dim / 2,
        "freqs must have dim/2 elements"
    );

    for i in 0..dim / 2 {
        let angle = position as f32 * freqs[i];
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_a - x1 * sin_a;
        x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

/// Precompute RoPE frequency table for all positions up to max_pos
/// base: frequency base (typically 10000.0 or 500000.0 for Llama 3)
/// dim: dimension of each query/key head
/// Returns a flat array of shape (max_pos, dim/2)
pub fn precompute_rope_freqs(max_pos: usize, dim: usize, base: f32) -> Vec<f32> {
    let half_dim = dim / 2;
    let mut freqs = vec![0.0f32; max_pos * half_dim];
    for pos in 0..max_pos {
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / dim as f32;
            let freq = 1.0 / base.powf(exponent);
            freqs[pos * half_dim + i] = freq;
        }
    }
    freqs
}

// ============================================================================
// Argmax
// ============================================================================

/// Return the index of the maximum value in a slice
pub fn argmax(a: &[f32]) -> usize {
    assert!(!a.is_empty(), "argmax: empty slice");
    let mut max_idx = 0;
    let mut max_val = a[0];
    for i in 1..a.len() {
        if a[i] > max_val {
            max_val = a[i];
            max_idx = i;
        }
    }
    max_idx
}

/// Top-K indices and values
pub fn top_k(a: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = a.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    // Partial sort: we only need the top k
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}
