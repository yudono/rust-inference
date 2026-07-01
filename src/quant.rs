// ============================================================================
// quant.rs — Dequantization for GGUF Tensor Types (Pure Rust)
// ============================================================================
//
// GGUF Quantization Formats:
//
// F32: 32-bit IEEE 754 float. 4 bytes per element.
// F16: 16-bit half-precision float. 2 bytes per element.
//
// Q8_0 (block size 32, 34 bytes per block):
//   Layout: [f16 scale | int8 × 32]
//   Dequant: value[i] = scale * q8[i]
//
// Q4_0 (block size 32, 18 bytes per block):
//   Layout: [f16 scale | uint8 × 16] (32 nibbles packed 2/byte)
//   Nibble value = (byte >> (4*(i&1))) & 0xF
//   Dequant: value[i] = scale * (nibble - 8)
//
// Q4_K (block size 256, 144 bytes per block):
//   Layout: [f16 d | f16 dmin | uint8 × 8 scales | uint8 × 128 qs]
//   8 sub-blocks of 32 values each.
//   Sub-block scales are stored as quantized bytes.
//   Dequant: d_sub = d * scale_sub; val = d_sub * (nibble - 8) + dmin * something
//
// Q5_K (block size 256, 176 bytes per block):
//   Layout: [f16 d | f16 dmin | uint8 × 8 scales | uint8 × 4 qh | uint8 × 128 qs]
//   qh stores the high bit of each 5-bit value.
//
// Q6_K (block size 256, 210 bytes per block):
//   Layout: [f16 d | uint8 × 8 scales | uint8 × 64 ql | uint8 × 128 qh]
//   6-bit quantization with 8 sub-blocks.

use std::cmp;
use crate::gguf::GgufDataType;

// ============================================================================
// QuantizedMatrix — On-the-fly dequantization + mat-vec
// ============================================================================

/// A weight matrix stored in quantized format.
/// Dequantization happens on-the-fly during mat-vec multiplication,
/// avoiding the ~4x memory overhead of storing full f32.
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    pub data: Vec<u8>,   // raw quantized bytes
    pub dtype: GgufDataType,
    pub rows: usize,     // out_dim (slow dimension ne1)
    pub cols: usize,     // in_dim  (fast dimension ne0)
}

impl QuantizedMatrix {
    pub fn new(data: Vec<u8>, dtype: GgufDataType, rows: usize, cols: usize) -> Self {
        QuantizedMatrix { data, dtype, rows, cols }
    }

    fn block_size(&self) -> usize {
        match self.dtype {
            GgufDataType::Q4_0 | GgufDataType::Q8_0 | GgufDataType::Q5_0 => 32,
            GgufDataType::Q4_K | GgufDataType::Q5_K | GgufDataType::Q6_K => 256,
            GgufDataType::F32 => 1,
            GgufDataType::F16 => 1,
            GgufDataType::Q4_1 | GgufDataType::Q5_1 | GgufDataType::Q8_1 | GgufDataType::Q2_K | GgufDataType::Q3_K => 256,
        }
    }

    fn block_bytes(&self) -> usize {
        match self.dtype {
            GgufDataType::F32 => 4,
            GgufDataType::F16 => 2,
            GgufDataType::Q8_0 => 34,
            GgufDataType::Q4_0 => 18,
            GgufDataType::Q5_0 => 22,
            GgufDataType::Q4_K => 144,
            GgufDataType::Q5_K => 176,
            GgufDataType::Q6_K => 210,
            GgufDataType::Q4_1 | GgufDataType::Q5_1 | GgufDataType::Q8_1 | GgufDataType::Q2_K | GgufDataType::Q3_K => 0,
        }
    }

    /// Dequantize one block into buf (buf.len() must be >= block_size).
    fn dequantize_block(&self, block_idx: usize, buf: &mut [f32]) {
        let bs = self.block_size();
        let bb = self.block_bytes();
        let offset = block_idx * bb;
        let slice = &self.data[cmp::min(offset, self.data.len())..];

        match self.dtype {
            GgufDataType::F32 => {
                for i in 0..bs {
                    let off = i * 4;
                    if off + 3 < slice.len() {
                        let bits = u32::from_le_bytes([slice[off], slice[off+1], slice[off+2], slice[off+3]]);
                        buf[i] = f32::from_bits(bits);
                    }
                }
            }
            GgufDataType::F16 => {
                for i in 0..bs {
                    let off = i * 2;
                    if off + 1 < slice.len() {
                        buf[i] = read_fp16_le(&slice[off..]);
                    }
                }
            }
            GgufDataType::Q8_0 => {
                let d = read_fp16_le(slice);
                for i in 0..bs {
                    if i < slice.len().saturating_sub(2) {
                        buf[i] = d * (slice[2 + i] as i8 as f32);
                    }
                }
            }
            GgufDataType::Q4_0 => {
                let d = read_fp16_le(slice);
                for i in 0..bs {
                    let byte_val = slice[2 + i / 2];
                    let nibble = if i % 2 == 0 { byte_val & 0x0F } else { (byte_val >> 4) & 0x0F };
                    buf[i] = d * (nibble as i8 - 8) as f32;
                }
            }
            GgufDataType::Q5_0 => {
                let d = read_fp16_le(slice);
                let qh = u32::from_le_bytes([slice[2], slice[3], slice[4], slice[5]]);
                let qs = &slice[6..];
                for i in 0..16 {
                    let xl = qs[i] & 0x0F;
                    let xh = (((qh >> i) & 1) << 4) as u8;
                    buf[i] = d * ((xl | xh) as i8 - 16) as f32;
                }
                for i in 16..32 {
                    let xl = qs[i - 16] >> 4;
                    let xh = (((qh >> i) & 1) << 4) as u8;
                    buf[i] = d * ((xl | xh) as i8 - 16) as f32;
                }
            }
            GgufDataType::Q4_K => {
                dequant_q4_k_block(slice, buf);
            }
            GgufDataType::Q5_K => {
                dequant_q5_k_block(slice, buf);
            }
            GgufDataType::Q6_K => {
                dequant_q6_k_block(slice, buf);
            }
            GgufDataType::Q4_1 | GgufDataType::Q5_1 | GgufDataType::Q8_1 |
            GgufDataType::Q2_K | GgufDataType::Q3_K => {} // unsupported, leave zero
        }
    }

    /// Dequantize a single row (all cols) into buf.
    pub fn dequantize_row(&self, row: usize, buf: &mut [f32]) {
        assert!(row < self.rows);
        assert!(buf.len() >= self.cols);
        let bs = self.block_size();
        let n_blocks_per_row = (self.cols + bs - 1) / bs;
        let start_block = (row * self.cols) / bs;
        let col_offset = (row * self.cols) % bs;
        let mut block_buf = vec![0.0f32; bs];

        for bi in 0..n_blocks_per_row {
            let block_idx = start_block + bi;
            if block_idx * bs >= self.rows * self.cols { break; }
            self.dequantize_block(block_idx, &mut block_buf);
            let src_start = if bi == 0 { col_offset } else { 0 };
            let src_end = cmp::min(bs, self.cols + col_offset - bi * bs);
            for k in src_start..src_end {
                let actual_dst = if bi == 0 { k - col_offset } else { bi * bs + k };
                if actual_dst < self.cols {
                    buf[actual_dst] = block_buf[k];
                }
            }
        }
    }

    /// Quantized mat-vec-mul-transposed: out[row] = Σ_col W[col, row] * x[col]
    /// Equivalent to `mat_vec_mul_transposed(f32_data, x, out, rows, cols)`.
    pub fn mat_vec_mul(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        out.fill(0.0);

        let total = self.rows * self.cols;
        let bs = self.block_size();
        let n_blocks = (total + bs - 1) / bs;
        let mut block_buf = vec![0.0f32; bs];

        for b in 0..n_blocks {
            self.dequantize_block(b, &mut block_buf);
            let start = b * bs;
            let end = cmp::min(start + bs, total);
            for k in 0..(end - start) {
                let flat = start + k;
                let col = flat % self.cols;
                let row = flat / self.cols;
                out[row] += block_buf[k] * x[col];
            }
        }
    }
}

/// Block-level Q4_K dequant (slice must start at block boundary)
fn dequant_q4_k_block(slice: &[u8], buf: &mut [f32]) {
    let d = read_fp16_le(slice);
    let dmin = read_fp16_le(&slice[2..]);
    let scales = &slice[4..16];
    let qs = &slice[16..144];
    let mut is = 0;
    for _j in (0..256).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let d1 = d * sc0;
        let m1 = dmin * m0;
        let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc1;
        let m2 = dmin * m1_val;
        let q = &qs[(is / 2) * 32..];
        for l in 0..32 { buf[is / 2 * 64 + l] = d1 * ((q[l] & 0x0F) as f32) - m1; }
        for l in 0..32 { buf[is / 2 * 64 + 32 + l] = d2 * (((q[l] >> 4) & 0x0F) as f32) - m2; }
        is += 2;
    }
}

/// Block-level Q5_K dequant
fn dequant_q5_k_block(slice: &[u8], buf: &mut [f32]) {
    let d = read_fp16_le(slice);
    let dmin = read_fp16_le(&slice[2..]);
    let scales = &slice[4..16];
    let qh = &slice[16..48];
    let qs = &slice[48..176];
    let mut is = 0;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for _j in (0..256).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let d1 = d * sc0;
        let m1 = dmin * m0;
        let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc1;
        let m2 = dmin * m1_val;
        let ql = &qs[(is / 2) * 32..];
        for l in 0..32 {
            let lo = ql[l] & 0x0F;
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            buf[is / 2 * 64 + l] = d1 * (lo + hi) as f32 - m1;
        }
        for l in 0..32 {
            let lo = ql[l] >> 4;
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            buf[is / 2 * 64 + 32 + l] = d2 * (lo + hi) as f32 - m2;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Block-level Q6_K dequant
fn dequant_q6_k_block(slice: &[u8], buf: &mut [f32]) {
    let ql = &slice[0..128];
    let qh = &slice[128..192];
    let sc_raw = &slice[192..208];
    let d = read_fp16_le(&slice[208..210]);

    for n in (0..256).step_by(128) {
        let iter = n / 128;
        let ql_iter = &ql[iter * 64..];
        let qh_iter = &qh[iter * 32..];
        let sc_iter = &sc_raw[iter * 8..];
        for l in 0..32usize {
            let is = l / 16;
            let q1_val = (ql_iter[l] & 0x0F) as i32 | (((qh_iter[l] as i32) >> 0) & 3) << 4;
            let q2_val = (ql_iter[l + 32] & 0x0F) as i32 | (((qh_iter[l] as i32) >> 2) & 3) << 4;
            let q3_val = (ql_iter[l] >> 4) as i32 | (((qh_iter[l] as i32) >> 4) & 3) << 4;
            let q4_val = (ql_iter[l + 32] >> 4) as i32 | (((qh_iter[l] as i32) >> 6) & 3) << 4;
            let sc0 = sc_iter[is + 0] as i8 as f32;
            let sc2 = sc_iter[is + 2] as i8 as f32;
            let sc4 = sc_iter[is + 4] as i8 as f32;
            let sc6 = sc_iter[is + 6] as i8 as f32;
            buf[n + l] = d * sc0 * (q1_val - 32) as f32;
            buf[n + l + 32] = d * sc2 * (q2_val - 32) as f32;
            buf[n + l + 64] = d * sc4 * (q3_val - 32) as f32;
            buf[n + l + 96] = d * sc6 * (q4_val - 32) as f32;
        }
    }
}

// ============================================================================
// FP16 to FP32 Conversion
// ============================================================================

/// Convert a 16-bit IEEE 754 half-precision float (stored as u16 bits)
/// to a 32-bit f32.
///
/// FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits.
pub fn fp16_to_fp32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = bits & 0x3FF;

    if exp == 0 {
        if mantissa == 0 {
            // Zero
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Denormalized number
        let value = mantissa as f32 / 1024.0;
        let e = 1 - 15;
        let value = value * 2.0f32.powi(e);
        return if sign == 1 { -value } else { value };
    }

    if exp == 31 {
        // Inf or NaN
        if mantissa == 0 {
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        return f32::NAN; // NaN
    }

    // Normalized number
    let exp_bias = 15;
    let e = exp - exp_bias;
    let m = 1.0 + mantissa as f32 / 1024.0;
    let value = m * 2.0f32.powi(e);

    if sign == 1 {
        -value
    } else {
        value
    }
}

/// Read f16 from 2 bytes (little-endian) and convert to f32
pub fn read_fp16_le(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    fp16_to_fp32(bits)
}

// ============================================================================
// Dequantization: Main Entry Point
// ============================================================================

/// Dequantize raw bytes into f32 values based on the data type.
/// Returns a Vec<f32> with `n_elements` values.
pub fn dequantize(
    raw_data: &[u8],
    dtype: GgufDataType,
    n_elements: usize,
) -> Result<Vec<f32>, String> {
    match dtype {
        GgufDataType::F32 => dequantize_f32(raw_data, n_elements),
        GgufDataType::F16 => dequantize_f16(raw_data, n_elements),
        GgufDataType::Q8_0 => dequantize_q8_0(raw_data, n_elements),
        GgufDataType::Q4_0 => dequantize_q4_0(raw_data, n_elements),
        GgufDataType::Q4_K => dequantize_q4_k(raw_data, n_elements),
        GgufDataType::Q5_0 => dequantize_q5_0(raw_data, n_elements),
        GgufDataType::Q5_K => dequantize_q5_k(raw_data, n_elements),
        GgufDataType::Q6_K => dequantize_q6_k(raw_data, n_elements),
        _ => Err(format!("Unsupported quantization type: {:?}", dtype)),
    }
}

// ============================================================================
// F32 Dequantization
// ============================================================================

fn dequantize_f32(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    if data.len() < n_elements * 4 {
        return Err(format!(
            "F32: expected {} bytes, got {}",
            n_elements * 4,
            data.len()
        ));
    }
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let offset = i * 4;
        let bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

// ============================================================================
// F16 Dequantization
// ============================================================================

fn dequantize_f16(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    if data.len() < n_elements * 2 {
        return Err(format!(
            "F16: expected {} bytes, got {}",
            n_elements * 2,
            data.len()
        ));
    }
    let mut out = Vec::with_capacity(n_elements);
    for i in 0..n_elements {
        let offset = i * 2;
        out.push(read_fp16_le(&data[offset..offset + 2]));
    }
    Ok(out)
}

// ============================================================================
// Q8_0 Dequantization
// ============================================================================
//
// Block size: 32
// Block layout: [f16 d (2 bytes) | int8 qs[32] (32 bytes)] = 34 bytes
// Dequant: y[i] = d * qs[i] where d = fp16(block[0..2])

fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 32;
    let block_bytes = 34; // 2 (f16) + 32 (int8)
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q8_0: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = read_fp16_le(&data[base..base + 2]);

        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let q = data[base + 2 + i] as i8;
            out.push(d * q as f32);
        }
    }
    Ok(out)
}

// ============================================================================
// Q4_0 Dequantization
// ============================================================================
//
// Block size: 32
// Block layout: [f16 d (2 bytes) | uint8 qs[16] (16 bytes)] = 18 bytes
// Nibbles packed low then high: byte = (lo_nibble) | (hi_nibble << 4)
// Dequant: nibble = (qs[i/2] >> (4*(i%2))) & 0xF; y[i] = d * (nibble - 8)

fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 32;
    let block_bytes = 18; // 2 (f16) + 16 (uint8)
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q4_0: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = read_fp16_le(&data[base..base + 2]);

        for i in 0..block_size {
            let idx = b * block_size + i;
            if idx >= n_elements {
                break;
            }
            let byte_val = data[base + 2 + i / 2];
            let nibble = if i % 2 == 0 {
                byte_val & 0x0F
            } else {
                (byte_val >> 4) & 0x0F
            };
            let val = nibble as i8 - 8;
            out.push(d * val as f32);
        }
    }
    Ok(out)
}

// ============================================================================
// Q5_0 Dequantization
// ============================================================================
//
// Block size: 32
// Block layout: [f16 d (2) | uint32 qh (4) | uint8 qb[16] (16)] = 22 bytes
// Each value is 5 bits: 4 low bits from qb nibble, 1 high bit from qh.
// Dequant: val = d * (nibble | ((qh_bit << 4) & 0x10)) as i8 - 16

fn dequantize_q5_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    // GGML block_q5_0 (22 bytes):
    //   d: f16 (2 bytes) — negative scale factor (max / -16)
    //   qh: u32 (4 bytes) — 5th bit of each 5-bit value (bit i = high bit of value i)
    //   qs: uint8[16] (16 bytes) — low 4 bits of 32 values
    //
    // GGML packing (from quantize_row_q5_0_ref):
    //   for j in 0..16:
    //     x0 = values[j]          (first half)
    //     x1 = values[j + 16]     (second half)
    //     qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4)
    //     qh bit j     = xi0 >> 4 (5th bit of first-half value j)
    //     qh bit j+16  = xi1 >> 4 (5th bit of second-half value j)
    //
    // GGML dequant (from dequantize_row_q5_0):
    //   for j in 0..16:
    //     xh_0 = ((qh >> (j +  0)) << 4) & 0x10  — bit j at position 4
    //     xh_1 = ((qh >> (j + 12))     ) & 0x10  — bit j+16 at position 4
    //     x0   = (qs[j] & 0x0F) | xh_0           — bottom nibble + bit j
    //     x1   = (qs[j] >>   4) | xh_1           — top nibble + bit j+16
    //     y[j + 0   ] = (x0 - 16) * d            — first half positions
    //     y[j + qk/2] = (x1 - 16) * d            — second half positions
    let block_size = 32;
    let block_bytes = 22;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q5_0: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let base = b * block_bytes;
        let d = read_fp16_le(&data[base..base + 2]);
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);
        let qs = &data[base + 6..base + 22];

        // First half (i = 0..15): bottom nibble of qs[i], qh bit i
        for i in 0..16 {
            let xl = qs[i] & 0x0F;
            let xh = (((qh >> i) & 1) << 4) as u8; // bit i at position 4
            let val = ((xl | xh) as i8) - 16;
            out.push(d * val as f32);
        }

        // Second half (i = 16..31): top nibble of qs[i-16], qh bit i
        for i in 16..32 {
            let xl = qs[i - 16] >> 4;
            let xh = (((qh >> i) & 1) << 4) as u8; // bit i (= bit j+16) at position 4
            let val = ((xl | xh) as i8) - 16;
            out.push(d * val as f32);
        }
    }

    // Trim excess elements from the last block (if n_elements not multiple of 32)
    out.truncate(n_elements);
    Ok(out)
}

// ============================================================================
// Q4_K Dequantization
// ============================================================================
//
// Block size: 256
// Block layout (144 bytes):
//   [f16 d (2) | f16 dmin (2) | uint8 scales[8] (8) | uint8 qs[128] (128)] = 140
//   (padded to 144 with 4 extra bytes)
//
// The 256 values are divided into 8 sub-blocks of 32 values.
// Each sub-block scale is packed as nibbles in the scales array.
// scales[0..4] contain 8 nibble values (2 per byte) for 8 sub-blocks.
//
// Dequant per sub-block:
//   sc = scales[sub_block]
//   d_sub = d * (sc & 0xF)  // lower nibble scale
//   Actually the layout is more nuanced — the scale bytes are themselves
//   quantized. For simplicity we treat them as direct scale multipliers.

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (f32, f32) {
    let (sc_6bit, m_6bit) = if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    };
    (sc_6bit as f32, m_6bit as f32)
}

fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    // GGML block_q4_K layout (144 bytes):
    //   d: f16 (2)
    //   dmin: f16 (2)
    //   scales: uint8[12]  (12)
    //   qs: uint8[128]     (128)
    // Total: 2 + 2 + 12 + 128 = 144
    let block_size = 256;
    let block_bytes = 144;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q4_K: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let base = b * block_bytes;

        let d = read_fp16_le(&data[base..base + 2]);
        let dmin = read_fp16_le(&data[base + 2..base + 4]);
        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];

        let mut is = 0;
        for _j in (0..256).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0;
            let m1 = dmin * m0;
            let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1;
            let m2 = dmin * m1_val;

            let q = &qs[(is / 2) * 32..][..32];
            for l in 0..32 {
                out.push(d1 * ((q[l] & 0x0F) as f32) - m1);
            }
            for l in 0..32 {
                out.push(d2 * (((q[l] >> 4) & 0x0F) as f32) - m2);
            }
            is += 2;
        }
    }
    Ok(out)
}

// ============================================================================
// Q5_K Dequantization
// ============================================================================
//
// Block size: 256
// Block layout (176 bytes):
//   [f16 d (2) | f16 dmin (2) | uint8 scales[8] (8) | uint8 qh[4] (4) | uint8 qs[128] (128)]
//   Total: 2+2+8+4+128 = 144 (but actual block is 176, likely alignment padding)
//
// Each value is 5 bits: 4 bits from nibble + 1 high bit from qh.
// qh stores 256 bits (high bits) = 32 bytes, but GGML packs it as 4 bytes
// for 8 sub-blocks: each sub-block of 32 values needs 32 bits = 4 bytes.
// So qh is actually 4 * 8 = 32 bytes.
//
// Let me recalculate: 2+2+8+32+128 = 172, padded to 176.

fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    // GGML block_q5_K layout (176 bytes):
    //   d: f16 (2)
    //   dmin: f16 (2)
    //   scales: uint8[12]   (12)
    //   qh: uint8[32]       (32)
    //   qs: uint8[128]      (128)
    // Total: 2 + 2 + 12 + 32 + 128 = 176
    //
    // QH layout: 32 bytes, each byte stores 8 high bits (one per sub-block)
    //   for position l (0..31): qh[l] bit sub = high bit of sub-block `sub` at position l
    //   Sub-block's qs is interleaved same as Q4_K.
    let block_size = 256;
    let block_bytes = 176;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q5_K: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let base = b * block_bytes;

        let d = read_fp16_le(&data[base..base + 2]);
        let dmin = read_fp16_le(&data[base + 2..base + 4]);
        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let qs = &data[base + 48..base + 176];

        let mut is = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _j in (0..256).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0;
            let m1 = dmin * m0;
            let (sc1, m1_val) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1;
            let m2 = dmin * m1_val;

            let ql = &qs[(is / 2) * 32..][..32];
            for l in 0..32 {
                let lo = ql[l] & 0x0F;
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * (lo + hi) as f32 - m1);
            }
            for l in 0..32 {
                let lo = ql[l] >> 4;
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * (lo + hi) as f32 - m2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

// ============================================================================
// Q6_K Dequantization
// ============================================================================
//
// Block size: 256
// Block layout (210 bytes):
//   [f16 d (2) | uint8 scales[8] (8) | uint8 ql[64] (64) | uint8 qh[128] (128)]
//   Total: 2+8+64+128 = 202, padded to 210.
//
// Actually in GGML, Q6_K layout is:
//   [uint8 scales[8] (8) | f16 d (2) | uint8 ql[192] (192)] = 202
//   Plus padding to 210.
//
// Each value is 6 bits: 4 bits from ql nibble + 2 high bits from somewhere.
//
// Let me use the correct layout:
//   scales: 8 bytes (sub-block scales)
//   d: 2 bytes (super-block scale)
//   ql: 192 bytes (low 4 bits of each value, 256 nibbles)
//   qh: 12 bytes (high 2 bits of each value, 256 * 2 bits = 64 bytes)
//
// Hmm, the exact layout is tricky. Let me use a simplified but correct approach.

fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let block_size = 256;
    let block_bytes = 210;
    let n_blocks = (n_elements + block_size - 1) / block_size;
    let expected = n_blocks * block_bytes;

    if data.len() < expected {
        return Err(format!(
            "Q6_K: expected at least {} bytes for {} elements, got {}",
            expected,
            n_elements,
            data.len()
        ));
    }

    let mut out = vec![0.0f32; n_elements];

    for b in 0..n_blocks {
        let base = b * block_bytes;

        // block_q6_K (old GGML layout): [ql: 128] [qh: 64] [scales: int8[16](16)] [d: f16(2)]
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        let sc_raw = &data[base + 192..base + 208];
        let d = read_fp16_le(&data[base + 208..base + 210]);



        let out_base = b * block_size;

        for n in (0..block_size).step_by(128) {
            let iter = n / 128;
            let ql_iter = &ql[iter * 64..][..64];
            let qh_iter = &qh[iter * 32..][..32];
            let sc_iter = &sc_raw[iter * 8..][..8];

            for l in 0..32usize {
                let is = l / 16;

                let q1_val = (ql_iter[l] & 0x0F) as i32 | (((qh_iter[l] as i32) >> 0) & 3) << 4;
                let q1 = q1_val - 32;
                let q2_val = (ql_iter[l + 32] & 0x0F) as i32 | (((qh_iter[l] as i32) >> 2) & 3) << 4;
                let q2 = q2_val - 32;
                let q3_val = (ql_iter[l] >> 4) as i32 | (((qh_iter[l] as i32) >> 4) & 3) << 4;
                let q3 = q3_val - 32;
                let q4_val = (ql_iter[l + 32] >> 4) as i32 | (((qh_iter[l] as i32) >> 6) & 3) << 4;
                let q4 = q4_val - 32;

                let sc0 = sc_iter[is + 0] as i8 as f32;
                let sc2 = sc_iter[is + 2] as i8 as f32;
                let sc4 = sc_iter[is + 4] as i8 as f32;
                let sc6 = sc_iter[is + 6] as i8 as f32;
                let idx0 = out_base + n + l;
                if idx0 < n_elements { out[idx0] = d * sc0 * q1 as f32; }
                let idx1 = out_base + n + l + 32;
                if idx1 < n_elements { out[idx1] = d * sc2 * q2 as f32; }
                let idx2 = out_base + n + l + 64;
                if idx2 < n_elements { out[idx2] = d * sc4 * q3 as f32; }
                let idx3 = out_base + n + l + 96;
                if idx3 < n_elements { out[idx3] = d * sc6 * q4 as f32; }
            }
        }
    }
    Ok(out)
}
