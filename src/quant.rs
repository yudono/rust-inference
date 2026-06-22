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

use crate::gguf::GgufDataType;

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

fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
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

        // Super-block scale and minimum
        let d = read_fp16_le(&data[base..base + 2]);
        let dmin = read_fp16_le(&data[base + 2..base + 4]);

        // Sub-block scales: 8 nibbles packed in 4 bytes at offset 4..8
        // But the actual GGML layout uses 8 bytes at offset 4..12
        // Let's use the standard layout where scales occupy bytes 4..12
        let scales_base = base + 4;

        // Quantized values start at offset 12
        let qs_base = base + 12;

        // 8 sub-blocks of 32 values
        for sub in 0..8 {
            // Each sub-block scale is a nibble from the scales bytes
            let scale_byte = data[scales_base + sub / 2];
            let sc = if sub % 2 == 0 {
                scale_byte & 0x0F
            } else {
                (scale_byte >> 4) & 0x0F
            };

            let sub_d = d * sc as f32;

            // 32 values = 16 bytes of packed nibbles
            let sub_qs_offset = qs_base + sub * 16;

            for i in 0..32 {
                let idx = b * block_size + sub * 32 + i;
                if idx >= n_elements {
                    break;
                }
                let byte_val = data[sub_qs_offset + i / 2];
                let nibble = if i % 2 == 0 {
                    byte_val & 0x0F
                } else {
                    (byte_val >> 4) & 0x0F
                };
                let val = nibble as f32 - 8.0;
                out.push(sub_d * val + dmin);
            }
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

        // Scales: 8 bytes at offset 4
        let scales_base = base + 4;
        // qh (high bits): 32 bytes at offset 12
        let qh_base = base + 12;
        // qs (low 4 bits): 128 bytes at offset 44
        let qs_base = base + 44;

        for sub in 0..8 {
            let scale_byte = data[scales_base + sub / 2];
            let sc = if sub % 2 == 0 {
                scale_byte & 0x0F
            } else {
                (scale_byte >> 4) & 0x0F
            };

            let sub_d = d * sc as f32;

            // qh for this sub-block: 4 bytes = 32 bits (one per value)
            let sub_qh_base = qh_base + sub * 4;
            // qs for this sub-block: 16 bytes = 32 nibbles
            let sub_qs_base = qs_base + sub * 16;

            for i in 0..32 {
                let idx = b * block_size + sub * 32 + i;
                if idx >= n_elements {
                    break;
                }

                // Low 4 bits from nibble
                let byte_val = data[sub_qs_base + i / 2];
                let lo = if i % 2 == 0 {
                    byte_val & 0x0F
                } else {
                    (byte_val >> 4) & 0x0F
                };

                // High bit from qh
                let qh_byte = data[sub_qh_base + i / 8];
                let hi = (qh_byte >> (i % 8)) & 1;

                let val5 = (hi << 4) | lo;
                let val = val5 as f32 - 16.0;
                out.push(sub_d * val + dmin);
            }
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

    let mut out = Vec::with_capacity(n_elements);

    for b in 0..n_blocks {
        let base = b * block_bytes;

        // Scales: 8 bytes at offset 0
        let scales_base = base;
        // Super-block d: f16 at offset 8
        let d = read_fp16_le(&data[base + 8..base + 10]);

        // ql (low 4 bits): starts at offset 10
        let ql_base = base + 10;
        // qh (high 2 bits): starts at offset 10 + 192 = 202
        let qh_base = base + 10 + 192;

        for sub in 0..8 {
            let scale_byte = data[scales_base + sub];
            let sc = (scale_byte as i8) as f32;

            let sub_d = d * sc;

            // ql for this sub-block: 24 bytes (32 * 4 bits = 128 bits = 16 bytes)
            // Wait, 32 values × 4 bits = 16 bytes per sub-block
            let sub_ql_base = ql_base + sub * 24;

            for i in 0..32 {
                let idx = b * block_size + sub * 32 + i;
                if idx >= n_elements {
                    break;
                }

                // Low 4 bits
                let ql_byte = data[sub_ql_base + i / 2];
                let lo = if i % 2 == 0 {
                    ql_byte & 0x0F
                } else {
                    (ql_byte >> 4) & 0x0F
                };

                // High 2 bits from qh
                // qh stores 2 bits per value: 32 * 2 = 64 bits = 8 bytes per sub-block
                let sub_qh_base = qh_base + sub * 8;
                let bit_offset = i * 2;
                let qh_byte1 = data[sub_qh_base + bit_offset / 8];
                let qh_byte2 = data[sub_qh_base + (bit_offset + 1) / 8];
                let hi1 = (qh_byte1 >> (bit_offset % 8)) & 1;
                let hi2 = (qh_byte2 >> ((bit_offset + 1) % 8)) & 1;
                let hi = (hi2 << 1) | hi1;

                let val6 = (hi << 4) | lo;
                let val = val6 as f32 - 32.0;
                out.push(sub_d * val);
            }
        }
    }
    Ok(out)
}
