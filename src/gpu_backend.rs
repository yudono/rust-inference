use std::collections::HashMap;
use wgpu::*;

use crate::gguf::GgufDataType;

// ============================================================================
// GPU Compute Backend
// ============================================================================

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pipelines: HashMap<u32, ComputePipeline>,
    bind_group_layouts: HashMap<u32, BindGroupLayout>,
}

fn dtype_key(dtype: GgufDataType) -> u32 {
    match dtype {
        GgufDataType::F32 => 0,
        GgufDataType::F16 => 1,
        GgufDataType::Q8_0 => 2,
        GgufDataType::Q4_0 => 3,
        GgufDataType::Q5_0 => 4,
        GgufDataType::Q4_K => 5,
        GgufDataType::Q5_K => 6,
        GgufDataType::Q6_K => 7,
        _ => 99,
    }
}

fn shader_src(dtype: GgufDataType) -> &'static str {
    match dtype {
        GgufDataType::F32 => WGSL_F32,
        GgufDataType::F16 => WGSL_F16,
        GgufDataType::Q8_0 => WGSL_Q8_0,
        GgufDataType::Q4_0 => WGSL_Q4_0,
        GgufDataType::Q5_0 => WGSL_Q5_0,
        GgufDataType::Q4_K => WGSL_Q4_K,
        GgufDataType::Q5_K => WGSL_Q5_K,
        GgufDataType::Q6_K => WGSL_Q6_K,
        _ => WGSL_F32,
    }
}

// ============================================================================
// WGSL Compute Shaders
// ============================================================================

// F32 shader: direct read
const WGSL_F32: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params {
    cols: u32,
    rows: u32,
    total_blocks: u32,
    _pad: u32,
};

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x;
    if (flat >= params.cols * params.rows) { return; }
    let base = flat * 4u;
    let bits = u32(weight[base]) | (u32(weight[base+1u])<<8u) | (u32(weight[base+2u])<<16u) | (u32(weight[base+3u])<<24u);
    let val = bitcast<f32>(bits);
    let col = flat % params.cols;
    let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// F16 shader
const WGSL_F16: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params {
    cols: u32, rows: u32, total_blocks: u32, _pad: u32,
};

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x;
    if (flat >= params.cols * params.rows) { return; }
    let base = flat * 2u;
    let val = read_f16(base);
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q8_0 shader
const WGSL_Q8_0: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params {
    cols: u32, rows: u32, total_blocks: u32, _pad: u32,
};

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 32u; let pos = flat % 32u;
    let w_base = block * 34u;
    let d = read_f16(w_base);
    let q = bitcast<i8>(weight[w_base + 2u + pos]);
    let val = d * f32(q);
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q4_0 shader
const WGSL_Q4_0: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params {
    cols: u32, rows: u32, total_blocks: u32, _pad: u32,
};

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 32u; let pos = flat % 32u;
    let w_base = block * 18u;
    let d = read_f16(w_base);
    let byte_val = weight[w_base + 2u + pos / 2u];
    let nibble = select(byte_val >> 4u, byte_val & 0x0Fu, (pos & 1u) == 0u);
    let val = d * f32(i32(nibble) - 8);
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q5_0 shader
const WGSL_Q5_0: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params {
    cols: u32, rows: u32, total_blocks: u32, _pad: u32,
};

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 32u; let pos = flat % 32u;
    let w_base = block * 22u;
    let d = read_f16(w_base);
    let qh = u32(weight[w_base+2u])|(u32(weight[w_base+3u])<<8u)|(u32(weight[w_base+4u])<<16u)|(u32(weight[w_base+5u])<<24u);
    let qs_addr = w_base + 6u;
    var xl: u32;
    if (pos < 16u) { xl = u32(weight[qs_addr + pos] & 0x0Fu); }
    else { xl = u32(weight[qs_addr + pos - 16u] >> 4u); }
    let xh = ((qh >> pos) & 1u) << 4u;
    let val = d * f32(i32(xl | xh) - 16);
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q4_K shader
const WGSL_Q4_K: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { cols: u32, rows: u32, total_blocks: u32, _pad: u32, };

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

fn get_scale_min_q4k(w_base: u32, j: u32) -> vec2<f32> {
    let sa = w_base + 4u;
    let sc_b = weight[sa + j]; let m_b = weight[sa + j + 4u];
    var d: u32; var m: u32;
    if (j < 4u) { d = u32(sc_b & 63u); m = u32(m_b & 63u); }
    else {
        d = (u32(weight[sa+j+4u] & 0x0Fu)) | ((u32(weight[sa+j-4u] >> 6u)) << 4u);
        m = (u32(weight[sa+j+4u] >> 4u)) | ((u32(weight[sa+j] >> 6u)) << 4u);
    }
    return vec2(f32(d), f32(m));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 256u; let pos = flat % 256u;
    let w_base = block * 144u;
    let sub = pos / 32u; let in_sub = pos % 32u;
    let qs_off = sub * 32u + in_sub;
    let qs_addr = w_base + 16u + qs_off / 2u;
    let qs_byte = weight[qs_addr];
    let nibble = select(qs_byte >> 4u, qs_byte & 0x0Fu, (qs_off & 1u) == 0u);
    let sm = get_scale_min_q4k(w_base, sub);
    let d_scale = read_f16(w_base);
    let d_min = read_f16(w_base + 2u);
    let val = d_scale * sm.x * f32(nibble) - d_min * sm.y;
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q5_K shader
const WGSL_Q5_K: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { cols: u32, rows: u32, total_blocks: u32, _pad: u32, };

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

fn get_scale_min(base: u32, j: u32) -> vec2<f32> {
    let sa = base + 4u;
    let sc_b = weight[sa + j]; let m_b = weight[sa + j + 4u];
    var d: u32; var m: u32;
    if (j < 4u) { d = u32(sc_b & 63u); m = u32(m_b & 63u); }
    else {
        d = (u32(weight[sa+j+4u] & 0x0Fu)) | ((u32(weight[sa+j-4u] >> 6u)) << 4u);
        m = (u32(weight[sa+j+4u] >> 4u)) | ((u32(weight[sa+j] >> 6u)) << 4u);
    }
    return vec2(f32(d), f32(m));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 256u; let pos = flat % 256u;
    let w_base = block * 176u;
    let sub = pos / 32u; let in_sub = pos % 32u;
    let sm = get_scale_min(w_base, sub);
    let d_scale = read_f16(w_base); let d_min = read_f16(w_base + 2u);
    let qs_off = (sub/2u)*32u + in_sub;
    let qs_byte = weight[w_base + 48u + qs_off];
    let lo = select(qs_byte >> 4u, qs_byte & 0x0Fu, (qs_off & 1u) == 0u);
    let hi_bit = u32(weight[w_base + 16u + in_sub]) & (1u << (sub/2u));
    let hi = select(0u, 16u, hi_bit != 0u);
    let val = d_scale * sm.x * f32(lo + hi) - d_min * sm.y;
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

// Q6_K shader
const WGSL_Q6_K: &str = "
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u8>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Params { cols: u32, rows: u32, total_blocks: u32, _pad: u32, };

fn read_f16(base: u32) -> f32 {
    let lo = u32(weight[base]); let hi = u32(weight[base+1u]);
    let bits = lo | (hi << 8u);
    let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
    if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
    if (exp == 31u) { return 0.0; }
    let e = i32(exp) - 15;
    let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
    if (sign == 1u) { return -result; } return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flat = id.x; if (flat >= params.cols * params.rows) { return; }
    let block = flat / 256u; let pos = flat % 256u;
    let w_base = block * 210u;
    let sub = pos / 128u; let idx = pos % 128u;
    let in32 = idx % 32u; let sub2 = idx / 32u;
    let ql_off = sub * 64u + in32;
    let qh_off = sub * 32u + in32;
    let ql_b0 = u32(weight[w_base + ql_off]);
    let ql_b1 = u32(weight[w_base + ql_off + 32u]);
    let qh_b = u32(weight[w_base + 128u + qh_off]);
    var lo: u32; var hi: u32;
    if (sub2 == 0u) { lo = ql_b0 & 0x0Fu; hi = (qh_b >> 0u) & 3u; }
    else if (sub2 == 1u) { lo = ql_b1 & 0x0Fu; hi = (qh_b >> 2u) & 3u; }
    else if (sub2 == 2u) { lo = ql_b0 >> 4u; hi = (qh_b >> 4u) & 3u; }
    else { lo = ql_b1 >> 4u; hi = (qh_b >> 6u) & 3u; }
    let q = i32(lo | (hi << 4u)) - 32;
    let sc_byte = weight[w_base + 192u + sub2 * 2u + in32 / 16u];
    let sc = f32(i32(sc_byte));
    let d_val = read_f16(w_base + 208u);
    let val = d_val * sc * f32(q);
    let col = flat % params.cols; let row = flat / params.cols;
    atomicAdd(&out[row], val * x[col]);
}
";

impl GpuContext {
    /// Try to initialize GPU backend. Returns None if GPU not available.
    pub fn try_init() -> Option<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("gguf-infer GPU"),
                required_features: Features::empty(),
                required_limits: Limits::downlevel_defaults(),
            },
            None,
        ))
        .ok()?;

        Some(GpuContext {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        })
    }

    fn get_or_create_pipeline(&mut self, dtype: GgufDataType) -> Option<&ComputePipeline> {
        let key = dtype_key(dtype);
        if !self.pipelines.contains_key(&key) {
            let src = shader_src(dtype);
            let module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(&format!("shader_{}", key)),
                source: ShaderSource::Wgsl(src.into()),
            });

            let layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("layout_{}", key)),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("pipeline_layout_{}", key)),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

            let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("pipeline_{}", key)),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

            self.bind_group_layouts.insert(key, layout);
            self.pipelines.insert(key, pipeline);
        }
        self.pipelines.get(&key)
    }

    pub fn mat_vec_mul(
        &mut self,
        weight_data: &[u8],
        dtype: GgufDataType,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        let total = rows * cols;
        let bs = match dtype {
            GgufDataType::Q4_K | GgufDataType::Q5_K | GgufDataType::Q6_K => 256,
            GgufDataType::Q4_0 | GgufDataType::Q5_0 | GgufDataType::Q8_0 => 32,
            GgufDataType::F32 | GgufDataType::F16 => 1,
            _ => 256,
        };
        let n_blocks = (total + bs - 1) / bs;

        let pipeline = match self.get_or_create_pipeline(dtype) {
            Some(p) => p,
            None => return self.fallback_cpu(weight_data, dtype, x, out, rows, cols),
        };
        let layout = &self.bind_group_layouts[&dtype_key(dtype)];

        // Uniform buffer: [cols: u32, rows: u32, total_blocks: u32, _pad: u32]
        let params_raw: [u32; 4] = [cols as u32, rows as u32, n_blocks as u32, 0];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_raw));

        // Weight buffer (read-only storage)
        use wgpu::util::DeviceExt;
        let weight_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weight"),
            contents: weight_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Input buffer x
        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(x),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Output buffer
        let out_size = (rows * 4) as u64;
        let out_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for readback
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bind"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("mat_vec_mul"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n_blocks as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let out_f32: &[f32] = bytemuck::cast_slice(&data);
            out.copy_from_slice(&out_f32[..rows]);
            drop(data);
        }
        staging.unmap();
    }

    fn fallback_cpu(
        &self,
        weight_data: &[u8],
        dtype: GgufDataType,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        default_fallback_cpu(weight_data, dtype, x, out, rows, cols);
    }
}

fn default_fallback_cpu(
    weight_data: &[u8],
    dtype: GgufDataType,
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    use crate::quant;
    let f32_w = quant::dequantize(weight_data, dtype, rows * cols).unwrap_or_default();
    crate::math::mat_vec_mul_transposed(&f32_w, x, out, rows, cols);
}
