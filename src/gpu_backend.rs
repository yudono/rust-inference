use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::*;

use crate::gguf::GgufDataType;

// ============================================================================
// GPU Compute Backend
// ============================================================================

type PipelineEntry = (Arc<ComputePipeline>, Arc<BindGroupLayout>);

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pipeline_cache: RefCell<HashMap<u32, PipelineEntry>>,
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

fn shader_src(dtype: GgufDataType) -> String {
    let body = match dtype {
        GgufDataType::F32 => WGSL_F32_BODY,
        GgufDataType::F16 => WGSL_F16_BODY,
        GgufDataType::Q8_0 => WGSL_Q8_0_BODY,
        GgufDataType::Q4_0 => WGSL_Q4_0_BODY,
        GgufDataType::Q5_0 => WGSL_Q5_0_BODY,
        GgufDataType::Q4_K => WGSL_Q4_K_BODY,
        GgufDataType::Q5_K => WGSL_Q5_K_BODY,
        GgufDataType::Q6_K => WGSL_Q6_K_BODY,
        _ => WGSL_F32_BODY,
    };
    format!("{}{}", WGSL_COMMON, body)
}

// ============================================================================
// WGSL Compute Shaders
// ============================================================================

const WGSL_COMMON: &str = "
struct Params { cols: u32, rows: u32, block_size: u32, _pad: u32, };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn read_u8(base: u32) -> u32 {
  let word = weight[base >> 2u];
  return (word >> ((base & 3u) * 8u)) & 0xFFu;
}
fn read_u16(base: u32) -> u32 {
  let word = weight[base >> 2u];
  return (word >> ((base & 3u) * 8u)) & 0xFFFFu;
}
fn read_f16_val(base: u32) -> f32 {
  let bits = read_u16(base);
  let sign = (bits >> 15u) & 1u; let exp = (bits >> 10u) & 31u; let mantissa = bits & 1023u;
  if (exp == 0u) { if (mantissa == 0u) { return 0.0; } return (f32(mantissa)/1024.0)*0.00006103515625; }
  if (exp == 31u) { return 0.0; }
  let e = i32(exp) - 15;
  let result = (1.0 + f32(mantissa)/1024.0) * pow(2.0, f32(e));
  if (sign == 1u) { return -result; } return result;
}
var<workgroup> wg_partial: array<f32, 256>;
";

// 2D dispatch: (block_size, ceiling(rows / block_size))
// Each workgroup computes one row (via shared memory reduction of 256 threads).
// Each thread handles columns at stride 256, accumulates in local var,
// then reduces via shared memory. row = wg_id.x + wg_id.y * params.block_size.
// IMPORTANT: use workgroup_id so all threads in a workgroup share the same row.

const WGSL_F32_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        sum += bitcast<f32>(weight[flat]) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_F16_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let base = (row * params.cols + col) * 2u;
        sum += read_f16_val(base) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q8_0_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 32u; let pos = flat % 32u;
        let d = read_f16_val(block * 34u);
        let q = i32(u32(read_u8(block * 34u + 2u + pos))) - 128;
        sum += d * f32(q) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q4_0_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 32u; let pos = flat % 32u;
        let w_base = block * 18u;
        let d = read_f16_val(w_base);
        let byte_val = read_u8(w_base + 2u + pos / 2u);
        let nibble = select(byte_val >> 4u, byte_val & 0x0Fu, (pos & 1u) == 0u);
        sum += d * f32(i32(nibble) - 8) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q5_0_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 32u; let pos = flat % 32u;
        let w_base = block * 22u;
        let d = read_f16_val(w_base);
        let qh = read_u8(w_base+2u)|(read_u8(w_base+3u)<<8u)|(read_u8(w_base+4u)<<16u)|(read_u8(w_base+5u)<<24u);
        let qs_addr = w_base + 6u;
        var xl: u32;
        if (pos < 16u) { xl = read_u8(qs_addr + pos) & 0x0Fu; }
        else { xl = read_u8(qs_addr + pos - 16u) >> 4u; }
        let xh = ((qh >> pos) & 1u) << 4u;
        sum += d * f32(i32(xl | xh) - 16) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q4_K_BODY: &str = "
fn get_scale_min_q4k(w_base: u32, j: u32) -> vec2<f32> {
    let sa = w_base + 4u;
    let sc_b = read_u8(sa + j); let m_b = read_u8(sa + j + 4u);
    var d: u32; var m: u32;
    if (j < 4u) { d = sc_b & 63u; m = m_b & 63u; }
    else {
        d = (read_u8(sa+j+4u) & 0x0Fu) | ((read_u8(sa+j-4u) >> 6u) << 4u);
        m = (read_u8(sa+j+4u) >> 4u) | ((read_u8(sa+j) >> 6u) << 4u);
    }
    return vec2(f32(d), f32(m));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 256u; let pos = flat % 256u;
        let w_base = block * 144u;
        let sub = pos / 32u; let in_sub = pos % 32u;
        let qs_base = (sub / 2u) * 32u + in_sub;
        let qs_byte = read_u8(w_base + 16u + qs_base);
        let is_even = (sub & 1u) == 0u;
        let nibble = select(qs_byte >> 4u, qs_byte & 0x0Fu, is_even);
        let sm = get_scale_min_q4k(w_base, sub);
        let d_scale = read_f16_val(w_base);
        let d_min = read_f16_val(w_base + 2u);
        sum += (d_scale * sm.x * f32(nibble) - d_min * sm.y) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q5_K_BODY: &str = "
fn get_scale_min(base: u32, j: u32) -> vec2<f32> {
    let sa = base + 4u;
    let sc_b = read_u8(sa + j); let m_b = read_u8(sa + j + 4u);
    var d: u32; var m: u32;
    if (j < 4u) { d = sc_b & 63u; m = m_b & 63u; }
    else {
        d = (read_u8(sa+j+4u) & 0x0Fu) | ((read_u8(sa+j-4u) >> 6u) << 4u);
        m = (read_u8(sa+j+4u) >> 4u) | ((read_u8(sa+j) >> 6u) << 4u);
    }
    return vec2(f32(d), f32(m));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 256u; let pos = flat % 256u;
        let w_base = block * 176u;
        let sub = pos / 32u; let in_sub = pos % 32u;
        let sm = get_scale_min(w_base, sub);
        let d_scale = read_f16_val(w_base); let d_min = read_f16_val(w_base + 2u);
        let qs_base = (sub / 2u) * 32u + in_sub;
        let qs_byte = read_u8(w_base + 48u + qs_base);
        let is_even = (sub & 1u) == 0u;
        let lo = select(qs_byte >> 4u, qs_byte & 0x0Fu, is_even);
        let hi_bit = read_u8(w_base + 16u + in_sub) & (1u << (sub/2u));
        let hi = select(0u, 16u, hi_bit != 0u);
        sum += (d_scale * sm.x * f32(lo + hi) - d_min * sm.y) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

const WGSL_Q6_K_BODY: &str = "
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = id.x + id.y * params.block_size; if (row >= params.rows) { return; }
    var sum = 0.0;
    for (var col = lid; col < params.cols; col += 256u) {
        let flat = row * params.cols + col;
        let block = flat / 256u; let pos = flat % 256u;
        let w_base = block * 210u;
        let sub = pos / 128u; let idx = pos % 128u;
        let in32 = idx % 32u; let sub2 = idx / 32u;
        let ql_off = sub * 64u + in32;
        let qh_off = sub * 32u + in32;
        let ql0 = read_u8(w_base + ql_off);
        let ql1 = read_u8(w_base + ql_off + 32u);
        let qh = read_u8(w_base + 128u + qh_off);
        var lo: u32; var hi: u32;
        if (sub2 == 0u) { lo = ql0 & 0x0Fu; hi = (qh >> 0u) & 3u; }
        else if (sub2 == 1u) { lo = ql1 & 0x0Fu; hi = (qh >> 2u) & 3u; }
        else if (sub2 == 2u) { lo = ql0 >> 4u; hi = (qh >> 4u) & 3u; }
        else { lo = ql1 >> 4u; hi = (qh >> 6u) & 3u; }
        let q = i32(lo | (hi << 4u)) - 32;
        let sc = read_u8(w_base + 192u + sub * 8u + sub2 * 2u + in32 / 16u);
        let d = read_f16_val(w_base + 208u);
        sum += d * (f32(i32(sc)) - 256.0 * f32(i32(sc >> 7u))) * f32(q) * x[col];
    }
    wg_partial[lid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) { wg_partial[lid] += wg_partial[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { out[row] = wg_partial[0]; }
}
";

impl GpuContext {
    /// Try to initialize GPU backend. Returns None if GPU not available.
    pub fn try_init() -> Option<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let required_limits = Limits {
            max_storage_buffer_binding_size: 1 << 30,
            ..Limits::downlevel_defaults()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                required_features: Features::empty(),
                required_limits,
                memory_hints: Default::default(),
            },
            None,
        ))
        .ok()?;

        Some(GpuContext {
            device,
            queue,
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }

    fn get_or_create_pipeline(&self, dtype: GgufDataType) -> Option<PipelineEntry> {
        let key = dtype_key(dtype);
        let mut cache = self.pipeline_cache.borrow_mut();
        if let Some(entry) = cache.get(&key) {
            return Some(entry.clone());
        }

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
                compilation_options: Default::default(),
                cache: None,
            });

        let entry = (Arc::new(pipeline), Arc::new(layout));
        cache.insert(key, entry.clone());
        Some(entry)
    }

    pub fn mat_vec_mul(
        &self,
        weight_data: &[u8],
        dtype: GgufDataType,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // Q6_K: fall back to CPU (WGSL dequant has GPU execution bug)
        if dtype == GgufDataType::Q6_K {
            return self.fallback_cpu(weight_data, dtype, x, out, rows, cols);
        }

        // Normal path for other dtypes (Q4_K, etc.)
        let (pipeline, layout) = match self.get_or_create_pipeline(dtype) {
            Some(p) => p,
            None => return self.fallback_cpu(weight_data, dtype, x, out, rows, cols),
        };

        self.do_mat_vec_mul(weight_data, dtype, x, out, rows, cols, &pipeline, &layout)
    }

    fn do_mat_vec_mul(
        &self,
        weight_data: &[u8],
        _dtype: GgufDataType,
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
    ) {
        let wg_limit = 65535u32;
        let dx = if rows as u32 > wg_limit { wg_limit } else { rows as u32 };
        let dy = (rows as u32 + dx - 1) / dx;

        let params_raw: [u32; 4] = [cols as u32, rows as u32, dx, 0];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_raw));

        use wgpu::util::DeviceExt;

        let weight_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weight"),
            contents: weight_data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let x_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(x),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let out_size = (rows * 4) as u64;
        let out_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("bind"),
            layout: &layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("mat_vec_mul"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(dx, dy, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

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
