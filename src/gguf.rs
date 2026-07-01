// ============================================================================
// gguf.rs — GGUF File Format Parser (Pure Rust, no external crates)
// ============================================================================
//
// GGUF Format Specification (v3):
// ┌─────────────────────────────────────────────────────────────┐
// │ Magic bytes:    "GGUF" (0x46554747)                4 bytes │
// │ Version:        u32 (3 for v3)                      4 bytes │
// │ Tensor count:   u64                                 8 bytes │
// │ Metadata count: u64                                 8 bytes │
// │ Metadata KV pairs: [key_string, value] × N_kv              │
// │ Tensor descriptors: [name, n_dims, dims, type, offset]     │
// │ Tensor data (aligned to multiples of 512 bytes)            │
// └─────────────────────────────────────────────────────────────┘
//
// Metadata value types:
//   0 = UINT8, 1 = INT8, 2 = UINT16, 3 = INT16,
//   4 = UINT32, 5 = INT32, 6 = FLOAT32, 7 = BOOL,
//   8 = STRING, 9 = ARRAY, 10 = UINT64, 11 = INT64, 12 = FLOAT64
//
// Tensor data types (GGMLType):
//   0 = F32, 1 = F16, 2 = Q4_0, 3 = Q4_1,
//   6 = Q5_0, 7 = Q5_1, 8 = Q8_0, 9 = Q8_1,
//   10 = Q2_K, 11 = Q3_K, 12 = Q4_K, 13 = Q5_K, 14 = Q6_K

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub const GGUF_VERSION: u32 = 3;
pub const ALIGNMENT: usize = 32; // GGUF v3 default alignment

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDataType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GgufDataType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            _ => None,
        }
    }

    /// Block size (number of elements per quantization block)
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
        }
    }

    /// Bytes per element (average for quantized types)
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 => 2.0,
            Self::Q4_0 => 18.0 / 32.0,
            Self::Q4_1 => 20.0 / 32.0,
            Self::Q5_0 => 22.0 / 32.0,
            Self::Q5_1 => 34.0 / 32.0,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q8_1 => 40.0 / 32.0,
            Self::Q2_K => 84.0 / 256.0,
            Self::Q3_K => 110.0 / 256.0,
            Self::Q4_K => 144.0 / 256.0,
            Self::Q5_K => 176.0 / 256.0,
            Self::Q6_K => 210.0 / 256.0,
        }
    }

    /// Bytes per block
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 34,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }
}

// ============================================================================
// Metadata Value
// ============================================================================

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn to_u8(&self) -> Option<u8> {
        match self {
            MetadataValue::UInt8(v) => Some(*v),
            MetadataValue::Int8(v) => Some(*v as u8),
            _ => None,
        }
    }

    pub fn to_i32(&self) -> Option<i32> {
        match self {
            MetadataValue::Int32(v) => Some(*v),
            MetadataValue::UInt32(v) => Some(*v as i32),
            MetadataValue::Int8(v) => Some(*v as i32),
            MetadataValue::UInt8(v) => Some(*v as i32),
            MetadataValue::Int16(v) => Some(*v as i32),
            MetadataValue::UInt16(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::UInt64(v) => Some(*v),
            MetadataValue::Int64(v) => Some(*v as u64),
            MetadataValue::UInt32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::Float64(v) => Some(*v),
            MetadataValue::Float32(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn to_string_ref(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn to_str(&self) -> Option<&str> {
        self.to_string_ref()
    }

    pub fn to_array(&self) -> Option<&Vec<MetadataValue>> {
        match self {
            MetadataValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor Descriptor
// ============================================================================

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: usize,
    pub dims: Vec<usize>,
    pub data_type: GgufDataType,
    pub offset: u64, // absolute offset in file
}

impl TensorInfo {
    /// Total number of elements in this tensor
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Total size in bytes of the raw tensor data
    pub fn n_bytes(&self) -> usize {
        (self.n_elements() as f64 * self.data_type.bytes_per_element()) as usize
    }
}

// ============================================================================
// GGUF File Structure
// ============================================================================

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub n_tensors: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64, // file offset where tensor data begins
}

impl GgufFile {
    /// Parse a GGUF file from disk
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        Self::parse(&mut file)
    }

    /// Parse a GGUF file from a reader
    pub fn parse<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        // --- Read magic bytes (4 bytes) ---
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid GGUF magic: 0x{:08X} (expected 0x{:08X})",
                    magic, GGUF_MAGIC
                ),
            ));
        }

        // --- Read version (4 bytes) ---
        let version = read_u32(reader)?;
        if version != GGUF_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unsupported GGUF version: {} (only v3 supported)",
                    version
                ),
            ));
        }

        // --- Read tensor count (8 bytes, little-endian u64) ---
        let n_tensors = read_u64(reader)?;

        // --- Read metadata count (8 bytes) ---
        let n_kv = read_u64(reader)?;

        // --- Parse metadata key-value pairs ---
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(reader)?;
            let value = read_metadata_value(reader)?;
            metadata.insert(key, value);
        }

        // --- Parse tensor descriptors ---
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(reader)?;
            let n_dims = read_u32(reader)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(reader)? as usize);
            }
            let dtype_id = read_u32(reader)?;
            let data_type = GgufDataType::from_u32(dtype_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown tensor data type: {}", dtype_id),
                )
            })?;
            let offset = read_u64(reader)?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                data_type,
                offset,
            });
        }

        // --- Calculate data section offset ---
        // Data is aligned to ALIGNMENT bytes after the current position
        let current_pos = reader.stream_position()?;
        let data_offset = align_to(current_pos, ALIGNMENT);

    

        Ok(GgufFile {
            version,
            n_tensors,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get tensor info by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }
}

// ============================================================================
// Alignment Helper
// ============================================================================

/// Align `value` up to the next multiple of `alignment`
pub fn align_to(value: u64, alignment: usize) -> u64 {
    let align = alignment as u64;
    (value + align - 1) / align * align
}

// ============================================================================
// Binary Reading Helpers (all little-endian)
// ============================================================================

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(r: &mut R) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(r: &mut R) -> io::Result<i16> {
    Ok(read_u16(r)? as i16)
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    Ok(read_u32(r)? as i32)
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> io::Result<i64> {
    Ok(read_u64(r)? as i64)
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    Ok(f64::from_bits(read_u64(r)?))
}

/// Read a length-prefixed UTF-8 string
fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8: {}", e)))
}

/// Read a metadata value (type-tagged union)
fn read_metadata_value<R: Read>(r: &mut R) -> io::Result<MetadataValue> {
    let type_id = read_u32(r)?;
    match type_id {
        0 => Ok(MetadataValue::UInt8(read_u8(r)?)),
        1 => Ok(MetadataValue::Int8(read_i8(r)?)),
        2 => Ok(MetadataValue::UInt16(read_u16(r)?)),
        3 => Ok(MetadataValue::Int16(read_i16(r)?)),
        4 => Ok(MetadataValue::UInt32(read_u32(r)?)),
        5 => Ok(MetadataValue::Int32(read_i32(r)?)),
        6 => Ok(MetadataValue::Float32(read_f32(r)?)),
        7 => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        8 => Ok(MetadataValue::String(read_string(r)?)),
        9 => {
            // Array: type of elements + count + elements
            let elem_type = read_u32(r)?;
            let n_elems = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(n_elems);
            for _ in 0..n_elems {
                let val = read_metadata_value_by_type(r, elem_type)?;
                arr.push(val);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::UInt64(read_u64(r)?)),
        11 => Ok(MetadataValue::Int64(read_i64(r)?)),
        12 => Ok(MetadataValue::Float64(read_f64(r)?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown metadata value type: {}", type_id),
        )),
    }
}

/// Read a metadata value when the type is already known (for arrays)
fn read_metadata_value_by_type<R: Read>(r: &mut R, type_id: u32) -> io::Result<MetadataValue> {
    match type_id {
        0 => Ok(MetadataValue::UInt8(read_u8(r)?)),
        1 => Ok(MetadataValue::Int8(read_i8(r)?)),
        2 => Ok(MetadataValue::UInt16(read_u16(r)?)),
        3 => Ok(MetadataValue::Int16(read_i16(r)?)),
        4 => Ok(MetadataValue::UInt32(read_u32(r)?)),
        5 => Ok(MetadataValue::Int32(read_i32(r)?)),
        6 => Ok(MetadataValue::Float32(read_f32(r)?)),
        7 => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        8 => Ok(MetadataValue::String(read_string(r)?)),
        10 => Ok(MetadataValue::UInt64(read_u64(r)?)),
        11 => Ok(MetadataValue::Int64(read_i64(r)?)),
        12 => Ok(MetadataValue::Float64(read_f64(r)?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown metadata array element type: {}", type_id),
        )),
    }
}

/// Read raw bytes from a file at a given offset
pub fn read_tensor_data(path: &Path, offset: u64, size: usize) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; size];
    file.read_exact(&mut buf)?;
    Ok(buf)
}
