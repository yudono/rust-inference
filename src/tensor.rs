// ============================================================================
// tensor.rs — Tensor Data Structure and Operations (Pure Rust)
// ============================================================================
//
// Design rationale:
//   - Tensor stores dequantized f32 data for simplicity and correctness.
//   - Quantized weights are dequantized on load; during inference, all
//     computation happens in f32. This trades memory for simplicity.
//   - Shape is stored as Vec<usize> for flexibility with arbitrary ranks.
//   - Data is a flat Vec<f32> for cache-friendly access.

use crate::gguf::GgufDataType;

// ============================================================================
// Tensor Struct
// ============================================================================

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor with zeros
    pub fn zeros(name: &str, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: vec![0.0f32; n],
        }
    }

    /// Create a tensor from existing data
    pub fn new(name: &str, shape: &[usize], data: Vec<f32>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Shape {:?} expects {} elements but got {}",
            shape,
            expected,
            data.len()
        );
        Tensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data,
        }
    }

    /// Total number of elements
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get value at flat index
    pub fn get(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Set value at flat index
    pub fn set(&mut self, idx: usize, val: f32) {
        self.data[idx] = val;
    }

    /// Get value by multi-dimensional index (row-major layout)
    pub fn get_indexed(&self, indices: &[usize]) -> f32 {
        assert_eq!(indices.len(), self.ndim());
        let flat = self.compute_flat_index(indices);
        self.data[flat]
    }

    /// Set value by multi-dimensional index
    pub fn set_indexed(&mut self, indices: &[usize], val: f32) {
        assert_eq!(indices.len(), self.ndim());
        let flat = self.compute_flat_index(indices);
        self.data[flat] = val;
    }

    /// Compute flat index from multi-dimensional indices (row-major)
    fn compute_flat_index(&self, indices: &[usize]) -> usize {
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..self.ndim()).rev() {
            flat += indices[i] * stride;
            stride *= self.shape[i];
        }
        flat
    }

    /// Get a row (2D tensor): returns a slice for row `i`
    pub fn row(&self, i: usize) -> &[f32] {
        assert!(self.ndim() >= 2, "row() requires at least 2D tensor");
        let cols = self.shape[self.ndim() - 1];
        let start = i * cols;
        &self.data[start..start + cols]
    }

    /// Get a mutable row (2D tensor)
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        assert!(self.ndim() >= 2, "row_mut() requires at least 2D tensor");
        let cols = self.shape[self.ndim() - 1];
        let start = i * cols;
        &mut self.data[start..start + cols]
    }

    /// Get a slice from the data buffer
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get a mutable slice from the data buffer
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Create a view as a 2D matrix (rows × cols)
    pub fn as_2d(&self) -> (&[f32], usize, usize) {
        assert!(self.ndim() >= 2, "as_2d() requires at least 2D tensor");
        let rows = self.shape[0];
        let cols: usize = self.shape[1..].iter().product();
        (&self.data, rows, cols)
    }

    /// Get the data type size in bytes for this tensor's element type
    pub fn element_size_bytes(dtype: GgufDataType) -> usize {
        match dtype {
            GgufDataType::F32 => 4,
            GgufDataType::F16 => 2,
            _ => 1, // quantized: use bytes_per_block / block_size for average
        }
    }
}

// ============================================================================
// Shape Utilities
// ============================================================================

/// Compute total elements from shape
pub fn shape_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Compute stride for each dimension (row-major)
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Validate that a multi-dimensional index is within bounds
pub fn validate_index(shape: &[usize], index: &[usize]) -> bool {
    if shape.len() != index.len() {
        return false;
    }
    shape.iter().zip(index.iter()).all(|(s, i)| i < s)
}
