/* This file implements the core mathematical operations required for the
 * forward and backward passes. I structured the matrix multiplication loops to
 * iterate in i, k, j order, prioritizing row-major memory access to maximize
 * CPU cache hits and avoid branch prediction penalties. */
#include "tensor.hpp"
#include <stdexcept>

tensor operator+(const tensor &a, const tensor &b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument("Shape mismatch for element-wise addition");
  }

  tensor result = a;
  for (size_t i = 0; i < a.size(); ++i) {
    result.data[i] += b.data[i];
  }
  return result;
}

tensor operator*(const tensor &a, const tensor &b) {
  if (a.shape != b.shape) {
    throw std::invalid_argument(
        "Shape mismatch for element-wise multiplication");
  }

  tensor result = a;
  for (size_t i = 0; i < a.size(); ++i) {
    result.data[i] *= b.data[i];
  }
  return result;
}

tensor operator*(const tensor &a, float s) {
  tensor result = a;
  for (size_t i = 0; i < a.size(); ++i) {
    result.data[i] *= s;
  }
  return result;
}

tensor operator*(float s, const tensor &a) { return a * s; }

tensor matmul(const tensor &a, const tensor &b) {
  if (a.shape[1] != b.shape[0]) {
    throw std::invalid_argument("Dimension mismatch");
  }

  tensor result({a.shape[0], b.shape[1]}, 0.0f);

  for (size_t i = 0; i < a.shape[0]; ++i) {
    for (size_t k = 0; k < a.shape[1]; ++k) {
      float a_ik = a[i, k];
      for (size_t j = 0; j < b.shape[1]; ++j) {
        result[i, j] += a_ik * b[k, j];
      }
    }
  }
  return result;
}

tensor transpose(const tensor &a) {
  if (a.ndim() != 2) {
    throw std::invalid_argument("Transpose requires a 2D tensor");
  }

  size_t rows = a.shape[0];
  size_t cols = a.shape[1];
  tensor result({cols, rows});

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result[j, i] = a[i, j];
    }
  }
  return result;
}

void add_bias(tensor &z, const tensor &bias) {
  if (z.shape[1] != bias.shape[0]) {
    throw std::invalid_argument("Bias dimension mismatch");
  }

  for (size_t i = 0; i < z.shape[0]; ++i) {
    for (size_t j = 0; j < z.shape[1]; ++j) {
      z[i, j] += bias[j];
    }
  }
}