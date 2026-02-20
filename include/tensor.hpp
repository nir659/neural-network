/* I designed this tensor struct to serve as the foundational data structure for
 * the neural network, ensuring contiguous memory layout for optimal cache
 * performance. */
#pragma once

#include <initializer_list>
#include <numeric>
#include <span>
#include <vector>

struct tensor {
  std::vector<float> data;
  std::vector<size_t> shape;

  tensor() = default;

  explicit tensor(std::initializer_list<size_t> dims, float fill_value = 0.0f)
      : shape(dims) {
    size_t n =
        std::reduce(shape.begin(), shape.end(), 1uz, std::multiplies<>());
    data.assign(n, fill_value);
  }

  size_t size() const { return data.size(); }
  size_t ndim() const { return shape.size(); }
  bool empty() const { return data.empty(); }

  float *ptr() { return data.data(); }
  const float *ptr() const { return data.data(); }

  std::span<float> span() { return data; }
  std::span<const float> span() const { return data; }

  float &operator[](size_t i) { return data[i]; }
  float operator[](size_t i) const { return data[i]; }

  float &operator[](size_t row, size_t col) {
    return data[row * shape[1] + col];
  }
  float operator[](size_t row, size_t col) const {
    return data[row * shape[1] + col];
  }
};

// element-wise ops
tensor operator+(const tensor &a, const tensor &b);
tensor operator*(const tensor &a, const tensor &b);
tensor operator*(const tensor &a, float s);
tensor operator*(float s, const tensor &a);

// matrix ops
tensor matmul(const tensor &a, const tensor &b);
tensor transpose(const tensor &a);
void add_bias(tensor &z, const tensor &bias);