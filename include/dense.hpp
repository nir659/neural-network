/* I separated the dense layer's parameter storage from its forward/backward
 * logic so that weights and biases live in contiguous flat tensors while the
 * execution functions operate statelessly on whatever input batch they receive. */
#pragma once

#include "tensor.hpp"
#include <cstddef>

struct dense_layer {
  tensor weights;
  tensor biases;

  dense_layer(size_t in_features, size_t out_features);

  tensor forward(const tensor &input) const;

  struct gradients {
    tensor d_weights;
    tensor d_biases;
    tensor d_input;
  };

  gradients backward(const tensor &d_out, const tensor &input_cache) const;
};