/* I implemented ReLU as standalone free functions rather than a class the
 * forward pass is a pure map and the backward pass only needs the original
 * input to build the derivative mask, so there is no state to store. */
#pragma once

#include "tensor.hpp"

tensor relu(const tensor& x);

tensor relu_backward(const tensor& d_out, const tensor& input_cache);