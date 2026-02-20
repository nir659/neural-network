/* I kept the loss as a pair of free functions so it stays purely functional like
 * the activations no state, no class, just tensors in and scalars/tensors out. */
#pragma once

#include "tensor.hpp"

float mse_forward(const tensor& predictions, const tensor& targets);

tensor mse_backward(const tensor& predictions, const tensor& targets);
