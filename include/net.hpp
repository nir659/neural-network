/* I wired layers and activations together through a flat vector of stages so
 * the forward pass is a single linear scan with no virtual dispatch — each
 * stage is a dense layer paired with an optional function-pointer activation.
 * The forward pass records every intermediate tensor into an execution cache
 * so the backward pass can reverse-iterate and compute exact gradients. */
#pragma once

#include "dense.hpp"
#include "relu.hpp"
#include <vector>

using activation_fn = tensor (*)(const tensor &);

struct forward_cache {
    std::vector<tensor> inputs;  // x entering each dense layer
    std::vector<tensor> zs;      // z = W·x + b, before activation
};

struct forward_result {
    tensor prediction;
    forward_cache cache;
};

struct network {
  struct stage {
    dense_layer layer;
    activation_fn activation;
  };

  std::vector<stage> stages;

  void add(dense_layer layer, activation_fn act = nullptr);

  forward_result forward(const tensor &input) const;

  std::vector<dense_layer::gradients> backward(
      const tensor& d_loss, const forward_cache& cache) const;
};
