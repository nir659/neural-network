/* I wired layers and activations together through a flat vector of stages so
 * the forward pass is a single linear scan with no virtual dispatch — each
 * stage is a dense layer paired with an optional function-pointer activation. */
#pragma once

#include "dense.hpp"
#include "relu.hpp"
#include <vector>

using activation_fn = tensor (*)(const tensor &);

struct network {
  struct stage {
    dense_layer layer;
    activation_fn activation;
  };

  std::vector<stage> stages;

  void add(dense_layer layer, activation_fn act = nullptr);

  tensor forward(const tensor &input) const;
};
