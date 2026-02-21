/* I made SGD a plain struct with a single step() that mutates network parameters
 * in-place — no virtual dispatch, no heap allocation, just a flat loop applying
 * parameter -= learning_rate * gradient over every weight and bias element. */
#pragma once

#include "net.hpp"

struct sgd {
    float learning_rate;

    explicit sgd(float lr) : learning_rate(lr) {}

    void step(network& net, const std::vector<dense_layer::gradients>& grads);
};
