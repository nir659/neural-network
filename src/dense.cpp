/* I used Xavier uniform initialization to keep the variance of activations
 * stable across layers, and I compute all three gradients in backward —
 * d_weights, d_biases, d_input — using transposed matmuls so the training
 * loop can update parameters and propagate the gradient chain in one call. */
#include "dense.hpp"
#include "rng.hpp" 
#include <cmath>
#include <random>

dense_layer::dense_layer(size_t in_features, size_t out_features)
    : weights({in_features, out_features}), biases({out_features}, 0.0f) {
    
    float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < weights.size(); ++i) {
        weights.data[i] = dist(global_rng());
    }
}

tensor dense_layer::forward(const tensor &input) const {
    tensor z = matmul(input, weights);
    add_bias(z, biases);
    return z;
}

dense_layer::gradients dense_layer::backward(const tensor &d_out, const tensor &input_cache) const {
    gradients grads;
    
    grads.d_weights = matmul(transpose(input_cache), d_out);
    
    grads.d_biases = tensor({d_out.shape[1]}, 0.0f);
    for (size_t i = 0; i < d_out.shape[0]; ++i) {
        for (size_t j = 0; j < d_out.shape[1]; ++j) {
            grads.d_biases[j] += d_out[i, j];
        }
    }
    
    grads.d_input = matmul(d_out, transpose(weights));
    
    return grads;
}