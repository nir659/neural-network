/* MSE loss implementation */
#include "loss.hpp"
#include <stdexcept>

float mse_forward(const tensor& predictions, const tensor& targets) {
    if (predictions.shape != targets.shape) {
        throw std::invalid_argument("Shape mismatch in mse_forward");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions.data[i] - targets.data[i];
        sum += diff * diff;
    }
    return sum / static_cast<float>(predictions.size());
}

tensor mse_backward(const tensor& predictions, const tensor& targets) {
    if (predictions.shape != targets.shape) {
        throw std::invalid_argument("Shape mismatch in mse_backward");
    }

    float scale = 2.0f / static_cast<float>(predictions.size());
    tensor grad = predictions;
    for (size_t i = 0; i < grad.size(); ++i) {
        grad.data[i] = (predictions.data[i] - targets.data[i]) * scale;
    }
    return grad;
}
