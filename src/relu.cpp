/* I used std::max for the forward pass so the compiler can emit branchless
 * cmov/maxss instructions. The backward multiplies the upstream gradient by a
 * binary mask derived from the cached forward input — 1 where input > 0, 0
 * otherwise — which is the exact derivative of max(0, x). */
#include "relu.hpp"
#include <algorithm>

tensor relu(const tensor& x) {
    tensor result = x;
    for (size_t i = 0; i < result.size(); ++i) {
        result.data[i] = std::max(0.0f, result.data[i]);
    }
    return result;
}

tensor relu_backward(const tensor& d_out, const tensor& input_cache) {
    tensor grad = d_out;
    for (size_t i = 0; i < grad.size(); ++i) {
        grad.data[i] *= (input_cache.data[i] > 0.0f) ? 1.0f : 0.0f;
    }
    return grad;
}
