/* I record every layer input and pre-activation output during the forward pass
 * so the backward pass can reverse-iterate through the same stages, feeding
 * each cached tensor to the matching backward function without recomputation. */
#include "net.hpp"

void network::add(dense_layer layer, activation_fn act) {
    stages.push_back({std::move(layer), act});
}

forward_result network::forward(const tensor& input) const {
    forward_cache cache;
    cache.inputs.resize(stages.size());
    cache.zs.resize(stages.size());

    tensor x = input;
    for (size_t i = 0; i < stages.size(); ++i) {
        cache.inputs[i] = x;

        tensor z = stages[i].layer.forward(x);
        cache.zs[i] = z;

        x = stages[i].activation ? stages[i].activation(z) : z;
    }

    return {x, std::move(cache)};
}

std::vector<dense_layer::gradients> network::backward(
    const tensor& d_loss, const forward_cache& cache) const {

    std::vector<dense_layer::gradients> all_grads(stages.size());
    tensor d = d_loss;

    for (size_t i = stages.size(); i-- > 0;) {
        if (stages[i].activation) {
            d = relu_backward(d, cache.zs[i]);
        }

        all_grads[i] = stages[i].layer.backward(d, cache.inputs[i]);
        d = all_grads[i].d_input;
    }

    return all_grads;
}
