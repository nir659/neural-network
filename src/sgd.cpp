/* I iterate through every stage's weights and biases in lockstep with the
 * matching gradient entry, applying the vanilla SGD rule element-by-element
 * over the flat data vectors so there are no extra tensor allocations. */
#include "sgd.hpp"

void sgd::step(network& net, const std::vector<dense_layer::gradients>& grads) {
    for (size_t i = 0; i < net.stages.size(); ++i) {
        auto& w = net.stages[i].layer.weights.data;
        auto& b = net.stages[i].layer.biases.data;
        const auto& dw = grads[i].d_weights.data;
        const auto& db = grads[i].d_biases.data;

        for (size_t j = 0; j < w.size(); ++j) {
            w[j] -= learning_rate * dw[j];
        }

        for (size_t j = 0; j < b.size(); ++j) {
            b[j] -= learning_rate * db[j];
        }
    }
}
