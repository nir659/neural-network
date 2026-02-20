/* I kept the network forward pass as a simple fold over the stages vector —
 * each iteration applies one dense transform then its activation, passing the
 * result into the next stage with no intermediate heap allocation. */
#include "net.hpp"

void network::add(dense_layer layer, activation_fn act) {
  stages.push_back({std::move(layer), act});
}

tensor network::forward(const tensor &input) const {
  tensor x = input;
  for (const auto &stage : stages) {
    x = stage.layer.forward(x);
    if (stage.activation) {
      x = stage.activation(x);
    }
  }
  return x;
}
