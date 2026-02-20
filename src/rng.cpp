/* I implemented a global RNG singleton to ensure consistent random distributions across all layers and data loaders. I added a set_seed method because deterministic initialization is strictly required to reproduce bugs and verify mathematical correctness during backpropagation testing. */
#include "rng.hpp"

std::mt19937& global_rng() {
  static std::mt19937 rnd(std::random_device{}());
  return rnd;
}

void set_seed(unsigned int seed) {
  global_rng().seed(seed);
}