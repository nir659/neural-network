/* I centralized all randomness behind a single mt19937 instance so every layer
 * draws from the same stream and a single set_seed call is enough to make an
 * entire training run fully reproducible. */
#pragma once

#include <random>

std::mt19937& global_rng();
void set_seed(unsigned int seed);