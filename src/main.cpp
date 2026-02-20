/* I use this file as a scratch entry point for quick smoke tests — it will
 * eventually hold the training loop once all layers and loss functions are
 * wired together. */
#include "tensor.hpp"

int main() {
    tensor a({2, 3}, 1.0f);
    tensor b({3, 2}, 1.0f);
    tensor c = matmul(a, b);
    return 0;
}
