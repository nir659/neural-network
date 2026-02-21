// Dataloader: handles batching and shuffling for supervised learning—randomises and returns minibatches efficiently.

#include "dataloader.hpp"
#include "rng.hpp"
#include <algorithm>
#include <numeric>

dataloader::dataloader(tensor x_data, tensor y_data, size_t b_size)
    : X(std::move(x_data)), Y(std::move(y_data)), batch_size(b_size) {
  indices.resize(X.shape[0]);
  std::iota(indices.begin(), indices.end(), 0);
  std::iota(indices.begin(), indices.end(), 0);
}

void dataloader::shuffle() { std::ranges::shuffle(indices, global_rng()); }

size_t dataloader::n_batches() const {
  return (X.shape[0] + batch_size - 1) / batch_size;
}

std::pair<tensor, tensor> dataloader::get_batch(size_t batch_idx) const {
  size_t start_r = batch_idx * batch_size;
  size_t end_r = std::min(start_r + batch_size, X.shape[0]);
  size_t current_batch_size = end_r - start_r;

  tensor batch_X({current_batch_size, X.shape[1]});
  tensor batch_Y({current_batch_size, Y.shape[1]});

  for (size_t i = 0; i < current_batch_size; ++i) {
    size_t orignal_r = indices[start_r + i];

    for (size_t j = 0; j < X.shape[1]; ++j) {
      batch_X[i, j] = X[orignal_r, j];
    }

    for (size_t j = 0; j < Y.shape[1]; ++j) {
      batch_Y[i, j] = Y[orignal_r, j];
    }
  }
  return {batch_X, batch_Y};
}