#pragma once

#include "tensor.hpp"
#include <vector>
#include <utility>
#include <cstddef>

struct dataloader {
    tensor X;
    tensor Y;
    size_t batch_size;
    std::vector<size_t> indices;

    dataloader(tensor x_data, tensor y_data, size_t b_size);

    void shuffle();
    size_t n_batches() const;
    std::pair<tensor, tensor> get_batch(size_t batch_idx) const;
};