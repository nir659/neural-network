/* throwing together the main training loop to tie all the pieces together */
#include "net.hpp"
#include "csv.hpp"
#include "dataloader.hpp"
#include "sgd.hpp"
#include "loss.hpp"
#include "rng.hpp"
#include <iostream>

int main() {
    set_seed(42);

    tensor raw_data = load_csv("data/mnist_train.csv");
    size_t num_samples = raw_data.shape[0];
    if (num_samples == 0 || raw_data.shape[1] != 785) {
        std::cerr << "Failed to load data/mnist_train.csv (got " << num_samples
                  << " rows, " << raw_data.shape[1] << " cols). Run from project root.\n";
        return 1;
    }

    tensor X({num_samples, 784}, 0.0f);
    tensor Y({num_samples, 10}, 0.0f);

    for (size_t i = 0; i < num_samples; ++i) {
        int label = static_cast<int>(raw_data[i, 0]);
        Y[i, label] = 1.0f;

        for (size_t j = 0; j < 784; ++j) {
            X[i, j] = raw_data[i, j + 1] / 255.0f;
        }
    }

    network net;
    net.add(dense_layer(784, 128), relu);
    net.add(dense_layer(128, 10), nullptr);

    sgd optimizer(0.01f);
    dataloader loader(X, Y, 32);

    for (int epoch = 1; epoch <= 10; ++epoch) {
        loader.shuffle();
        float epoch_loss = 0.0f;

        for (size_t b = 0; b < loader.n_batches(); ++b) {
            auto [batch_X, batch_Y] = loader.get_batch(b);

            forward_result result = net.forward(batch_X);

            float loss = mse_forward(result.prediction, batch_Y);
            epoch_loss += loss;

            tensor d_loss = mse_backward(result.prediction, batch_Y);
            auto grads = net.backward(d_loss, result.cache);
            optimizer.step(net, grads);
        }

        epoch_loss /= loader.n_batches();
        std::cout << "Epoch " << epoch << " loss: " << epoch_loss << std::endl;
    }

    return 0;
}
