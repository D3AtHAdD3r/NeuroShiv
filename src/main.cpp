#include "Network_b.hpp"
#include "mnistLoader.hpp"
#include "utils.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

// helpers
std::string lossTypeToString(Network_b::LossType loss);
std::string neuronTypeToString(Network_b::NeuronType neuron);
void printNetworkParams(const std::string &label, const std::vector<int> &sizes, double l2strength, Network_b::LossType loss, Network_b::NeuronType neuron);
void print_training_params(double learning_rate, int epochs, int batch_size, double lambda);

//784→300→100→10 - ~98% on MNIST
int main()
{
    std::cout << "-----------program started----------\n";
    std::vector<int> sizes = {784, 30, 10};
    std::string train_images = "../assets/mnist_dataset/train-images-idx3-ubyte";
    std::string train_labels = "../assets/mnist_dataset/train-labels-idx1-ubyte";
    std::string test_images = "../assets/mnist_dataset/t10k-images-idx3-ubyte";
    std::string test_labels = "../assets/mnist_dataset/t10k-labels-idx1-ubyte";

    auto training_data = load_mnist_training(train_images, train_labels, 65000);
    if (training_data.empty())
    {
        throw std::runtime_error("Training data is empty. Please check the input files or path.");
    }

    auto test_data = load_mnist_test(test_images, test_labels, 10000);
    if (test_data.empty())
    {
        throw std::runtime_error("Test data is empty. Please check the input files or path.");
    }

    double l2strength = 0.001;
    Network_b::LossType loss = Network_b::LossType::CROSS_ENTROPY;
    Network_b::NeuronType neuron = Network_b::NeuronType::SIGMOID;

    Network_b net(sizes, l2strength, loss, neuron);

    int epochs = 10;
    int mini_batch_size = 64;
    double learning_rate = 1.5;

    printNetworkParams("GPU", sizes, l2strength, loss, neuron);
    print_training_params(learning_rate, epochs, mini_batch_size, l2strength);

    std::cout << "Training with Gpu-Batched context...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    net.SGD(training_data, epochs, mini_batch_size, learning_rate, &test_data, true);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::cout << "GPU training completed in " << gpu_duration.count() << " seconds.\n";

    return 0;
}

std::string lossTypeToString(Network_b::LossType loss)
{
    switch (loss)
    {
    case Network_b::LossType::MSE:
        return "Mean Squared Error";
    case Network_b::LossType::CROSS_ENTROPY:
        return "Cross Entropy";
    default:
        return "Unknown Loss";
    }
}

std::string neuronTypeToString(Network_b::NeuronType neuron)
{
    switch (neuron)
    {
    case Network_b::NeuronType::SIGMOID:
        return "Sigmoid";
        // case Network::NeuronType::RELU: return "ReLU";
        // case Network::NeuronType::TANH: return "Tanh";
    default:
        return "Unknown Neuron";
    }
}

void printNetworkParams(const std::string &label, const std::vector<int> &sizes, double l2strength,
                        Network_b::LossType loss, Network_b::NeuronType neuron)
{
    std::cout << "=== " << label << " Network Configuration ===\n";
    std::cout << "Layer sizes: ";
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        std::cout << sizes[i];
        if (i != sizes.size() - 1)
            std::cout << " -> ";
    }
    std::cout << "\nL2 Strength: " << l2strength << "\n";
    std::cout << "Loss function: " << lossTypeToString(loss) << "\n";
    std::cout << "Neuron type: " << neuronTypeToString(neuron) << "\n\n";
}

void print_training_params(double learning_rate, int epochs, int batch_size, double lambda) {
    std::cout << "Training Parameters ===" << std::endl;
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch Size: " << batch_size << std::endl;
    std::cout << "  Regularization (lambda): " << lambda << std::endl;
}
