#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <Eigen/Dense>

void cuda_test_function();

int main() {
    std::cout << "Starting NeuroShiv test..." << std::endl;

    // Test CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA devices found: " << device_count << std::endl;

    // Test cuDNN
    cuda_test_function();

    // Test Eigen
    Eigen::MatrixXf mat(2, 2);
    mat << 1, 2, 3, 4;
    std::cout << "Eigen test matrix:\n" << mat << std::endl;

    // Test your classes (example)
    // Network net;
    // net.some_function();
    std::cout << "Test complete!" << std::endl;
    return 0;
}