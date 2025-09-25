#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

// Simple CUDA kernel for vector addition
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void cuda_test_function() {
    // Initialize cuDNN
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN initialization failed: " << cudnnGetErrorString(status) << std::endl;
        return;
    }
    std::cout << "cuDNN initialized successfully!" << std::endl;

    // --- cuDNN Activation Test (ReLU) ---
    // Input: 1 image, 1 channel, 4x4
    cudnnTensorDescriptor_t input_desc;
    status = cudnnCreateTensorDescriptor(&input_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create input descriptor: " << cudnnGetErrorString(status) << std::endl;
        cudnnDestroy(cudnn_handle);
        return;
    }
    status = cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to set input descriptor: " << cudnnGetErrorString(status) << std::endl;
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Activation descriptor: ReLU
    cudnnActivationDescriptor_t activation_desc;
    status = cudnnCreateActivationDescriptor(&activation_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to create activation descriptor: " << cudnnGetErrorString(status) << std::endl;
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    status = cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0f);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to set activation descriptor: " << cudnnGetErrorString(status) << std::endl;
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Allocate input/output memory
    float *d_input, *d_output;
    cudaError_t cuda_status;
    cuda_status = cudaMalloc(&d_input, 1 * 1 * 4 * 4 * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for input: " << cudaGetErrorString(cuda_status) << std::endl;
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    cuda_status = cudaMalloc(&d_output, 1 * 1 * 4 * 4 * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for output: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(d_input);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Initialize input with test data (mix of positive and negative for ReLU)
    float h_input[16] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -1.5f, 0.5f, 3.0f, -0.5f, 1.5f, 2.5f, -2.5f, 0.0f, 1.0f, -1.0f, 4.0f};
    cuda_status = cudaMemcpy(d_input, h_input, 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for input: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Verify input data
    float h_input_check[16];
    cuda_status = cudaMemcpy(h_input_check, d_input, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for input check: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    std::cout << "Input check (first element): " << h_input_check[0] << std::endl;

    // Perform ReLU activation
    float alpha = 1.0f, beta = 0.0f;
    status = cudnnActivationForward(cudnn_handle, activation_desc, &alpha, input_desc, d_input, &beta, input_desc, d_output);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "Activation failed: " << cudnnGetErrorString(status) << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    // Copy output back and print sample
    float h_output[16];
    cuda_status = cudaMemcpy(h_output, d_output, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for output: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(d_input); cudaFree(d_output);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    std::cout << "ReLU result (first few elements): " << h_output[0] << ", " << h_output[1] << ", " << h_output[2] << ", " << h_output[3] << std::endl;

    // --- CUDA Vector Add Test ---
    const int N = 1024;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    cuda_status = cudaMalloc(&d_a, N * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    cuda_status = cudaMalloc(&d_b, N * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_a);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    
    cuda_status = cudaMalloc(&d_c, N * sizeof(float));
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_a); cudaFree(d_b);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    cuda_status = cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    cuda_status = cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }

    vector_add<<<(N + 255)/256, 256>>>(d_a, d_b, d_c, N);

    cuda_status = cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for vector add: " << cudaGetErrorString(cuda_status) << std::endl;
        delete[] h_a; delete[] h_b; delete[] h_c;
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn_handle);
        return;
    }
    std::cout << "Vector add result (last element): " << h_c[N-1] << std::endl;

    // Cleanup
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroy(cudnn_handle);
}