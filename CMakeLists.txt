cmake_minimum_required(VERSION 3.18)
project(NeuroShiv LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package (this sets up NVCC for .cu files)
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found! Ensure CUDA_PATH is set.")
endif()

# Include directories for CUDA, cuDNN, and Eigen
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}  # From find_package(CUDA)
    #$ENV{CUDA_PATH}/include
    $ENV{CUDNN_PATH_include}
    ${CMAKE_SOURCE_DIR}/extern/eigen
)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Link directories for CUDA and cuDNN libraries
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64  # Standard CUDA libs
    #$ENV{CUDA_PATH}/lib/x64
    $ENV{CUDNN_PATH_lib}
)

# Add executable: Compile main.cpp and cuda_test.cu into an exe
add_executable(NeuroShiv
    src/main.cpp
    src/cuda_test.cu
    src/cuda_kernels.cu
    src/utils.cpp
    src/GPUPass.cu
    src/Layer_b.cpp
    src/Network_b.cpp
)

# Link libraries: CUDA runtime, CUBLAS, and cuDNN
# target_link_libraries(NeuroShiv
#     ${CUDA_LIBRARIES}
#     ${CUDA_CUBLAS_LIBRARIES}
#     cudnn  # cuDNN library
# )

target_link_libraries(NeuroShiv
    CUDA::cudart
    CUDA::cublas
    cudnn
)

# Set properties for CUDA (separable compilation for better linking)
set_target_properties(NeuroShiv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;80;90"  # Targets common GPU architectures; adjust if needed for your GPU
)

# Cross-platform handling: On Linux, assume standard paths (override env vars if needed)
if(NOT WIN32)
    include_directories(/usr/local/cuda/include /usr/local/cudnn/include)
    link_directories(/usr/local/cuda/lib64 /usr/local/cudnn/lib)
endif()