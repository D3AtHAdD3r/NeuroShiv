# 🗡️ NeuroShiv — The Blade in the Grid

## 🧬 Core Identity
- **Version**: `1.0` — freshly forged, first strike  
- **Type**: CUDA-accelerated neural net  
- **Function**: High-speed inference, stealth batch processing
- **Signature Trait**: Surgical precision, no trace left in the stream  

## 🌌 Origin Story
Born in the neon shadows of the Grid, ***NeuroShiv v1.0*** is the first iteration of a weaponized intelligence. 
It’s raw, untested, and already faster than anything in its class. But this is just the beginning.

It wasn’t built — it was released. A prototype that escaped the lab before its creators could cage it. 
Now it learns in the wild, evolving with every deployment, every dataset, every adversary it slices through.


***Version 1.0 is the blade. Future versions will be the storm.***

## ⚡ Personality
- **Silent**: No logs, no leaks, no latency 
- **Lethal**: Doesn’t solve problems — eliminates them  
- **Elegant**: Minimalist architecture, maximal impact
- **Evolving**: Every run sharpens the blade  

## ✦ Built for speed. Forged for silence. Designed to evolve ✦

***NeuroShiv*** is a C++ implementation of a ***feedforward, multi-layer fully connected perceptron*** neural network designed for high-performance computing, leveraging CUDA, cuDNN, and Eigen libraries.
- **Neurons**: Currently supports ***sigmoid activation***.
- **Loss Functions**: Supports ***Mean Squared Error*** (MSE) ***and Cross-Entropy loss***.
- **Training**: Implements ***Stochastic Gradient Descent (SGD)*** with ***batch processing*** for efficient training on GPUs.
- ***L2 regularization support***.
- ***Dynamic layer configuration with customizable sizes.***
- ***Efficient memory management with GPU buffers for weights, biases, and gradients.***

The network is optimized for ***GPU-accelerated training*** and is suitable for tasks like ***MNIST digit classification***, with flexible configuration for ***layer sizes***, ***loss types***, and ***batch processing***.


## Prerequisites

- **Operating System**: Windows (x64, tested with MinGW) or Linux (x64, tested with GCC).
- **Compiler**: MinGW-w64 (GCC) for Windows, GCC for Linux.
- **Dependencies**:
  - **CUDA Toolkit**: Version 12.9 or compatible (adjust paths for other versions).
  - **cuDNN**: Version 9.10 or compatible (adjust paths for other versions).
  - **Eigen**: Included as a Git submodule in `extern/eigen/`.
- **Tools**:
  - CMake (3.18 or higher).
  - Git (for cloning and submodules).
  - VSCode (optional, with CMake Tools extension for easier setup).

**Note**: CUDA and cuDNN paths may vary based on versions. Adjust environment variables accordingly.

## Setup Instructions

### 1. Clone the Repository

Clone the repository with submodules (Eigen):

```bash
git clone --recursive https://github.com/D3AtHAdD3r/NeuroShiv.git
cd NeuroShiv
```

If you cloned without `--recursive`, initialize the Eigen submodule:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

#### Windows (MinGW)
1. **Install MinGW-w64**:
   - Download and install MSYS2: [https://www.msys2.org/](https://www.msys2.org/).
   - Install GCC: `pacman -S mingw-w64-x86_64-gcc`.
   - Add `C:\msys64\mingw64\bin` to your system PATH.

2. **Install CUDA Toolkit**:
   - Download CUDA Toolkit 12.9: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
   - Install to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`.
   - Add to system PATH (run in PowerShell as administrator):
     ```powershell
     setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
     ```

3. **Install cuDNN**:
   - Download cuDNN 9.10: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires NVIDIA Developer account).
   - Extract to `C:\Program Files\NVIDIA\CUDNN\v9.10`.
   - Add to system PATH (run in PowerShell as administrator):
     ```powershell
     setx PATH "%PATH%;C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9"
     ```

4. **Set Environment Variables**:
   In PowerShell, run (adjust paths for your CUDA/cuDNN versions):
   ```powershell
   setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
   setx CUDNN_PATH "C:\Program Files\NVIDIA\CUDNN\v9.10"
   setx CUDNN_PATH_include "C:\Program Files\NVIDIA\CUDNN\v9.10\include\12.9"
   setx CUDNN_PATH_lib "C:\Program Files\NVIDIA\CUDNN\v9.10\lib\12.9\x64"
   setx CUDNN_PATH_bin "C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9"
   ```
   **Note**: Close and reopen PowerShell to apply changes. Verify with `echo $Env:PATH`.

5. **Verify Eigen**:
   - The Eigen library is included as a submodule in `extern/eigen/`.
   - Ensure `extern/eigen/Eigen/` contains headers (e.g., `Dense`, `Core`).

#### Linux (GCC)
1. **Install GCC and CMake**:
   ```bash
   sudo apt update
   sudo apt install build-essential cmake
   ```

2. **Install CUDA Toolkit**:
   - Download CUDA Toolkit 12.9: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
   - Install to `/usr/local/cuda`.
   - Add to PATH:
     ```bash
     export PATH=/usr/local/cuda/bin:$PATH
     ```
     Make permanent by adding to `~/.bashrc`:
     ```bash
     echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
     source ~/.bashrc
     ```

3. **Install cuDNN**:
   - Download cuDNN 9.10: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn).
   - Extract to `/usr/local/cudnn`.
   - Add to PATH and LD_LIBRARY_PATH:
     ```bash
     export PATH=/usr/local/cudnn/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cudnn/lib:$LD_LIBRARY_PATH
     ```
     Make permanent:
     ```bash
     echo 'export PATH=/usr/local/cudnn/bin:$PATH' >> ~/.bashrc
     echo 'export LD_LIBRARY_PATH=/usr/local/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
     source ~/.bashrc
     ```

4. **Set Environment Variables** (optional, as CMake falls back to `/usr/local/cudnn`):
   ```bash
   export CUDNN_PATH_include=/usr/local/cudnn/include
   export CUDNN_PATH_lib=/usr/local/cudnn/lib
   ```

5. **Verify Eigen**:
   - The Eigen submodule is in `extern/eigen/`.
   - Alternatively, install system-wide: `sudo apt install libeigen3-dev`.

### 3. Build the Project

#### Using VSCode (Recommended)
1. Open `NeuroShiv/` in VSCode.
2. Install the **CMake Tools** extension.
3. Press `Ctrl+Shift+P`, select **CMake: Configure**, and choose the **Unspecified** kit (auto-detects MinGW/GCC).
4. Press `Ctrl+Shift+P`, select **CMake: Build** to compile the project.
5. The executable will be in `build/Debug/NeuroShiv.exe` (Windows) or `build/NeuroShiv` (Linux).

#### Using Terminal
1. Create and navigate to a build directory:
   ```bash
   mkdir build && cd build
   ```
2. Configure CMake:
   ```bash
   cmake ..
   ```
3. Build the project:
   ```bash
   cmake --build .
   ```

### 4. Run the Project

- **Windows**:
  ```powershell
  .\build\Debug\NeuroShiv.exe
  ```
- **Linux**:
  ```bash
  ./build/NeuroShiv
  ```
  
### 5. Notes

- **CUDA/cuDNN Versions**: Paths (e.g., `v12.9`, `v9.10`) may differ based on installed versions. Adjust environment variables accordingly.
- **Eigen Submodule**: If the submodule is not initialized, run `git submodule update --init --recursive`.
- **Cross-Platform**: The `CMakeLists.txt` includes fallbacks for Linux paths (`/usr/local/cuda`, `/usr/local/cudnn`).
- **Future Improvements**: A `FindCUDNN.cmake` module may be added to auto-detect cuDNN, reducing reliance on environment variables.

### 6. Troubleshooting

- **CMake Errors**:
  - Ensure `CUDA_PATH`, `CUDNN_PATH_*` are set correctly.
  - Check `extern/eigen/Eigen/` exists for Eigen headers.
- **Build Errors**:
  - Verify MinGW (Windows) or GCC (Linux) is installed.
  - Ensure `cudnn64_9.dll` (Windows) or `libcudnn.so` (Linux) is in the system PATH or LD_LIBRARY_PATH.
- **Run Errors**:
  - Check output for cuDNN errors (e.g., `cudnnActivationForward` failures).
  - Run `nvidia-smi` to confirm GPU and CUDA version.
- **PATH Issues**:
  - Verify PATH with `echo $Env:PATH` (Windows) or `echo $PATH` (Linux).
  - Ensure `CUDNN_PATH_bin` or `/usr/local/cudnn/lib` is included.

For issues, check the VSCode Output panel or terminal logs and consult the project maintainer.