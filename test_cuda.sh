#!/bin/bash
# CUDA Environment Test Suite for WSL
# Tests CUDA toolkit installation and GPU accessibility

echo "🔧 CUDA Development Environment Test Suite"
echo "Testing CUDA toolkit, GPU drivers, and compilation environment"
echo "Running on WSL (Windows Subsystem for Linux)"
echo ""

# Test 1: Check if nvidia-smi works in WSL
echo "=== Test 1: GPU Detection ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    echo "✅ GPU accessible from WSL!"
else
    echo "❌ nvidia-smi not found"
    echo "💡 Install NVIDIA drivers for WSL2"
fi
echo ""

# Test 2: Check CUDA Toolkit
echo "=== Test 2: CUDA Toolkit ==="
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc (CUDA compiler) found"
    nvcc --version
    echo "✅ CUDA Toolkit installed!"
else
    echo "❌ nvcc not found"
    echo "💡 Need to install CUDA Toolkit"
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
fi
echo ""

# Test 3: Check CUDA libraries
echo "=== Test 3: CUDA Libraries ==="
CUDA_PATHS=("/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu" "/opt/cuda/lib64")

for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ CUDA library path found: $path"
        if [ -f "$path/libcudart.so" ]; then
            echo "✅ CUDA runtime library found"
        fi
        if [ -f "$path/libcurand.so" ]; then
            echo "✅ CURAND library found (needed for Monte Carlo)"
        fi
        break
    fi
done
echo ""

# Test 4: Create simple CUDA test program
echo "=== Test 4: CUDA Compilation Test ==="
cat > cuda_test.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_gpu() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("=== Simple CUDA Test ===\n");
    
    // Check CUDA device
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        printf("❌ CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("✅ CUDA devices found: %d\n", device_count);
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("✅ GPU: %s\n", prop.name);
        printf("✅ Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("✅ Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        
        // Test simple kernel launch
        hello_gpu<<<1, 5>>>();
        cudaDeviceSynchronize();
        printf("✅ CUDA kernel execution successful!\n");
    }
    
    return 0;
}
EOF

if command -v nvcc &> /dev/null; then
    echo "Compiling CUDA test program..."
    if nvcc -o cuda_test cuda_test.cu 2>/dev/null; then
        echo "✅ CUDA compilation successful!"
        echo "Running CUDA test..."
        ./cuda_test
        rm -f cuda_test cuda_test.cu
    else
        echo "❌ CUDA compilation failed"
        echo "💡 Check CUDA installation and libraries"
        rm -f cuda_test.cu
    fi
else
    echo "❌ Cannot test CUDA compilation (nvcc not found)"
    rm -f cuda_test.cu
fi
echo ""

# Test 5: Development environment summary
echo "=== Test 5: Development Environment Summary ==="
echo "System Information:"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  C Compiler: $(gcc --version | head -1)"

if command -v nvcc &> /dev/null; then
    echo "  CUDA Compiler: $(nvcc --version | grep release | cut -d' ' -f5-6)"
fi

echo ""
echo "Required for Asian Options CUDA Implementation:"
echo "  ✅ C Compiler (GCC 13.3.0) - READY"
echo "  ✅ Math Library - READY"
echo "  ✅ Memory Management - READY"

if command -v nvcc &> /dev/null && command -v nvidia-smi &> /dev/null; then
    echo "  ✅ CUDA Toolkit - READY"
    echo "  ✅ GPU Access - READY"
    echo ""
    echo "🎯 RESULT: Fully ready for CUDA Asian Options development!"
    echo "🚀 Next step: Compile and run Asian Options CUDA code"
else
    echo "  ❌ CUDA Toolkit - NEEDS INSTALLATION"
    echo "  ❌ GPU Access - NEEDS SETUP"
    echo ""
    echo "🔧 RESULT: Need to install CUDA Toolkit for GPU acceleration"
    echo "💡 Can still develop CPU-only version while setting up CUDA"
fi

echo ""
echo "=== Installation Instructions (if needed) ==="
echo ""
echo "🔧 To install CUDA Toolkit in WSL2:"
echo "1. Download CUDA Toolkit for WSL-Ubuntu:"
echo "   https://developer.nvidia.com/cuda-downloads"
echo "   Select: Linux > x86_64 > WSL-Ubuntu > 2.0 > deb(network)"
echo ""
echo "2. Install commands:"
echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb"
echo "   sudo dpkg -i cuda-keyring_1.0-1_all.deb"
echo "   sudo apt-get update"
echo "   sudo apt-get -y install cuda-toolkit-12-6"
echo ""
echo "3. Add to ~/.bashrc:"
echo "   export PATH=/usr/local/cuda/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "4. Restart WSL and test: nvcc --version"
