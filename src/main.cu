// Smoke test: proves the CUDA toolchain compiles and links, and runs a trivial
// kernel when a GPU is actually present. Exits cleanly on a machine without one.

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

__global__ void saxpy(int n, float a, const float* x, float* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  int deviceCount = 0;
  const cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::printf("compiled and linked OK, no CUDA device available (%s)\n",
                cudaGetErrorString(err));
    return 0;
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device 0: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  constexpr int kN = 1 << 20;
  std::vector<float> hx(kN, 1.0f);
  std::vector<float> hy(kN, 2.0f);

  float* dx = nullptr;
  float* dy = nullptr;
  cudaMalloc(&dx, kN * sizeof(float));
  cudaMalloc(&dy, kN * sizeof(float));
  cudaMemcpy(dx, hx.data(), kN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy.data(), kN * sizeof(float), cudaMemcpyHostToDevice);

  constexpr int kBlock = 256;
  saxpy<<<(kN + kBlock - 1) / kBlock, kBlock>>>(kN, 3.0f, dx, dy);
  cudaMemcpy(hy.data(), dy, kN * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(dx);
  cudaFree(dy);

  const bool ok = hy.front() == 5.0f && hy.back() == 5.0f;
  std::printf("saxpy: %s (y[0]=%.1f)\n", ok ? "OK" : "FAILED", hy.front());
  return ok ? 0 : 1;
}
