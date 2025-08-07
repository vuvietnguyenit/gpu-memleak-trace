#include <cuda.h>
#include <stdio.h>

int main() {
  CUresult res;
  CUdevice device;
  CUcontext context;
  CUdeviceptr dptr;
  size_t size = 1024 * 1024; // 1 MB

  // Initialize CUDA driver
  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuInit failed: %d\n", res);
    return 1;
  }

  // Get CUDA device
  res = cuDeviceGet(&device, 0);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuDeviceGet failed: %d\n", res);
    return 1;
  }

  // Create context
  res = cuCtxCreate(&context, 0, device);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuCtxCreate failed: %d\n", res);
    return 1;
  }

  // Allocate memory using cuMemAlloc
  res = cuMemAlloc(&dptr, size);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemAlloc failed: %d\n", res);
    return 1;
  }

  printf("Allocated %zu bytes at device pointer: 0x%llx\n", size,
         (unsigned long long)dptr);

  // Free memory (optional)
  res = cuMemFree(dptr);
  if (res != CUDA_SUCCESS) {
    fprintf(stderr, "cuMemFree failed: %d\n", res);
    return 1;
  }

  printf("Memory freed at 0x%llx.\n", (unsigned long long)dptr);

  return 0;
}
