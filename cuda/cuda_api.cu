// cuda_api.cu
#include <cuda_runtime.h>
#include <string.h>

extern "C" int getDeviceName(int deviceID, char *nameBuffer,
                             size_t bufferSize) {
  struct cudaDeviceProp prop; // Use struct for C compatibility
  cudaError_t err = cudaGetDeviceProperties(&prop, deviceID);
  if (err != cudaSuccess) {
    return (int)err;
  }
  strncpy(nameBuffer, prop.name, bufferSize - 1);
  nameBuffer[bufferSize - 1] = '\0';
  return 0; // success
}

extern "C" const char *checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return NULL;
}
