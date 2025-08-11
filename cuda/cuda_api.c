#include <cuda_runtime.h>
#include <string.h>

int get_device_name(int deviceID, char *nameBuffer, size_t bufferSize) {
  struct cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, deviceID);
  if (err != cudaSuccess) {
    return (int)err;
  }
  strncpy(nameBuffer, prop.name, bufferSize - 1);
  nameBuffer[bufferSize - 1] = '\0';
  return 0; // success
}

const char *check_cuda(cudaError_t err) {
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return NULL;
}
