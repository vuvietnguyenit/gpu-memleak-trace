
#include "nvml.h"

const char *get_driver_version() {
  static char version[80];
  nvmlReturn_t result = nvmlInit();
  if (result != NVML_SUCCESS) {
    return "Failed to init NVML";
  }

  result = nvmlSystemGetDriverVersion(version, sizeof(version));
  if (result != NVML_SUCCESS) {
    return "Failed to get driver version";
  }
  return version;
}

nvmlUtilization_t get_gpu_utilization_rate(int device_index, int *err) {
  nvmlUtilization_t util;
  util.gpu = 0;
  util.memory = 0;

  nvmlReturn_t result;
  nvmlDevice_t device;
  result = nvmlDeviceGetHandleByIndex(device_index, &device);
  if (result != NVML_SUCCESS) {
    *err = result;
    nvmlShutdown();
    return util;
  }

  result = nvmlDeviceGetUtilizationRates(device, &util);
  if (result != NVML_SUCCESS) {
    *err = result;
  } else {
    *err = 0;
  }

  nvmlShutdown();
  return util;
}

unsigned int get_gpu_temperature(int device_index, int *err) {
  unsigned int temp = 0;

  nvmlReturn_t result;
  nvmlDevice_t device;
  result = nvmlDeviceGetHandleByIndex(device_index, &device);
  if (result != NVML_SUCCESS) {
    *err = result;
    nvmlShutdown();
    return temp;
  }

  result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  if (result != NVML_SUCCESS) {
    *err = result;
  } else {
    *err = 0;
  }

  nvmlShutdown();
  return temp;
}