package cuda

/*
#cgo LDFLAGS: -L. -lgpudevice -lcudart -Wl,-rpath=.
#include <cuda_runtime.h>
extern char* check_cuda(cudaError_t err);
*/
import "C"
import (
	"fmt"
	"log"
)

func RealCudaGetDeviceCount(count *int) error {
	var cCount C.int
	errStr := C.check_cuda(C.cudaGetDeviceCount(&cCount))
	if errStr != nil {
		return fmt.Errorf("cuda get device count error: %s", C.GoString(errStr))
	}
	*count = int(cCount)
	return nil
}

func SetCudaDevice(deviceID int) error {
	errStr := C.check_cuda(C.cudaSetDevice(C.int(deviceID)))
	if errStr != nil {
		return fmt.Errorf("cudaSetDevice failed: %s", C.GoString(errStr))
	}
	return nil
}

func CudaGetMemInfo(deviceID int) (*MemInfo, error) {
	if err := SetCudaDevice(deviceID); err != nil {
		return nil, err
	}

	var free C.size_t
	var total C.size_t

	errStr := C.check_cuda(C.cudaMemGetInfo(&free, &total))
	if errStr != nil {
		return nil, fmt.Errorf("cudaMemGetInfo failed: %s", C.GoString(errStr))
	}
	return &MemInfo{
		DeviceID: deviceID,
		Free:     int64(free),
		Total:    int64(total),
	}, nil
}

type MemInfo struct {
	DeviceID int
	Free     int64
	Total    int64
}

func RunGetMemInfo() ([]MemInfo, error) {
	var deviceCount int
	if err := RealCudaGetDeviceCount(&deviceCount); err != nil {
		return nil, fmt.Errorf("cudaGetDeviceCount failed: %v", err)
	}

	data := make([]MemInfo, deviceCount)
	for i := 0; i < deviceCount; i++ {
		memInfo, err := CudaGetMemInfo(i)
		if err != nil {
			log.Printf("Error getting memory info for device %d: %v", i, err)
			continue
		}
		data[i] = *memInfo
		log.Printf("Device %d: Free memory: %d bytes, Total memory: %d bytes",
			memInfo.DeviceID, memInfo.Free, memInfo.Total)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("not found any CUDA devices or no memory info available")
	}
	return data, nil
}
