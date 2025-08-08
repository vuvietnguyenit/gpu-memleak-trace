// meminfo_c.go
//go:build !test

package cuda

/*
#cgo LDFLAGS: -lcudart
#include <cuda_runtime.h>
#include <stdlib.h>

const char* checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    return NULL;
}
*/
import "C"
import (
	"fmt"
)

func RealCudaGetDeviceCount(count *int) error {
	var cCount C.int
	errStr := C.checkCuda(C.cudaGetDeviceCount(&cCount))
	if errStr != nil {
		return fmt.Errorf("cuda get device count error: %s", C.GoString(errStr))
	}
	*count = int(cCount)
	return nil
}

func SetCudaDevice(deviceID int) error {
	errStr := C.checkCuda(C.cudaSetDevice(C.int(deviceID)))
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

	errStr := C.checkCuda(C.cudaMemGetInfo(&free, &total))
	if errStr != nil {
		return nil, fmt.Errorf("cudaMemGetInfo failed: %s", C.GoString(errStr))
	}
	return &MemInfo{
		DeviceID: deviceID,
		Free:     int64(free),
		Total:    int64(total),
	}, nil
}
