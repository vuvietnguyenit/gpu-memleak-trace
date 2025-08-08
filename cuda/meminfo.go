package cuda

import (
	"fmt"
	"log"
)

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
