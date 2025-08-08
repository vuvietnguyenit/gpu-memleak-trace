package cuda_test

import (
	"ebpf-test/cuda"
	"testing"
)

func TestGetMemInfo(t *testing.T) {
	deviceID := 0 // Assuming we want to test the first device
	info, err := cuda.CudaGetMemInfo(deviceID)
	if err != nil {
		t.Fatalf("getMemInfo failed: %v", err)
	}

	if info.Total == 0 {
		t.Errorf("Total memory is zero: %+v", info)
	}
	if info.Free == 0 {
		t.Logf("Warning: free memory is zero (GPU might be fully allocated)")
	}

	t.Logf("Device %d: Free=%d, Total=%d", info.DeviceID, info.Free, info.Total)
}
