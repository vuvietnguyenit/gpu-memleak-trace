package cuda_test

import (
	"ebpf-test/cuda"
	"testing"
)

func TestGetDeviceName(t *testing.T) {
	name, err := cuda.GetDeviceName(0) // first GPU
	if err != nil {
		t.Skipf("Skipping test — CUDA device not available: %v", err)
	}
	if name == "" {
		t.Errorf("Expected non-empty device name, got empty string")
	}
	t.Logf("Device 0 name: %s", name)
}
