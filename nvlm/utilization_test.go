package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGetUtilization(t *testing.T) {
	// Initialize NVLM
	err := nvlm.InitNVLM()
	if err != nil {
		t.Fatalf("Failed to initialize NVLM API: %v", err)
	}

	util, err := nvlm.GetGPUUtilizationRates(0) // Assuming we want to test the first device
	if err != nil {
		t.Fatalf("Failed to get GPU utilization: %v", err)
	}

	if util.GPU == 0 && util.Memory == 0 {
		t.Errorf("Expected non-zero GPU and Memory utilization, got: %+v", util)
	}
	t.Logf("GPU Utilization: %+v", util)
}
