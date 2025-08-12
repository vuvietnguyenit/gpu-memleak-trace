package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGetUtilization(t *testing.T) {
	// Initialize NVLM
	_, err := nvlm.GetDriverVersion()
	if err != nil {
		t.Fatalf("Failed to initialize NVLM API: %v", err)
	}
	defer nvlm.ShutdownNVLM()

	util, err := nvlm.GetGPUUtilizationRates(0) // Assuming we want to test the first device
	if err != nil {
		t.Fatalf("Failed to get GPU utilization: %v", err)
	}
	t.Logf("GPU Utilization: %+v", util)
}
