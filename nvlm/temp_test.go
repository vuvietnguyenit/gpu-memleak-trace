package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGetGPUTemperature(t *testing.T) {
	// Initialize NVLM
	err := nvlm.InitNVLM()
	if err != nil {
		t.Fatalf("Failed to initialize NVLM API: %v", err)
	}
	temp, err := nvlm.GetGPUTemperature(0)
	if err != nil {
		t.Fatalf("Failed to get GPU temperature: %v", err)
	}

	if temp == 0 {
		t.Errorf("Expected non-zero GPU temperature, got: %d", temp)
	}
	t.Logf("GPU Temperature: %d", temp)
}
