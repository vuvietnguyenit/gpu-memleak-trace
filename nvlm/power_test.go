package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGPUPowerUsage(t *testing.T) {
	_, err := nvlm.GetDriverVersion()
	if err != nil {
		panic("Failed to initialize NVLM API: " + err.Error())
	}
	defer nvlm.ShutdownNVLM()
	v, err := nvlm.GetGPUPowerUsage(0)
	if err != nil {
		t.Fatalf("Failed to get GPU power usage: %v", err)
	}
	t.Logf("GPU Power Usage: %d Watts", v)

}
