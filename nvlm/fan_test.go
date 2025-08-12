package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGetFanSpeed(t *testing.T) {
	_, err := nvlm.GetDriverVersion()
	if err != nil {
		panic("Failed to initialize NVLM API: " + err.Error())
	}
	defer nvlm.ShutdownNVLM()
	v, err := nvlm.GetFanSpeed(0)
	if err != nil {
		t.Fatalf("Failed to get fan speed: %v", err)
	}
	t.Logf("Fan Speed: %d RPM", v)
}
