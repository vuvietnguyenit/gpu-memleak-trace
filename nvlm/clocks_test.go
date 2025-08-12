package nvlm_test

import (
	"ebpf-test/nvlm"
	"testing"
)

func TestGetGpuClocks(t *testing.T) {
	_, err := nvlm.GetDriverVersion()
	if err != nil {
		panic("Failed to initialize NVLM API: " + err.Error())
	}
	defer nvlm.ShutdownNVLM()

	clocks, err := nvlm.GetGpuClocks(0)
	if err != nil {
		t.Fatalf("GetGpuClocks returned error: %v", err)
	}

	if len(clocks) == 0 {
		t.Fatalf("Expected clock data, got empty map")
	}

	// Optional: check for expected clock keys
	for _, key := range []string{"Graphics", "SM", "Memory", "Video"} {
		if _, ok := clocks[key]; !ok {
			t.Errorf("Missing clock type: %s", key)
		}
	}

	// Print clocks for debugging
	for k, v := range clocks {
		t.Logf("%s Clock: %d MHz", k, v)
	}
}
