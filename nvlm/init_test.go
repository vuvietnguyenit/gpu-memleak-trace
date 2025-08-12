package nvlm_test

import (
	nvlmapi "ebpf-test/nvlm"
	"testing"
)

func TestInitNVLM(t *testing.T) {
	_, err := nvlmapi.GetDriverVersion()
	if err != nil {
		t.Fatalf("Failed to initialize NVLM API: %v", err)
	}
	defer nvlmapi.ShutdownNVLM()

	t.Log("NVLM API initialized successfully")
}
