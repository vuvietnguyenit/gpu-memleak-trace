package nvlm_test

import (
	nvlmapi "ebpf-test/nvlm"
	"testing"
)

func TestInitNVLM(t *testing.T) {
	err := nvlmapi.InitNVLM()
	if err != nil {
		t.Fatalf("Failed to initialize NVLM API: %v", err)
	}

	t.Log("NVLM API initialized successfully")
}
