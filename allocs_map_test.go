package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestAllocMapWarnings(t *testing.T) {
	allocs := NewAllocMap()

	// Capture stdout for warning checks
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// 1. Normal malloc
	allocs.AddAlloc(100, 0xABC, 1024)

	// 2. Duplicate malloc (should warn)
	allocs.AddAlloc(100, 0xABC, 2048)

	// 3. Free non-existent PID (should warn)
	allocs.FreeAlloc(200, 0x123)

	// 4. Free existing PID but missing ptr (should warn)
	allocs.FreeAlloc(100, 0xDEF)

	// 5. Normal free
	allocs.FreeAlloc(100, 0xABC)

	// Close writer and restore stdout
	w.Close()
	var buf bytes.Buffer
	_, _ = buf.ReadFrom(r)
	os.Stdout = oldStdout

	output := buf.String()

	// Check warnings
	if !strings.Contains(output, "already has ptr") {
		t.Errorf("Expected duplicate malloc warning, got:\n%s", output)
	}
	if !strings.Contains(output, "PID 200 not found") {
		t.Errorf("Expected missing PID warning, got:\n%s", output)
	}
	if !strings.Contains(output, "has no record for ptr") {
		t.Errorf("Expected missing ptr warning, got:\n%s", output)
	}

	fmt.Println("Captured warnings:\n" + output)
}
