package main

import "fmt"

type Dptr uint64

func (d Dptr) GPUInstance() string {
	// TODO: integrate with CUDA driver, NVML, etc.
	return fmt.Sprintf("GPU instance for Dptr=0x%x", uint64(d))
}

type Size uint64

type AllocInfo struct {
	S    *StackInfo
	Dptr Dptr
	Size Size
}

// Human-readable format for size
func (s Size) HumanSize() string {
	val := float64(s)
	units := []string{"B", "KB", "MB", "GB", "TB"}
	i := 0
	for val >= 1024 && i < len(units)-1 {
		val /= 1024
		i++
	}
	return fmt.Sprintf("%.2f %s", val, units[i])
}
