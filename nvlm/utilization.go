package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
#include <nvml.h>
extern nvmlUtilization_t get_gpu_utilization_rate(int device_index, int *err);
*/
import "C"
import "fmt"

type Utilization struct {
	GPU    uint
	Memory uint
}

func GetGPUUtilizationRates(device int) (Utilization, error) {
	var cerr C.int
	cu := C.get_gpu_utilization_rate(C.int(device), &cerr)
	if cerr != 0 {
		return Utilization{}, fmt.Errorf("NVML error code: %d", int(cerr))
	}
	return Utilization{
		GPU:    uint(cu.gpu),
		Memory: uint(cu.memory),
	}, nil
}
