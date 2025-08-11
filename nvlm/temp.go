package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
unsigned int get_gpu_temperature(int device_index, int *err);
*/
import "C"
import "fmt"

func GetGPUTemperature(device int) (uint, error) {
	var cerr C.int
	t := C.get_gpu_temperature(C.int(device), &cerr)
	if cerr != 0 {
		return 0, fmt.Errorf("NVML error code: %d", int(cerr))
	}
	return uint(t), nil
}
