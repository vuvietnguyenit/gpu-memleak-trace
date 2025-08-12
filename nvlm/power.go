package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
unsigned int get_power_usage(int device_index, int *err);
*/
import "C"
import "fmt"

func GetGPUPowerUsage(device int) (uint, error) {
	var cerr C.int
	power := C.get_power_usage(C.int(device), &cerr)
	if cerr != 0 {
		return 0, fmt.Errorf("NVML error code: %d", int(cerr))
	}
	if power <= 0 {
		return 0, fmt.Errorf("invalid power usage value: %d", power)
	}
	return uint(power), nil
}
