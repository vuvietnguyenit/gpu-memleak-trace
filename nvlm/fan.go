package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
unsigned int get_fan_speed(int device_index, int *err);
*/
import "C"
import "fmt"

func GetFanSpeed(device int) (uint, error) {
	var cerr C.int
	fanSpeed := C.get_fan_speed(C.int(device), &cerr)
	if cerr != 0 {
		return 0, fmt.Errorf("NVML error code: %d", int(cerr))
	}
	var r = uint(fanSpeed)
	return r, nil
}
