package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
#include <nvml.h>
nvmlReturn_t get_clocks_info(unsigned int index, nvmlClockType_t clockType,
                             unsigned int *clock);
*/
import "C"
import "fmt"

func GetClocksInfo(deviceIndex int, clockType C.nvmlClockType_t) (uint, error) {
	var clock C.uint

	result := C.get_clocks_info(C.uint(deviceIndex), clockType, &clock)
	if result != C.NVML_SUCCESS {
		return 0, fmt.Errorf("NVML error code: %d", int(result))
	}

	return uint(clock), nil
}

func GetGpuClocks(deviceIndex int) (map[string]uint, error) {
	clocks := map[string]C.nvmlClockType_t{
		"Graphics": C.NVML_CLOCK_GRAPHICS,
		"SM":       C.NVML_CLOCK_SM,
		"Memory":   C.NVML_CLOCK_MEM,
		"Video":    C.NVML_CLOCK_VIDEO,
	}

	result := make(map[string]uint)
	for name, clkType := range clocks {
		val, err := GetClocksInfo(deviceIndex, clkType)
		if err != nil {
			fmt.Printf("failed to get %s clock: %v\n", name, err)
			continue
		}
		clock := C.uint(val)
		result[name] = uint(clock)
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("failed to retrieve any clocks for device %d", deviceIndex)
	}
	return result, nil
}
