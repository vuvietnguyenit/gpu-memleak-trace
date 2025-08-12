package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
#include "nvml.h"

extern const char* get_driver_version();
nvmlReturn_t nvml_shutdown();
*/
import "C"
import (
	"fmt"
	"strings"
)

func GetDriverVersion() (string, error) {
	version := C.GoString(C.get_driver_version())
	if strings.Contains(version, "Failed") {
		return "", fmt.Errorf("error: %q", version)
	}
	return version, nil
}

func ShutdownNVLM() error {
	result := C.nvml_shutdown()
	if result != C.NVML_SUCCESS {
		return fmt.Errorf("failed to shutdown NVML: %d", int(result))
	}
	fmt.Println("NVML shutdown successfully")
	return nil
}
