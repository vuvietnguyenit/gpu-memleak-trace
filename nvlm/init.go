package nvlm

/*
#cgo LDFLAGS: -L. -lnvlmstats -lnvidia-ml -Wl,-rpath=.
extern const char* get_driver_version();
*/
import "C"
import (
	"fmt"
	"strings"
)

func InitNVLM() error {
	version := C.GoString(C.get_driver_version())
	if strings.Contains(version, "Failed") {
		return fmt.Errorf("error: %q", version)
	}
	fmt.Println("NVIDIA Driver Version:", version)
	return nil
}
