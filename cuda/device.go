package cuda

/*
#cgo LDFLAGS: -L. -lgpudevice -lcudart -Wl,-rpath=.
#include <stdlib.h>

extern int get_device_name(int deviceID, char *nameBuffer, size_t bufferSize);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func GetDeviceName(deviceID int) (string, error) {
	bufSize := 256
	cbuf := (*C.char)(C.malloc(C.size_t(bufSize)))
	defer C.free(unsafe.Pointer(cbuf))

	errCode := C.get_device_name(C.int(deviceID), cbuf, C.size_t(bufSize))
	if errCode != 0 {
		return "", fmt.Errorf("CUDA error code: %d", int(errCode))
	}
	return C.GoString(cbuf), nil
}
