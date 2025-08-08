package cuda

/*
#cgo LDFLAGS: -L. -lgpudevice -lcudart -Wl,-rpath=.
#include <stdlib.h>

// Declaration of the C function from our CUDA code
int getDeviceName(int deviceID, char *nameBuffer, size_t bufferSize);
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

	errCode := C.getDeviceName(C.int(deviceID), cbuf, C.size_t(bufSize))
	if errCode != 0 {
		return "", fmt.Errorf("CUDA error code: %d", int(errCode))
	}
	return C.GoString(cbuf), nil
}
