package main

import "fmt"

const (
	EVENT_MALLOC EventType = 0
	EVENT_FREE   EventType = 1
)

type EventType int32
type Dptr uint64
type AllocSize uint64
type Pid uint32
type DeivceID int32 // If will return -1 if can't get device ID
type Uid uint32     // For example: 0 = root
type StackID uint32
type Comm [16]byte
type Tid uint32
type Retval int32

// Human-readable format for size
func (s AllocSize) HumanSize() string {
	val := float64(s)
	units := []string{"B", "KB", "MB", "GB", "TB"}
	i := 0
	for val >= 1024 && i < len(units)-1 {
		val /= 1024
		i++
	}
	return fmt.Sprintf("%.2f %s", val, units[i])
}
