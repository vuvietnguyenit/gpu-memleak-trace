package main

import (
	"fmt"
	"strings"
)

func (e EventType) String() string {
	switch e {
	case EVENT_MALLOC:
		return "MALLOC"
	case EVENT_FREE:
		return "FREE"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", e)
	}
}

type Event struct {
	Pid       Pid
	Tid       Tid
	DeivceID  DeivceID
	Uid       Uid
	StackID   StackID
	_         uint32 // padding to make struct 64 bytes
	Size      AllocSize
	Dptr      Dptr
	Comm      Comm
	EventType EventType
	Retval    Retval
}

func (e Event) String() string {
	comm := strings.TrimRight(string(e.Comm[:]), "\x00")

	switch e.EventType {
	case EVENT_MALLOC:
		return fmt.Sprintf(
			"[MALLOC] pid=%d tid=%d uid=%d comm=%s size=%d dptr=0x%x stack_id=%d retval=%d",
			e.Pid, e.Tid, e.Uid, comm, e.Size, e.Dptr, e.StackID, e.Retval,
		)
	case EVENT_FREE:
		return fmt.Sprintf(
			"[FREE]   pid=%d tid=%d uid=%d comm=%s dptr=0x%x stack_id=%d retval=%d",
			e.Pid, e.Tid, e.Uid, comm, e.Dptr, e.StackID, e.Retval,
		)
	default:
		return "[UNKNOWN]"
	}
}
