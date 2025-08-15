package main

import "fmt"

type EventType int32

const (
	EVENT_MALLOC EventType = 0
	EVENT_FREE   EventType = 1
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
	Pid       uint32
	Tid       uint32
	Uid       uint32
	StackID   int32
	Size      uint64
	Dptr      uint64
	Comm      [16]byte
	EventType EventType
	Retval    int32
}

func (a Event) String() string {
	commStr := string(a.Comm[:])
	// trim trailing null bytes
	for i := 0; i < len(commStr); i++ {
		if commStr[i] == 0 {
			commStr = commStr[:i]
			break
		}
	}

	switch a.EventType {
	case EVENT_MALLOC:
		return fmt.Sprintf(
			"[PID=%d TID=%d UID=%d COMM=%s] Event=%s StackID=%d Size=%d bytes Dptr=0x%x Retval=%d",
			a.Pid, a.Tid, a.Uid, commStr, a.EventType, a.StackID, a.Size, a.Dptr, a.Retval,
		)
	case EVENT_FREE:
		return fmt.Sprintf(
			"[PID=%d TID=%d UID=%d COMM=%s] Event=%s Dptr=0x%x Retval=%d",
			a.Pid, a.Tid, a.Uid, commStr, a.EventType, a.Dptr, a.Retval,
		)
	default:
		return fmt.Sprintf(
			"[PID=%d TID=%d UID=%d COMM=%s] Event=%s StackID=%d Size=%d bytes Dptr=0x%x Retval=%d",
			a.Pid, a.Tid, a.Uid, commStr, a.EventType, a.StackID, a.Size, a.Dptr, a.Retval,
		)
	}
}
