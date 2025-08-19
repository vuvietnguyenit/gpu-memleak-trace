package main

import "sync"

type AllocTable struct {
	mu   sync.Mutex
	data map[AllocKey]*AllocEntry
}

type AllocKey struct {
	TID uint32 // or even TID if you want per-thread precision
	Ptr Dptr
}

type AllocEntry struct {
	Size  Size
	Stack *StackInfo
}

func NewAllocTable() *AllocTable {
	return &AllocTable{
		data: make(map[AllocKey]*AllocEntry),
	}
}

func (t *AllocTable) Add(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	p := &ProcessInfo{
		PID:  e.Pid,
		Comm: e.Comm,
		UID:  e.Uid,
	}
	th := &ThreadInfo{P: p, TID: e.Tid}
	s := &StackInfo{T: th, SID: e.StackID}

	key := AllocKey{TID: e.Tid, Ptr: Dptr(e.Dptr)}

	t.data[key] = &AllocEntry{
		Size:  Size(e.Size),
		Stack: s,
	}
}

func (t *AllocTable) Free(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := AllocKey{TID: e.Tid, Ptr: Dptr(e.Dptr)}
	delete(t.data, key)
}
