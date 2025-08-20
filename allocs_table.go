package main

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
)

type Dptr uint64

type Size uint64

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

// Human-readable format for size
func (s Size) HumanSize() string {
	val := float64(s)
	units := []string{"B", "KB", "MB", "GB", "TB"}
	i := 0
	for val >= 1024 && i < len(units)-1 {
		val /= 1024
		i++
	}
	return fmt.Sprintf("%.2f %s", val, units[i])
}

func (d Dptr) GPUInstance() string {
	// TODO: integrate with CUDA driver, NVML, etc.
	return fmt.Sprintf("GPU instance for Dptr=0x%x", uint64(d))
}

func (t *AllocTable) Alloc(e Event) {
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

func Bytes16ToString(b [16]byte) string {
	s := string(b[:])
	return strings.TrimRight(s, "\x00")
}

func (t *AllocTable) Aggregate() *Grouped {
	t.mu.Lock()
	defer t.mu.Unlock()

	df := &DF{}
	df.InitHeader([]Header{"PID", "COMM", "DPTR", "TID", "SID", "TOTAL"})
	for key, entry := range t.data {
		df.Insert(Row{
			PID:   entry.Stack.T.P.PID,
			Comm:  Bytes16ToString(entry.Stack.T.P.Comm),
			Dptr:  fmt.Sprintf("0x%x", key.Ptr),
			Tid:   key.TID,
			Sid:   entry.Stack.SID,
			Total: entry.Size,
		})
	}
	return df.GroupAlloc()
}

func (t *AllocTable) Print(ctx context.Context) {
	ticker := time.NewTicker(FlagPrintInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Debug("Stopping printAllocMapPeriodically...")
			return
		case <-ticker.C:
			g := t.Aggregate()
			g.Print()
		}
	}
}
