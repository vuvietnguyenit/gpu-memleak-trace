package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type Dptr uint64

type AllocSize uint64

type AllocTable struct {
	mu    sync.Mutex
	data  map[AllocKey]*AllocEntry
	index map[uint64]map[AllocKey]struct{}
}

type AllocKey struct {
	PID uint32
	TID uint32
	Ptr Dptr
}

type AllocEntry struct {
	Size  AllocSize
	Stack *StackInfo
}

func pidTgid(pid, tid uint32) uint64 {
	return (uint64(pid) << 32) | uint64(tid)
}

func NewAllocTable() *AllocTable {
	return &AllocTable{
		data:  make(map[AllocKey]*AllocEntry),
		index: make(map[uint64]map[AllocKey]struct{}),
	}
}

// tidExists checks whether a given PID/TID thread is alive in /proc.
func tidExists(pid, tid uint32) bool {
	path := filepath.Join("/proc", fmt.Sprint(pid), "task", fmt.Sprint(tid))
	_, err := os.Stat(path)
	return err == nil
}

func (t *AllocTable) Cleanup(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for h, set := range t.index {
				// hash -> all AllocKeys with same (PID,TID)
				var pid, tid uint32
				pid = uint32(h >> 32)
				tid = uint32(h & 0xFFFFFFFF)

				if !tidExists(pid, tid) {
					// remove all keys in data + index
					for k := range set {
						delete(t.data, k)
					}
					delete(t.index, h)
				}
			}
		}
	}

}

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

	key := AllocKey{TID: e.Tid, PID: e.Pid, Ptr: Dptr(e.Dptr)}

	t.data[key] = &AllocEntry{
		Size:  AllocSize(e.Size),
		Stack: s,
	}
	// Insert to index
	h := pidTgid(key.PID, key.TID)
	if _, ok := t.index[h]; !ok {
		t.index[h] = make(map[AllocKey]struct{})
	}
	t.index[h][key] = struct{}{}
}

func (t *AllocTable) Free(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := AllocKey{TID: e.Tid, PID: e.Pid, Ptr: Dptr(e.Dptr)}
	delete(t.data, key)
}

func (t *AllocTable) Aggregate() *Grouped {
	t.mu.Lock()
	defer t.mu.Unlock()

	df := &DF{}
	df.InitHeader([]Header{"PID", "COMM", "DPTR", "TID", "SID", "TOTAL"})
	for key, entry := range t.data {
		df.Insert(Row{
			PID:   key.PID,
			Comm:  entry.Stack.T.P.Comm,
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
			slog.Debug("Stopping Print action...")
			return
		case <-ticker.C:
			fmt.Println(strings.Repeat("-", 20), time.Now().Format(time.RFC3339), strings.Repeat("-", 20))
			g := t.Aggregate()
			if len(g.Group) == 0 {
				fmt.Println("NO EVENT.")
				continue
			}
			g.Print()
		}
	}
}
