package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type AllocTable struct {
	mu    sync.RWMutex
	data  map[AllocKey]*AllocEntry
	index map[uint64]map[AllocKey]struct{}
}

type AllocKey struct {
	DeviceID DeviceID
	Uid      Uid
	Pid      Pid
	Comm     Comm
	Tid      Tid
}

type AllocEntry struct {
	Size      AllocSize // allocated size
	TotalSize AllocSize // accumulated size
	LastTs    Timestamp // last allocation timestamp
}

func pidTgid(pid Pid, tid Tid) uint64 {
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
			t.mu.Lock()
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
			t.mu.Unlock()
		}
	}

}

func (d Dptr) GPUInstance() string {
	// TODO: integrate with CUDA driver, NVML, etc.
	return fmt.Sprintf("GPU instance for Dptr=0x%x", uint64(d))
}

func (t *AllocTable) Alloc(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := AllocKey{
		DeviceID: e.DeviceID,
		Tid:      e.Tid,
		Pid:      e.Pid,
		Comm:     e.Comm,
		Uid:      e.Uid,
	}

	entry, ok := t.data[key]
	if !ok {
		entry = &AllocEntry{}
		t.data[key] = entry
		// update index
		h := pidTgid(key.Pid, key.Tid)
		if _, ok := t.index[h]; !ok {
			t.index[h] = make(map[AllocKey]struct{})
		}
		t.index[h][key] = struct{}{}
	}

	// accumulate size
	entry.Size = AllocSize(e.Size)
	entry.TotalSize += AllocSize(e.Size)
	entry.LastTs = e.TsNs
}

func (t *AllocTable) Free(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := AllocKey{
		DeviceID: e.DeviceID,
		Tid:      e.Tid,
		Pid:      e.Pid,
		Comm:     e.Comm,
		Uid:      e.Uid,
	}

	entry, ok := t.data[key]
	if !ok {
		// no existing record for this key, ignore
		return
	}

	// decrement safely
	if entry.TotalSize >= AllocSize(e.Size) {
		entry.TotalSize -= AllocSize(e.Size)
	} else {
		entry.TotalSize = 0
	}

	entry.LastTs = e.TsNs

	// if everything is zero, remove entry
	if entry.TotalSize == 0 {
		delete(t.data, key)
		h := pidTgid(key.Pid, key.Tid)
		if idx, ok := t.index[h]; ok {
			delete(idx, key)
			if len(idx) == 0 {
				delete(t.index, h)
			}
		}
	}
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
			r := t.Aggregate()
			if len(r) == 0 {
				continue
			}
			PrintResults(r, 5, 5)
		}
	}
}
