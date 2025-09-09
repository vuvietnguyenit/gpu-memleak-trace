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

type AllocTable struct {
	mu    sync.Mutex
	data  map[AllocKey]*AllocEntry
	index map[uint64]map[AllocKey]struct{}
}

type AllocKey struct {
	DeviceID DeivceID
	Uid      Uid
	Pid      Pid
	Tid      Tid
	Dptr     Dptr
}

type AllocEntry struct {
	Size      AllocSize
	Stack     *StackInfo
	Timestamp Timestamp
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

	key := AllocKey{DeviceID: e.DeivceID, Tid: e.Tid, Pid: e.Pid, Dptr: e.Dptr, Uid: e.Uid}

	t.data[key] = &AllocEntry{
		Size:      AllocSize(e.Size),
		Timestamp: e.TsNs,
		Stack:     s,
	}
	// Insert to index
	h := pidTgid(key.Pid, key.Tid)
	if _, ok := t.index[h]; !ok {
		t.index[h] = make(map[AllocKey]struct{})
	}
	t.index[h][key] = struct{}{}
}

func (t *AllocTable) Free(e Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := AllocKey{DeviceID: e.DeivceID, Tid: e.Tid, Pid: e.Pid, Dptr: Dptr(e.Dptr)}
	delete(t.data, key)
}

func (t *AllocTable) Aggregate() map[Pid]Result {
	t.mu.Lock()
	defer t.mu.Unlock()

	df := &DF{}
	df.InitHeader([]Header{"TS", "DEVICEID", "PID", "UID", "COMM", "DPTR", "TID", "SID", "TOTAL"})
	for key, entry := range t.data {
		df.Insert(Row{
			Timestamp: entry.Timestamp,
			DeviceID:  key.DeviceID,
			Pid:       key.Pid,
			Uid:       key.Uid,
			Comm:      entry.Stack.T.P.Comm,
			Dptr:      key.Dptr,
			Tid:       key.Tid,
			Sid:       entry.Stack.SID,
			Size:      entry.Size,
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
			r := t.Aggregate()
			if len(r) == 0 {
				fmt.Println("NO EVENT.")
				continue
			}
			PrintResults(r)
		}
	}
}
