package main

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

type TableKey struct {
	DeviceID DeviceID
	Uid      Uid
	Pid      Pid
	Comm     Comm
	Tid      Tid
}

type AllocData struct {
	TotalSize AllocSize
	LastTs    Timestamp
}

type AllocateResult struct {
	mu   sync.RWMutex
	data map[TableKey]AllocData
}

// Constructor
func NewAllocateResult() *AllocateResult {
	return &AllocateResult{
		data: make(map[TableKey]AllocData),
	}
}

func (ar *AllocateResult) Add(key TableKey, size AllocSize, ts Timestamp) {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	current := ar.data[key]
	current.TotalSize += size
	current.LastTs = ts
	ar.data[key] = current
}

func (ar *AllocateResult) Sub(key TableKey, size AllocSize, ts Timestamp) {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	if current, ok := ar.data[key]; ok {
		if current.TotalSize <= size {
			delete(ar.data, key)
		} else {
			current.TotalSize -= size
			current.LastTs = ts
			ar.data[key] = current
		}
	}
}

// Get returns the alloc data for a key
func (ar *AllocateResult) Get(key TableKey) AllocData {
	ar.mu.RLock()
	defer ar.mu.RUnlock()
	return ar.data[key]
}

// Delete removes a key entirely
func (ar *AllocateResult) Delete(key TableKey) {
	ar.mu.Lock()
	defer ar.mu.Unlock()
	delete(ar.data, key)
}

// Summary returns a copy of the current state
func (ar *AllocateResult) Summary() map[TableKey]AllocData {
	ar.mu.RLock()
	defer ar.mu.RUnlock()
	snapshot := make(map[TableKey]AllocData, len(ar.data))
	for k, v := range ar.data {
		snapshot[k] = v
	}
	return snapshot
}

type AllocateTable struct {
	mu       sync.RWMutex
	tEvents  *TopAllocTracker // top-N allocations per Tid
	allocRes *AllocateResult  // summary per Pid
	topN     int              // max entries per Tid heap
}

func NewAllocTableV2(topN int) *AllocateTable {
	return &AllocateTable{
		tEvents:  NewTopAllocTracker(topN),
		allocRes: NewAllocateResult(),
		topN:     topN,
	}
}

func (t *AllocateTable) StartPrinter(interval time.Duration, ctx context.Context) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if len(t.allocRes.data) == 0 {
					continue
				}
				t.Print()
			case <-ctx.Done():
				slog.Debug("Stopping Print action...")
				return
			}
		}
	}()
}

// Print shows the current state of AllocateTable
func (t *AllocateTable) Print() {
	t.mu.RLock()
	defer t.mu.RUnlock()

	fmt.Printf("--- Scan Time: %s ---\n\n", time.Now().Format("2006-01-02 15:04:05"))
	// print summaries
	summary := t.allocRes.Summary()
	for k, data := range summary {
		fmt.Printf("PID=%d TID=%d UID=%d DEV=%d Comm=%s -> TotalSize=%s LastTs=%s\n",
			k.Pid, k.Tid, k.Uid, k.DeviceID, k.Comm[:],
			data.TotalSize.HumanSize(), data.LastTs.HumanTime().Format("2006-01-02 15:04:05.000000000"))
	}

	// optionally print heaps (top-N allocations per TID)

	for tid := range t.tEvents.data {
		fmt.Printf("Top allocations for TID=%d:\n", tid)
		evs := t.tEvents.GetTopN(tid)
		for _, e := range evs {
			fmt.Printf("  Size=%s, Ptr=0x%016x\n", e.Size.HumanSize(), e.Dptr)
		}
	}
	fmt.Println()
}

func (t *AllocateTable) Malloc(ev Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := TableKey{ev.DeviceID, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

	// update AllocateRes
	t.allocRes.Add(key, ev.Size, ev.TsNs)
	te := TEvent{Tid: ev.Tid, Size: ev.Size, Dptr: ev.Dptr}
	// update top-N heap
	t.tEvents.AddEvent(te)
}

func (t *AllocateTable) Free(ev Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := TableKey{ev.DeviceID, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

	// update summary
	t.allocRes.Sub(key, ev.Size, ev.TsNs)

	te := TEvent{Tid: ev.Tid, Size: ev.Size, Dptr: ev.Dptr}
	// remove from heap
	t.tEvents.DelEvent(te)
}

func (t *AllocateTable) CleanupStale(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			t.cleanupOnce()
		case <-ctx.Done():
			slog.Debug("Stopping Cleanup action...")
			return
		}
	}
}

func (t *AllocateTable) cleanupOnce() {
	t.mu.Lock()
	defer t.mu.Unlock()

	// check each key in allocRes
	for key := range t.allocRes.data {
		if !pidExists(int(key.Pid)) {
			// stale PID -> remove
			delete(t.allocRes.data, key)

			// also remove its heap (TID related)
			delete(t.tEvents.data, key.Tid)
		}
	}

	// also scan heaps for TIDs not linked to any live PID
	for tid := range t.tEvents.data {
		// if no PID maps to this tid, drop it
		found := false
		for key := range t.allocRes.data {
			if key.Tid == tid {
				found = true
				break
			}
		}
		if !found {
			delete(t.tEvents.data, tid)
		}
	}
}

type LookupRecord struct {
	Key  TableKey
	Data AllocData
	Dptr *Dptr // optional, nil if no pointer
}

// Lookup returns either all or filtered records
func (t *AllocateTable) Lookup(key TableKey) []LookupRecord {
	t.mu.RLock()
	defer t.mu.RUnlock()

	records := []LookupRecord{}

	summary := t.allocRes.Summary()

	for k, data := range summary {
		if (key == TableKey{}) || (k == key) {
			rec := LookupRecord{Key: k, Data: data}

			// Try to find matching Dptr from heap
			if h, ok := t.tEvents.data[k.Tid]; ok {
				for _, te := range *h {
					if te.Size == data.TotalSize {
						d := te.Dptr
						rec.Dptr = &d
						break
					}
				}
			}

			records = append(records, rec)
		}
	}

	return records
}
