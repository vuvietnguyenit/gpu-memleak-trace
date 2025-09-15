package main

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"
)

type TableKey struct {
	DeviceID DeviceID
	Dptr     Dptr
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
	delete(ar.data, key)
}

// Get returns the alloc data for a key
func (ar *AllocateResult) Get(key TableKey) AllocData {
	ar.mu.RLock()
	defer ar.mu.RUnlock()
	return ar.data[key]
}

type SummaryRow struct {
	DeviceID  DeviceID
	Uid       Uid
	Pid       Pid
	Tid       Tid
	Comm      Comm
	TotalSize AllocSize
	LastTs    Timestamp
}

func (ar *AllocateResult) Summary() []SummaryRow {
	ar.mu.RLock()
	defer ar.mu.RUnlock()

	grouped := make(map[string]*SummaryRow)

	for key, val := range ar.data {
		// Create composite key (string) to group by DeviceID, Uid, Pid, Tid, Comm
		gk := fmt.Sprintf("%d:%d:%d:%d:%s",
			key.DeviceID, key.Uid, key.Pid, key.Tid, key.Comm)

		if row, ok := grouped[gk]; ok {
			row.TotalSize += val.TotalSize
			if val.LastTs > row.LastTs {
				row.LastTs = val.LastTs
			}
		} else {
			grouped[gk] = &SummaryRow{
				DeviceID:  key.DeviceID,
				Uid:       key.Uid,
				Pid:       key.Pid,
				Tid:       key.Tid,
				Comm:      key.Comm,
				TotalSize: val.TotalSize,
				LastTs:    val.LastTs,
			}
		}
	}

	// Convert to slice
	result := make([]SummaryRow, 0, len(grouped))
	for _, row := range grouped {
		result = append(result, *row)
	}

	// Sort (descending by TotalSize)
	sort.Slice(result, func(i, j int) bool {
		return result[i].TotalSize > result[j].TotalSize
	})

	return result
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
	for _, row := range summary {
		fmt.Printf("PID=%d TID=%d UID=%d DEV=%d Comm=%s -> TotalSize=%s LastTs=%s\n",
			row.Pid, row.Tid, row.Uid, row.DeviceID, row.Comm[:],
			row.TotalSize.HumanSize(), row.LastTs.HumanTime().Format("2006-01-02 15:04:05.000000000"))
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

	key := TableKey{ev.DeviceID, ev.Dptr, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

	// update AllocateRes
	t.allocRes.Add(key, ev.Size, ev.TsNs)
	te := TEvent{Tid: ev.Tid, Size: ev.Size, Dptr: ev.Dptr}
	// update top-N heap
	t.tEvents.AddEvent(te)
}

func (t *AllocateTable) Free(ev Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := TableKey{ev.DeviceID, ev.Dptr, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

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

	livePIDs, err := getLivePIDs()
	if err != nil {
		slog.Warn("failed to read /proc", "err", err)
		return
	}

	// check each key in allocRes
	for key := range t.allocRes.data {
		if _, ok := livePIDs[key.Pid]; !ok {
			delete(t.allocRes.data, key)
			delete(t.tEvents.data, key.Tid)
		}
	}

	// Build a set of TIDs still alive in allocRes
	liveTids := make(map[Tid]struct{})
	for key := range t.allocRes.data {
		liveTids[key.Tid] = struct{}{}
	}

	// Remove stale TIDs
	for tid := range t.tEvents.data {
		if _, ok := liveTids[tid]; !ok {
			delete(t.tEvents.data, tid)
		}
	}
}
