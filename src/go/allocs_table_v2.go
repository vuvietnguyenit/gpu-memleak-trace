package main

import (
	"container/heap"
	"sync"
)

type TableKey struct {
	DeviceID DeviceID
	Uid      Uid
	Pid      Pid
	Comm     Comm
	Tid      Tid
}

type TableEntry struct {
	Size AllocSize // allocated size
	Ts   Timestamp // last allocation timestamp
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
	tEvents  map[Tid]*MinHeap // top-N allocations per Tid
	allocRes *AllocateResult  // summary per Pid
	topN     int              // max entries per Tid heap
}

func NewAllocTableV2(topN int) *AllocateTable {
	return &AllocateTable{
		tEvents:  make(map[Tid]*MinHeap),
		allocRes: NewAllocateResult(),
		topN:     topN,
	}
}

// Insert event
func (t *AllocateTable) Insert(ev Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := TableKey{ev.DeviceID, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

	// update heap (top-N per Tid)
	h, ok := t.tEvents[ev.Tid]
	if !ok {
		h = &MinHeap{}
		heap.Init(h)
		t.tEvents[ev.Tid] = h
	}
	if h.Len() < t.topN {
		heap.Push(h, TEvent{Tid: ev.Tid, Size: ev.Size, Dptr: ev.Dptr})
	} else if (*h)[0].Size < ev.Size {
		heap.Pop(h)
		heap.Push(h, TEvent{Tid: ev.Tid, Size: ev.Size, Dptr: ev.Dptr})
	}

	// update AllocateRes
	t.allocRes.Add(key, ev.Size, ev.TsNs)
}

func (t *AllocateTable) Delete(ev Event) {
	t.mu.Lock()
	defer t.mu.Unlock()

	key := TableKey{ev.DeviceID, ev.Uid, ev.Pid, ev.Comm, ev.Tid}

	// update summary
	t.allocRes.Sub(key, ev.Size, ev.TsNs)

	// remove from heap (linear scan rebuild)
	if h, ok := t.tEvents[ev.Tid]; ok {
		newHeap := &MinHeap{}
		heap.Init(newHeap)
		for _, e := range *h {
			if e.Dptr != ev.Dptr {
				heap.Push(newHeap, e)
			}
		}
		t.tEvents[ev.Tid] = newHeap
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
			if h, ok := t.tEvents[k.Tid]; ok {
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
