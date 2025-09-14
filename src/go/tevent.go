package main

import (
	"container/heap"
	"sort"
)

// TEvent represents your allocation event
type TEvent struct {
	Tid  Tid
	Size AllocSize
	Dptr Dptr // pointer value (optional, useful for debugging)
}

// ---- Heap implementation ----
type MinHeap []TEvent

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].Size < h[j].Size } // smallest first
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(TEvent)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// ---- Tracker ----
type TopAllocTracker struct {
	N    int
	data map[Tid]*MinHeap // key = TID, value = min-heap
}

func NewTopAllocTracker(n int) *TopAllocTracker {
	return &TopAllocTracker{
		N:    n,
		data: make(map[Tid]*MinHeap),
	}
}

func (t *TopAllocTracker) AddEvent(ev TEvent) {
	h, ok := t.data[ev.Tid]
	if !ok {
		h = &MinHeap{}
		heap.Init(h)
		t.data[ev.Tid] = h
	}

	if h.Len() < t.N {
		heap.Push(h, ev)
	} else if (*h)[0].Size < ev.Size {
		heap.Pop(h)      // remove smallest
		heap.Push(h, ev) // push new one
	}
}

func (t *TopAllocTracker) GetTopN(tid Tid) []TEvent {
	h, ok := t.data[tid]
	if !ok {
		return nil
	}
	// copy and sort descending
	result := make([]TEvent, len(*h))
	copy(result, *h)
	sort.Slice(result, func(i, j int) bool {
		return result[i].Size > result[j].Size
	})
	return result
}
