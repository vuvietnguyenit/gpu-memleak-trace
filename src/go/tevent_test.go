package main

import (
	"testing"
)

func TestTopAllocTrackerBasic(t *testing.T) {
	tracker := NewTopAllocTracker(3)

	events := []TEvent{
		{Tid: 1, Size: 100},
		{Tid: 1, Size: 200},
		{Tid: 1, Size: 50},
	}

	for _, e := range events {
		tracker.AddEvent(e)
	}

	top := tracker.GetTopN(1)

	if len(top) != 3 {
		t.Fatalf("expected 3 events, got %d", len(top))
	}

	// Expect descending order: 200, 100, 50
	want := []AllocSize{200, 100, 50}
	for i, e := range top {
		if e.Size != want[i] {
			t.Errorf("unexpected order at %d: got %d, want %d", i, e.Size, want[i])
		}
	}
}

func TestTopAllocTrackerOverflow(t *testing.T) {
	tracker := NewTopAllocTracker(3)

	events := []TEvent{
		{Tid: 1, Size: 100},
		{Tid: 1, Size: 200},
		{Tid: 1, Size: 50},
		{Tid: 1, Size: 500}, // should kick out 50
	}

	for _, e := range events {
		tracker.AddEvent(e)
	}

	top := tracker.GetTopN(1)

	if len(top) != 3 {
		t.Fatalf("expected 3 events, got %d", len(top))
	}

	// Expect descending: 500, 200, 100
	want := []AllocSize{500, 200, 100}
	for i, e := range top {
		if e.Size != want[i] {
			t.Errorf("unexpected order at %d: got %d, want %d", i, e.Size, want[i])
		}
	}
}

func TestTopAllocTrackerMultipleTIDs(t *testing.T) {
	tracker := NewTopAllocTracker(2)

	events := []TEvent{
		{Tid: 1, Size: 100},
		{Tid: 1, Size: 200},
		{Tid: 2, Size: 300},
		{Tid: 2, Size: 50},
		{Tid: 2, Size: 400}, // should kick out 50
	}

	for _, e := range events {
		tracker.AddEvent(e)
	}

	top1 := tracker.GetTopN(1)
	want1 := []AllocSize{200, 100}
	if len(top1) != 2 {
		t.Fatalf("tid=1 expected 2 events, got %d", len(top1))
	}
	for i, e := range top1 {
		if e.Size != want1[i] {
			t.Errorf("tid=1 unexpected order at %d: got %d, want %d", i, e.Size, want1[i])
		}
	}

	top2 := tracker.GetTopN(2)
	want2 := []AllocSize{400, 300}
	if len(top2) != 2 {
		t.Fatalf("tid=2 expected 2 events, got %d", len(top2))
	}
	for i, e := range top2 {
		if e.Size != want2[i] {
			t.Errorf("tid=2 unexpected order at %d: got %d, want %d", i, e.Size, want2[i])
		}
	}
}
