package main

import (
	"testing"
)

func TestGroupAlloc_SinglePID(t *testing.T) {
	df := &DF{}
	df.InitHeader([]Header{"PID", "COMM", "DPTR", "TID", "SID", "TOTAL"})

	df.Insert(Row{PID: 100, Comm: "python", Dptr: "0xabc", Tid: 100, Sid: 1, Total: 10})
	df.Insert(Row{PID: 100, Comm: "python", Dptr: "0xabc", Tid: 100, Sid: 1, Total: 5}) // duplicate
	df.Insert(Row{PID: 100, Comm: "worker", Dptr: "0xdef", Tid: 101, Sid: 2, Total: 20})

	df.GroupAlloc().Print()
}

func TestGroupAlloc_MultiPID(t *testing.T) {
	df := &DF{}
	df.InitHeader([]Header{"PID", "COMM", "DPTR", "TID", "SID", "TOTAL"})

	df.Insert(Row{PID: 200, Comm: "python", Dptr: "0xaaa", Tid: 200, Sid: 1, Total: 50})
	df.Insert(Row{PID: 201, Comm: "java", Dptr: "0xbbb", Tid: 201, Sid: 2, Total: 100})
	df.GroupAlloc().Print()
	if len(df.Rows) != 2 {
		t.Fatalf("expected 2 rows inserted, got %d", len(df.Rows))
	}
}

func TestGroupAlloc_MultiplePIDs(t *testing.T) {
	df := &DF{}
	df.Insert(Row{PID: 100, Comm: "python", Dptr: "0xaaaa", Tid: 100, Sid: 1, Total: 10})
	df.Insert(Row{PID: 100, Comm: "python", Dptr: "0xbbbb", Tid: 100, Sid: 1, Total: 20})
	df.Insert(Row{PID: 200, Comm: "worker", Dptr: "0xcccc", Tid: 201, Sid: 2, Total: 5})
	df.Insert(Row{PID: 200, Comm: "worker", Dptr: "0xcccc", Tid: 201, Sid: 2, Total: 5}) // duplicate row

	got := df.GroupAlloc()

	// PID 100 should total 30
	if got.Group[100].total != 30 {
		t.Errorf("PID 100 Total = %d, want 30", got.Group[100].total)
	}
	// PID 200 should total 10 (5 + 5)
	if got.Group[200].total != 10 {
		t.Errorf("PID 200 Total = %d, want 10", got.Group[200].total)
	}

	// Deduplication check
	if len(got.Group[200].dptrs) != 1 {
		t.Errorf("PID 200 Dptrs = %v, want 1 unique", got.Group[200].dptrs)
	}
	got.Print()

}
