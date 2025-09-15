package main

import (
	"sync"
	"testing"
	"time"
)

func StringToBytes(s string, n int) []byte {
	b := []byte(s)
	if len(b) > n {
		return b[:n]
	}
	// pad with zeros if needed
	padded := make([]byte, n)
	copy(padded, b)
	return padded
}
func TestAllocateResult_AddAndSub(t *testing.T) {
	ar := NewAllocateResult()
	key := TableKey{DeviceID: 1, Uid: 10, Pid: 20, Tid: 30, Comm: Comm(StringToBytes("test", 16))}
	ts := Timestamp(time.Now().UnixNano())

	// Add
	ar.Add(key, 512, ts)
	data := ar.Get(key)
	if data.TotalSize != 512 {
		t.Errorf("expected TotalSize=512, got %d", data.TotalSize)
	}
	if data.LastTs != ts {
		t.Errorf("expected LastTs=%v, got %v", ts, data.LastTs)
	}

	// Sub (partial)
	ar.Sub(key, 128, ts+1)
	data = ar.Get(key)
	if data.TotalSize != 384 {
		t.Errorf("expected TotalSize=384, got %d", data.TotalSize)
	}
	if data.LastTs != ts+1 {
		t.Errorf("expected LastTs=%v, got %v", ts+1, data.LastTs)
	}

	// Sub (remove completely)
	ar.Sub(key, 384, ts+2)
	data = ar.Get(key)
	if data.TotalSize != 0 {
		t.Errorf("expected TotalSize=0, got %d", data.TotalSize)
	}
}

func TestAllocateResult_Delete(t *testing.T) {
	ar := NewAllocateResult()
	key := TableKey{DeviceID: 2, Uid: 11, Pid: 21, Tid: 31, Comm: Comm(StringToBytes("delete_test", 16))}
	ts := Timestamp(time.Now().UnixNano())

	ar.Add(key, 100, ts)
	ar.Delete(key)

	data := ar.Get(key)
	if data.TotalSize != 0 {
		t.Errorf("expected TotalSize=0 after delete, got %d", data.TotalSize)
	}
}

func TestAllocateResult_Summary(t *testing.T) {
	ar := NewAllocateResult()
	key1 := TableKey{DeviceID: 3, Uid: 12, Pid: 22, Tid: 32, Comm: Comm(StringToBytes("one", 16))}
	key2 := TableKey{DeviceID: 4, Uid: 13, Pid: 23, Tid: 33, Comm: Comm(StringToBytes("two", 16))}
	ts := Timestamp(time.Now().UnixNano())

	ar.Add(key1, 50, ts)
	ar.Add(key2, 75, ts+1)

	summary := ar.Summary()
	if len(summary) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(summary))
	}
	if summary[key1].TotalSize != 50 {
		t.Errorf("expected key1=50, got %d", summary[key1].TotalSize)
	}
	if summary[key2].TotalSize != 75 {
		t.Errorf("expected key2=75, got %d", summary[key2].TotalSize)
	}
}

func TestAllocateTable_InsertAndLookup(t *testing.T) {
	at := NewAllocTableV2(3)
	key := TableKey{DeviceID: 5, Uid: 14, Pid: 24, Tid: 34, Comm: Comm(StringToBytes("lookup_test", 16))}
	ptr := Dptr(0xcafebabe)
	ts := Timestamp(time.Now().UnixNano())

	// Insert
	ev := Event{DeviceID: key.DeviceID, Uid: key.Uid, Pid: key.Pid, Comm: key.Comm, Tid: key.Tid, Size: 256, Dptr: ptr, TsNs: ts}
	at.Malloc(ev)

	// Lookup all
	records := at.Lookup(TableKey{})
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	rec := records[0]
	if rec.Key != key {
		t.Errorf("expected key=%v, got %v", key, rec.Key)
	}
	if rec.Data.TotalSize != 256 {
		t.Errorf("expected TotalSize=256, got %d", rec.Data.TotalSize)
	}
	if rec.Dptr == nil || *rec.Dptr != ptr {
		t.Errorf("expected Dptr=%v, got %v", ptr, rec.Dptr)
	}
}

func TestAllocateTable_Delete(t *testing.T) {
	at := NewAllocTableV2(3)
	key := TableKey{DeviceID: 6, Uid: 15, Pid: 25, Tid: 35, Comm: Comm(StringToBytes("delete_test", 16))}
	ptr := Dptr(0xdeadbeef)
	ts := Timestamp(time.Now().UnixNano())

	ev := Event{DeviceID: key.DeviceID, Uid: key.Uid, Pid: key.Pid, Comm: key.Comm, Tid: key.Tid, Size: 128, Dptr: ptr, TsNs: ts}
	at.Malloc(ev)

	// Delete
	at.Free(ev)

	records := at.Lookup(key)
	if len(records) != 0 {
		t.Errorf("expected no records after delete, got %d", len(records))
	}
}

func TestAllocateTable_Concurrency(t *testing.T) {
	at := NewAllocTableV2(5)
	key := TableKey{DeviceID: 7, Uid: 16, Pid: 26, Tid: 36, Comm: Comm(StringToBytes("concurrency", 16))}
	ptr := Dptr(0x12345678)
	ts := Timestamp(time.Now().UnixNano())

	done := make(chan struct{})
	const numGoroutines = 100
	for i := 0; i < numGoroutines; i++ {
		go func(i int) {
			ev := Event{
				DeviceID: key.DeviceID, Uid: key.Uid, Pid: key.Pid,
				Comm: key.Comm, Tid: key.Tid,
				Size: 64, Dptr: ptr, TsNs: ts + Timestamp(i),
			}
			at.Malloc(ev)
			done <- struct{}{}
		}(i)
	}
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	records := at.Lookup(key)
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	if records[0].Data.TotalSize != 6400 {
		t.Errorf("expected TotalSize=6400, got %d", records[0].Data.TotalSize)
	}
}

func TestAllocateTable_ConcurrencyInsertDeleteCycle(t *testing.T) {
	at := NewAllocTableV2(5)
	key := TableKey{
		DeviceID: 42,
		Uid:      1001,
		Pid:      2002,
		Tid:      3003,
		Comm:     Comm(StringToBytes("cycle", 16)),
	}
	ptr := Dptr(0xbeefcafe)
	ts := Timestamp(time.Now().UnixNano())

	const numGoroutines = 10
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				ev := Event{
					DeviceID: key.DeviceID,
					Uid:      key.Uid,
					Pid:      key.Pid,
					Comm:     key.Comm,
					Tid:      key.Tid,
					Size:     64,
					Dptr:     ptr,
					TsNs:     ts + Timestamp(g*iterations+i),
				}
				// Insert
				at.Malloc(ev)
				// Delete (simulate free right after alloc)
				at.Free(ev)
			}
		}(g)
	}

	wg.Wait()

	records := at.Lookup(key)

	// After balanced insert/delete cycles, we expect no leftover allocation
	if len(records) != 0 {
		t.Fatalf("expected no records after balanced insert/delete, got %d", len(records))
	}
}
