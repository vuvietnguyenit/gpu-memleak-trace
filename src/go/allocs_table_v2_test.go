package main

import (
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
