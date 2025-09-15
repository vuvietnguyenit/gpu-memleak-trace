package main

import (
	"os"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sys/unix"
)

type WG struct {
	sync.WaitGroup
}

func (wg *WG) Go(f func()) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		f()
	}()
}

var (
	offsetOnce       sync.Once
	monoToRealOffset int64
)

// initOffset calculates (CLOCK_REALTIME - CLOCK_MONOTONIC)
func initOffset() {
	var rt unix.Timespec
	var mono unix.Timespec

	_ = unix.ClockGettime(unix.CLOCK_REALTIME, &rt)
	_ = unix.ClockGettime(unix.CLOCK_MONOTONIC, &mono)

	rtNs := rt.Sec*1_000_000_000 + int64(rt.Nsec)
	monoNs := mono.Sec*1_000_000_000 + int64(mono.Nsec)

	monoToRealOffset = rtNs - monoNs
}

// KtimeToTime converts a bpf_ktime_get_ns() value into wall-clock time.Time
func KtimeToTime(tsNs Timestamp) time.Time {
	offsetOnce.Do(initOffset)
	return time.Unix(0, int64(tsNs)+monoToRealOffset)
}

func pidExists(pid int) bool {
	_, err := os.Stat("/proc/" + strconv.Itoa(pid))
	return err == nil
}
