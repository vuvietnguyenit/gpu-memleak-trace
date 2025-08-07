package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/cilium/ebpf/rlimit"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -tags linux -cc clang -cflags "-g -O2 -D__TARGET_ARCH_x86" MemleakCountAlloc bpf/memleak_count_alloc.bpf.c -- -I../headers

const (
	LIBC_PATH = "/usr/lib/x86_64-linux-gnu/libc.so.6"
)

func init() {
	// Ensure the BPF program is compiled with the correct tags and options.
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("failed to remove memlock: %v", err)
	}
	fmt.Println("BPF memory lock removed, ready to load BPF program")
	time.Sleep(1 * time.Second) // Allow some time for the system to adjust
}

const (
	EventMalloc = 0
	EventFree   = 1
)

type Event struct {
	Pid   uint32
	Tid   uint64
	Type  uint32 // Padding: enum is 4 bytes
	Data  uint64 // .ptr or .size
	IsRet uint8
	_     [3]byte // padding to align struct to 24 bytes
}

func loadLibcPath() *link.Executable {
	if LIBC_PATH == "" {
		log.Fatalf("LIBC_PATH is empty")
	}
	ex, err := link.OpenExecutable(LIBC_PATH)
	if err != nil {
		log.Fatalf("failed to open executable: %s", err)
	}
	return ex
}

func runMemleakTraceProgram() {
	stopper := make(chan os.Signal, 1)
	signal.Notify(stopper, os.Interrupt, syscall.SIGTERM)
	var objs MemleakCountAllocObjects
	if err := LoadMemleakCountAllocObjects(&objs, nil); err != nil {
		log.Fatalf("loading objects: %v", err)
	}
	defer objs.Close()
	log.Println("Load MemleakCountAllocObjects successfully")
	ex := loadLibcPath()

	ex.Uprobe("malloc", objs.TraceMallocEntry, nil)
	ex.Uretprobe("malloc", objs.TraceMallocReturn, nil)
	ex.Uprobe("free", objs.TraceFree, nil)

	rb, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("opening ringbuf: %v", err)
	}
	defer rb.Close()
	go func() {
		<-stopper

		if err := rb.Close(); err != nil {
			log.Fatalf("closing ringbuf reader: %s", err)
		}
	}()

	log.Println("eBPF program running... Press Ctrl+C to exit.")
	for {
		record, err := rb.Read()
		if err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				log.Println("Received signal, exiting..")
				return
			}
			log.Printf("reading from reader: %s", err)
			continue
		}
		var e Event

		// Parse the ringbuf event entry into a bpfEvent structure.
		if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &e); err != nil {
			log.Printf("parsing ringbuf event: %s", err)
			continue
		}
		fmt.Println("Event:", e.Type, "PID:", e.Pid, "TID:", e.Tid, "Data:", e.Data, "IsRet:", e.IsRet)

	}

}
