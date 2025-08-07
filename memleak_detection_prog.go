package main

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"log"
	"os"
	"os/signal"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/cilium/ebpf/rlimit"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -tags linux -cc clang -cflags "-g -O2 -D__TARGET_ARCH_x86" CudaMemleakDelection bpf/cuda_memleak_detection.bpf.c -- -I../headers
const (
	LIBCUDA_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so"
)

func init() {
	// Ensure we can lock memory for the eBPF program
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("failed to remove memlock: %v", err)
	}
}

type EventType int32

const (
	EVENT_MALLOC EventType = 0
	EVENT_FREE   EventType = 1
)

type AllocEvent struct {
	Pid       uint32
	_         uint32 // padding to align to 8-byte boundary
	Size      uint64
	Dptr      uint64
	EventType EventType
	Retval    int32
}

func main() {

	var objs CudaMemleakDelectionObjects
	if err := LoadCudaMemleakDelectionObjects(&objs, nil); err != nil {
		log.Fatalf("loading objects: %v", err)
	}
	defer objs.Close()
	log.Println("BPF program loaded successfully")
	log.Println("Starting to trace memory allocations...")
	ex, err := link.OpenExecutable(LIBCUDA_PATH)
	if err != nil {
		log.Fatalf("opening executable: %s", err)
	}

	mallocSym, err := ex.Uprobe("cuMemAlloc_v2", objs.TraceCuMemAllocEntry, nil)
	if err != nil {
		log.Fatalf("attach malloc enter: %v", err)
	}
	defer mallocSym.Close()

	mallocRetSym, err := ex.Uretprobe("cuMemAlloc_v2", objs.TraceMallocReturn, nil)
	if err != nil {
		log.Fatalf("attach malloc exit: %v", err)
	}
	defer mallocRetSym.Close()

	freeRetSym, err := ex.Uprobe("cuMemFree_v2", objs.TraceCuMemFree, nil)
	if err != nil {
		log.Fatalf("attach free exit: %v", err)
	}
	defer freeRetSym.Close()

	log.Println("eBPF program running... Press Ctrl+C to exit.")
	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("failed to read ring buffer: %v", err)
	}
	defer rd.Close()

	// Channel to handle Ctrl+C
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	go func() {
		for {
			record, err := rd.Read()
			if err != nil {
				log.Fatalf("ringbuf read failed: %v", err)
			}

			var e AllocEvent
			if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &e); err != nil {
				log.Printf("failed to parse event: %v", err)
				continue
			}
			log.Printf("Received event: %+v\n", e)
			switch e.EventType {
			case EVENT_MALLOC:
			case EVENT_FREE:
			default:
			}
		}

	}()

	<-stop

	log.Println("Exiting...")

}
