//go:build amd64 && linux

package main

import (
	"bytes"
	"encoding/binary"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/cilium/ebpf/rlimit"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -tags linux -target amd64 bpf ebpf/cuda_memleak_detection.bpf.c -- -I../headers

func init() {
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
	initFlags()
	stopper := make(chan os.Signal, 1)
	signal.Notify(stopper, os.Interrupt, syscall.SIGTERM)

	objs := bpfObjects{}
	if err := loadBpfObjects(&objs, nil); err != nil {
		log.Fatalf("loading objects: %s", err)
	}
	defer objs.Close()
	ex, err := link.OpenExecutable(FlagLibCUDAPath)
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
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	allocsData := NewAllocMap()
	if FlagTracePrint {
		go allocsData.printAllocMapPeriodically()
	}
	if FlagExportMetrics {
		go startPrometheusExporter()
	}
	go allocsData.CleanupExited()
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
			switch e.EventType {
			case EVENT_MALLOC:
				allocsData.AddAlloc(e.Pid, e.Dptr, e.Size)
			case EVENT_FREE:
				allocsData.FreeAlloc(e.Pid, e.Dptr)
			default:
				log.Printf("Unknown event type: %d", e.EventType)
			}
		}

	}()

	<-stop

	log.Println("Exiting...")

}
