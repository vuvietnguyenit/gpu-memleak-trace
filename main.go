//go:build amd64 && linux

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"log"
	"os"
	"os/signal"
	"sync"
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

func main() {
	initFlags()
	ctx, cancel := context.WithCancel(context.Background())

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
	var wg sync.WaitGroup

	allocsData := NewAllocMap()
	if FlagTracePrint {
		wg.Add(1)
		go func() {
			defer wg.Done()
			allocsData.printAllocMapPeriodically(ctx)
		}()
	}
	if FlagExportMetrics {
		wg.Add(1)
		go func() {
			defer wg.Done()
			startPrometheusExporter(ctx)
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		allocsData.CleanupExited(ctx)
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopper:
				return
			default:
				record, err := rd.Read()
				if err != nil {
					log.Fatalf("ringbuf read failed: %v", err)
				}

				var e Event
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

		}

	}()

	<-stopper

	log.Println("Shutting down gracefully...")
	cancel()
	wg.Wait()
	log.Println("All goroutines stopped.")
}
