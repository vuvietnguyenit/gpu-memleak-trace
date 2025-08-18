//go:build amd64 && linux

package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/cilium/ebpf/link"
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
		rb := RingBuffer{
			Event:     objs.Events,
			AllocsMap: allocsData,
		}
		rb.RbReserve(ctx)

	}()

	<-stopper

	log.Println("Shutting down gracefully...")
	cancel()
	wg.Wait()
	log.Println("All goroutines stopped.")
}
