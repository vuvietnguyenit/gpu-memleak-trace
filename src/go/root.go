package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/spf13/cobra"
)

var (
	// Global flags
	FlagVerbose     string
	FlagLibCUDAPath string

	// Debug flags
	FlagDebug           bool
	FlagPrintEvents     bool
	FlagPrintMallocOnly bool
	FlagPrintJSON       bool

	// Prod/runtime flags
	FlagTracePrint     bool
	FlagPrintInterval  time.Duration
	FlagUpdateInterval time.Duration
	FlagExportMetrics  bool
)

func RootCmd() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:   "gpu-memleak-trace",
		Short: "GPU memleak tracer",
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			if err := validateFlags(); err != nil {
				return err
			}
			if err := initLogger(); err != nil {
				return err
			}
			return nil
		},
		Run: func(cmd *cobra.Command, args []string) {
			appRun()
		},
	}

	addDebugFlags(rootCmd)
	addProdFlags(rootCmd)
	validateFlags()

	return rootCmd
}

func Execute() {
	if err := RootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func appRun() {
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

	// Attach uprobes/uretprobes to libcuda symbols
	var links []link.Link
	attach := func(sym string, prog *ebpf.Program, isRet bool) {
		var l link.Link
		var err error
		if isRet {
			l, err = ex.Uretprobe(sym, prog, nil)
		} else {
			l, err = ex.Uprobe(sym, prog, nil)
		}
		if err != nil {
			log.Fatalf("attach %s ret=%v: %v", sym, isRet, err)
		}
		links = append(links, l)
	}
	// Context management
	attach("cuCtxCreate", objs.UpCuCtxCreate, false)
	attach("cuCtxCreate", objs.UrCuCtxCreate, true)
	attach("cuDevicePrimaryCtxRetain", objs.UpCuDevicePrimaryCtxRetain, false)
	attach("cuDevicePrimaryCtxRetain", objs.UrCuDevicePrimaryCtxRetain, true)
	attach("cuCtxSetCurrent", objs.UpCuCtxSetCurrent, false)
	attach("cuCtxPushCurrent", objs.UpCuCtxPushCurrent, false)
	attach("cuCtxPopCurrent", objs.UrCuCtxPopCurrent, true)

	// Memory actions
	attach("cuMemAlloc", objs.TraceCuMemAllocEntry, false)
	attach("cuMemAlloc", objs.TraceMallocReturn, true)
	attach("cuMemAlloc_v2", objs.TraceCuMemAllocEntry, false)
	attach("cuMemAlloc_v2", objs.TraceMallocReturn, true)
	attach("cuMemFree", objs.TraceCuMemFree, false)
	attach("cuMemFree_v2", objs.TraceCuMemFree, false)

	defer func() {
		for _, l := range links {
			_ = l.Close()
		}
	}()

	log.Println("eBPF program running... Press Ctrl+C to exit.")

	var wg WG
	allocsData := NewAllocTableV2(5)

	if !FlagDebug {
		// Debug mode: print events, no periodic metrics or exporters
		slog.Info("Running in DEBUG mode: ignoring --trace-print and --export-metrics")
		wg.Go(func() { allocsData.CleanupStale(ctx, 2*time.Second) })
		if FlagTracePrint {
			wg.Go(func() { allocsData.StartPrinter(2*time.Second, ctx) })
		}
	}
	wg.Go(func() {
		rb := RingBuffer{
			Event:       objs.Events,
			AllocsTable: allocsData,
		}
		rb.RbReserve(ctx)
	})

	<-stopper

	log.Println("Shutting down gracefully...")
	cancel()
	wg.Wait()
	log.Println("All goroutines stopped.")
}
