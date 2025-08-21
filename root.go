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

	var wg WG
	allocsData := NewAllocTable()

	if !FlagDebug {
		// Debug mode: print events, no periodic metrics or exporters
		slog.Info("Running in DEBUG mode: ignoring --trace-print and --export-metrics")
		if FlagTracePrint {
			wg.Go(func() { allocsData.Print(ctx) })
		}
		if FlagExportMetrics {
			wg.Go(func() { startPrometheusExporter(ctx) })
		}
		wg.Go(func() { allocsData.Cleanup(ctx) })
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
