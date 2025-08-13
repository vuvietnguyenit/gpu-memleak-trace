package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"
)

var (
	FlagVerbose        string        // log level
	FlagLibCUDAPath    string        // path to libcuda
	FlagTracePrint     bool          // enable printAllocMapPeriodically
	FlagPrintInterval  time.Duration // interval for trace print
	FlagUpdateInterval time.Duration // metrics update interval
	FlagExportMetrics  bool          // export metrics as Prometheus exporter
)

func initFlags() {
	flag.StringVar(&FlagVerbose, "log-verbose", "INFO", "Log verbosity level (DEBUG, INFO, WARN, ERROR)")
	flag.StringVar(&FlagLibCUDAPath, "libcuda-path", "/usr/lib/x86_64-linux-gnu/libcuda.so", "Path to libcuda.so")
	flag.BoolVar(&FlagTracePrint, "trace-print", false, "Enable periodic printing of allocation map")
	flag.DurationVar(&FlagPrintInterval, "interval", 2*time.Second, "Trace print interval")
	flag.DurationVar(&FlagUpdateInterval, "update-interval", 2*time.Second, "Metrics update interval")
	flag.BoolVar(&FlagExportMetrics, "export-metrics", false, "Export metrics as Prometheus exporter")
	flag.Parse()

	if !FlagTracePrint && !FlagExportMetrics {
		fmt.Fprintln(os.Stderr, "Error: You must enable either --trace-print or --export-metrics")
		flag.Usage()
		os.Exit(1)
	}

	// Configure slog logger level
	var level slog.Level
	switch FlagVerbose {
	case "DEBUG":
		level = slog.LevelDebug
	case "WARN":
		level = slog.LevelWarn
	case "ERROR":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	initLogger(level)
}
