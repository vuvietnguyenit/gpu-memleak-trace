package main

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/pflag"
)

var (
	FlagVerbose        string        // log level
	FlagLibCUDAPath    string        // path to libcuda
	FlagTracePrint     bool          // enable printAllocMapPeriodically
	FlagPrintInterval  time.Duration // interval for trace print
	FlagUpdateInterval time.Duration // metrics update interval
	FlagExportMetrics  bool          // export metrics as Prometheus exporter
)

func parseIntervalFlag(name string) time.Duration {
	raw := pflag.Lookup(name).Value.String()
	if strings.HasSuffix(raw, "s") || strings.HasSuffix(raw, "m") || strings.HasSuffix(raw, "h") {
		if dur, err := time.ParseDuration(raw); err == nil {
			return dur
		}
	}
	if sec, err := strconv.Atoi(raw); err == nil {
		return time.Duration(sec) * time.Second
	}
	if dur, err := time.ParseDuration(raw); err == nil {
		return dur
	}
	return 0
}

func initFlags() {
	pflag.StringVar(&FlagVerbose, "log-verbose", "INFO", "Log verbosity level (DEBUG, INFO, WARN, ERROR)")
	pflag.StringVar(&FlagLibCUDAPath, "libcuda-path", "/usr/lib/x86_64-linux-gnu/libcuda.so", "Path to libcuda.so")
	pflag.BoolVar(&FlagTracePrint, "trace-print", false, "Enable periodic printing of allocation map")
	pflag.DurationVar(&FlagPrintInterval, "interval", 2*time.Second, "Trace print interval")
	pflag.DurationVar(&FlagUpdateInterval, "update-interval", 2*time.Second, "Metrics update interval")
	pflag.BoolVar(&FlagExportMetrics, "export-metrics", false, "Export metrics as Prometheus exporter")
	pflag.Parse()

	FlagPrintInterval = parseIntervalFlag("interval")
	FlagUpdateInterval = parseIntervalFlag("update-interval")

	if !FlagTracePrint && !FlagExportMetrics {
		fmt.Fprintln(os.Stderr, "Error: You must enable either --trace-print or --export-metrics")
		pflag.Usage()
		os.Exit(1)
	}
	if !FlagTracePrint && pflag.Lookup("interval").Changed {
		fmt.Fprintln(os.Stderr, "\nError: --interval can only be used with --trace-print")
		pflag.Usage()
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
