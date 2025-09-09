package main

import (
	"fmt"
	"log/slog"
	"time"

	"github.com/spf13/cobra"
)

func validateFlags() error {
	if FlagDebug {
		return nil
	}
	if !FlagTracePrint && !FlagExportMetrics {
		return fmt.Errorf("you must enable either --trace-print or --export-metrics (unless --debug is used)")
	}
	if !FlagTracePrint && RootCmd().Flags().Changed("interval") {
		return fmt.Errorf("--interval can only be used with --trace-print")
	}

	return nil
}

func addProdFlags(cmd *cobra.Command) {
	cmd.PersistentFlags().StringVar(&FlagVerbose, "log-verbose", slog.LevelInfo.String(), "Log verbosity level (DEBUG, INFO, WARN, ERROR)")
	cmd.PersistentFlags().StringVar(&FlagLibCUDAPath, "libcuda-path", "/usr/lib/x86_64-linux-gnu/libcuda.so", "Path to libcuda.so")

	cmd.PersistentFlags().BoolVar(&FlagTracePrint, "trace-print", false, "Enable periodic printing of allocation map")
	cmd.PersistentFlags().DurationVar(&FlagPrintInterval, "interval", 2*time.Second, "Trace print interval")
	cmd.PersistentFlags().DurationVar(&FlagUpdateInterval, "update-interval", 2*time.Second, "Metrics update interval")
	cmd.PersistentFlags().BoolVar(&FlagExportMetrics, "export-metrics", false, "Export metrics as Prometheus exporter")
}
