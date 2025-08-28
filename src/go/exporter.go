package main

import (
	"context"
	"fmt"
	"log/slog"
)

func startPrometheusExporter(ctx context.Context) {
	slog.Info("Starting Prometheus exporter...")
	for {
		select {
		case <-ctx.Done():
			slog.Debug("Stop exporter")
			return
		default:
			fmt.Println("Nothing")
		}
	}
	// TODO: Implement metrics server
}
