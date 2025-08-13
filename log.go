package main

import (
	"log/slog"
	"os"
)

func init() {
	// Explicitly use text handler
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug, // set log level
	}))
	slog.SetDefault(logger)
}
