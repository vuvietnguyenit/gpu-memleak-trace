package main

import (
	"log/slog"
	"os"
)

func initLogger(level slog.Level) {
	// Explicitly use text handler
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level, // set log level
	}))
	slog.SetDefault(logger)
}
