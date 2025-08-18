package main

import (
	"log/slog"
	"os"
)

func initLogger() {
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
	// Explicitly use text handler
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level, // set log level
	}))
	slog.SetDefault(logger)
}
