package main

import (
	"fmt"
	"log/slog"
	"os"
)

func initLogger() error {
	var level slog.Level
	switch FlagVerbose {
	case "DEBUG":
		level = slog.LevelDebug
	case "INFO":
		level = slog.LevelInfo
	case "WARN":
		level = slog.LevelWarn
	case "ERROR":
		level = slog.LevelError
	default:
		return fmt.Errorf("invalid vlevel %s", level)
	}
	// Explicitly use text handler
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: level, // set log level
	}))
	slog.SetDefault(logger)
	return nil
}
