package main

import "github.com/spf13/cobra"

func addDebugFlags(cmd *cobra.Command) {
	cmd.PersistentFlags().BoolVar(&FlagDebug, "debug", false, "Enable debug mode")

	cmd.PersistentFlags().BoolVar(&FlagPrintEvents, "print-events", false, "Print all events (requires --debug)")
	cmd.PersistentFlags().BoolVar(&FlagPrintMallocOnly, "print-malloc-only-events", false, "Print only malloc events (requires --debug)")
	cmd.PersistentFlags().BoolVar(&FlagPrintJSON, "print-json", false, "Print events as JSON (requires --debug)")
}
