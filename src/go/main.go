//go:build amd64 && linux

package main

import (
	"log"
	"os"

	"github.com/cilium/ebpf/rlimit"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -tags linux -target amd64 bpf ../bpf/cuda_memleak_detection.bpf.c -- -I../headers

func init() {
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("failed to remove memlock: %v", err)
	}
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}
