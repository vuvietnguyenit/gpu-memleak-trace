ARCH := x86
BINARY := gpu-memleak-trace
SRC     := ./...
BUILD   := ./bin

# Go related variables
GO      ?= go
GOFLAGS :=
LDFLAGS := -s -w

.PHONY: all build run clean

all: build

# Generate eBPF code from .bpf.c
bpf-gen:
	@echo ">> Generate eBPF code from .bpf.c"
	go generate $(GO_SRC)

## Build binary
build: bpf-gen
	@echo ">> Building $(BINARY)"
	@mkdir -p $(BUILD)
	$(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BUILD)/$(BINARY) .

# Clean generated files
clean:
	rm -f bpf_$(ARCH)_*.o bpf_$(PROG)_*.go
	rm -rf $(BINARY)