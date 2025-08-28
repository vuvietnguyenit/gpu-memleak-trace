ARCH := x86
BINARY := gpu-memleak-trace
SRC     := ./src
BUILD   := ./bin
GOSRC   := $(SRC)/go

# Go related variables
GO      ?= go
GOFLAGS :=
LDFLAGS := -s -w

.PHONY: all build run clean

all: build

# Generate eBPF code from .bpf.c
bpf-gen:
	@echo ">> Generate eBPF code from .bpf.c"
	go generate $(GOSRC)

## Build binary
build: bpf-gen
	@echo ">> Building $(BINARY)"
	@mkdir -p $(BUILD)
	$(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BUILD)/$(BINARY) $(GOSRC)

# Clean generated files
clean:
	rm -f bpf_$(ARCH)_*.o bpf_$(PROG)_*.go
	rm -rf $(BINARY)