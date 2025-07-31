PROG := memtraceprog


all: build

# Generate eBPF code from .bpf.c
build:
	go generate $(GO_SRC)

# Run the Go program (after building)
run: build
	go run .

# Clean generated files
clean:
	rm -f $(PROG)_*.o $(PROG)_*.go