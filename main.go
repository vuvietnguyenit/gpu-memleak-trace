package main

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"os/signal"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/cilium/ebpf/rlimit"
)

func main() {
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("failed to remove memlock: %v", err)
	}

	var objs MemleakCountAllocObjects
	if err := LoadMemleakCountAllocObjects(&objs, nil); err != nil {
		log.Fatalf("loading objects: %v", err)
	}
	defer objs.Close()
	fmt.Println("BPF program loaded successfully")
	fmt.Println("Starting to trace memory allocations...")
	ex, err := link.OpenExecutable(LIBC_PATH)
	if err != nil {
		log.Fatalf("opening executable: %s", err)
	}

	mallocSym, err := ex.Uprobe("malloc", objs.TraceMallocEntry, nil)
	if err != nil {
		log.Fatalf("attach malloc enter: %v", err)
	}
	defer mallocSym.Close()

	mallocRetSym, err := ex.Uretprobe("malloc", objs.TraceMallocReturn, nil)
	if err != nil {
		log.Fatalf("attach malloc exit: %v", err)
	}
	defer mallocRetSym.Close()

	freeSym, err := ex.Uprobe("free", objs.TraceFree, nil)
	if err != nil {
		log.Fatalf("attach free: %v", err)
	}
	defer freeSym.Close()
	log.Println("eBPF program running... Press Ctrl+C to exit.")
	rd, err := ringbuf.NewReader(objs.Events)
	if err != nil {
		log.Fatalf("failed to read ring buffer: %v", err)
	}
	defer rd.Close()

	// Channel to handle Ctrl+C
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	go func() {
		for {
			record, err := rd.Read()
			if err != nil {
				continue
			}

			var event Event
			if err := binary.Read(bytes.NewReader(record.RawSample), binary.LittleEndian, &event); err != nil {
				// log.Printf("failed to decode event: %v", err)
				continue
			}
			fmt.Printf("Raw sample: %x\n", record.RawSample)
			// switch event.Type {
			// case EventMalloc:
			// 	fmt.Printf("[malloc] pid=%d size=%d\n", event.Pid, event.Data)
			// case EventFree:
			// 	// fmt.Printf("[free]   pid=%d ptr=0x%x\n", event.Pid, event.Data)
			// }
		}

	}()

	<-stop

	fmt.Println("Exiting...")

}
