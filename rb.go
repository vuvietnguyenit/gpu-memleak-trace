package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/ringbuf"
)

type RingBuffer struct {
	Event     *ebpf.Map
	AllocsMap *AllocMap
}

func (r *RingBuffer) RbReserve(ctx context.Context) {
	rd, err := ringbuf.NewReader(r.Event)
	if err != nil {
		log.Fatalf("failed to read ring buffer: %v", err)
	}
	defer rd.Close()

	records := make(chan ringbuf.Record)
	errs := make(chan error)

	go func() {
		for {
			record, err := rd.Read()
			if err != nil {
				errs <- err
				return
			}
			records <- record
		}
	}()

	for {
		select {
		case <-ctx.Done():
			rd.Close()
			slog.Debug("Stopping consume ringbuffer...")
			return
		case err := <-errs:
			if errors.Is(err, ringbuf.ErrClosed) {
				slog.Debug("Ringbuffer reader closed")
				return
			}
			log.Fatalf("ringbuf read failed: %v", err)
		case record := <-records:
			var e Event
			if err := binary.Read(bytes.NewBuffer(record.RawSample), binary.LittleEndian, &e); err != nil {
				slog.Warn("failed to parse event", "error", err)
				continue
			}
			if FlagDebug {
				if FlagPrintMallocOnly && e.EventType != EVENT_MALLOC {
					return
				}
				if FlagPrintJSON {
					data, _ := json.MarshalIndent(e, "", "  ")
					fmt.Println(string(data))
				} else if FlagPrintEvents {
					fmt.Println(e)
				}
			}
			switch e.EventType {
			case EVENT_MALLOC:
				r.AllocsMap.AddAlloc(e.Pid, e.Dptr, e.Size)
			case EVENT_FREE:
				r.AllocsMap.FreeAlloc(e.Pid, e.Dptr)
			default:
				slog.Warn("unknown event type", "etype", e.EventType)
			}
		}

	}

}
