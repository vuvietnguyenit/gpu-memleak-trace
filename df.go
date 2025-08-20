package main

import (
	"fmt"
	"strings"
)

type Row struct {
	PID   uint32
	Comm  string
	Dptr  string
	Tid   uint32
	Sid   uint32
	Total Size
}

type Header string

type DF struct {
	Headers []Header
	Rows    []Row
}

func (df *DF) InitHeader(headers []Header) {
	df.Headers = headers
}

func (df *DF) Insert(r Row) {
	df.Rows = append(df.Rows, r)
}

type Agg struct {
	total Size
	comms map[string]struct{}
	dptrs map[string]struct{}
	tids  map[string]struct{}
	sids  map[string]struct{}
}

type Grouped struct {
	Group map[uint32]*Agg
}

func (gr Grouped) Print() {
	// Print results
	for pid, g := range gr.Group {
		fmt.Printf("PID: %d\n", pid)

		fmt.Println("COMM:")
		for c := range g.comms {
			fmt.Printf("  %s\n", c)
		}
		fmt.Println("DPTR:")
		for d := range g.dptrs {
			fmt.Printf("  %s\n", d)
		}
		fmt.Println("TID:")
		for t := range g.tids {
			fmt.Printf("  %s\n", t)
		}
		fmt.Println("SID:")
		for s := range g.sids {
			fmt.Printf("  %s\n", s)
		}

		fmt.Printf("TOTAL: %d\n", g.total.HumanSize())
		fmt.Println(strings.Repeat("-", 40))
	}
}

func (df *DF) GroupAlloc() *Grouped {
	grouped := &Grouped{Group: make(map[uint32]*Agg)}
	for _, r := range df.Rows {
		if _, ok := grouped.Group[r.PID]; !ok {
			grouped.Group[r.PID] = &Agg{
				comms: make(map[string]struct{}),
				dptrs: make(map[string]struct{}),
				tids:  make(map[string]struct{}),
				sids:  make(map[string]struct{}),
			}
		}
		g := grouped.Group[r.PID]

		g.total += r.Total
		g.comms[fmt.Sprintf("%s:%d", r.Comm, r.Tid)] = struct{}{}
		g.dptrs[r.Dptr] = struct{}{}
		g.tids[fmt.Sprintf("%d", r.Tid)] = struct{}{}
		g.sids[fmt.Sprintf("%d", r.Sid)] = struct{}{}
	}
	return grouped

}
