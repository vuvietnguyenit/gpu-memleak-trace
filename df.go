package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
	"github.com/olekukonko/tablewriter/renderer"
	"github.com/olekukonko/tablewriter/tw"
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

func (g *Grouped) PrintTable() {
	table := tablewriter.NewTable(os.Stdout, tablewriter.WithRenderer(renderer.NewBlueprint(tw.Rendition{
		Settings: tw.Settings{Separators: tw.Separators{BetweenRows: tw.On}},
	})))
	table.Header([]string{"PID", "COMM", "SID", "TOTAL"})

	for pid, agg := range g.Group {
		// turn sets into joined strings with newlines
		comms := strings.Join(setToSlice(agg.comms), "\n")
		dptrs := strings.Join(setToSlice(agg.dptrs), "\n")
		sids := strings.Join(setToSlice(agg.sids), "\n")

		row := []string{
			strconv.Itoa(int(pid)),
			comms,
			dptrs,
			sids,
			agg.total.HumanSize(),
		}
		table.Append(row)
	}
	fmt.Print("\033[H\033[2J")
	table.Render()
}

func setToSlice(m map[string]struct{}) []string {
	out := []string{}
	for k := range m {
		out = append(out, k)
	}
	return out
}

func (gr Grouped) Print() {
	// Print results
	for pid, g := range gr.Group {
		fmt.Printf("PID: %d\n", pid)

		fmt.Println("COMM:TID:")
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

		fmt.Printf("TOTAL: %s\n", g.total.HumanSize())
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
