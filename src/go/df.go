package main

import (
	"fmt"
	"slices"
	"sort"
	"sync"
)

type Row struct {
	Timestamp Timestamp
	DeviceID  DeivceID
	Pid       Pid
	Uid       Uid
	Comm      Comm
	Dptr      Dptr
	Tid       Tid
	Sid       StackID
	Size      AllocSize
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

type PtrAllocateInfo struct {
	Dptr      Dptr
	DeviceID  DeivceID
	Size      AllocSize
	Timestamp Timestamp
}

type CommGroup struct {
	Comm Comm
	Tid  Tid
	Ptrs []PtrAllocateInfo
}

type Result struct {
	Pid   Pid
	Uid   Uid
	Comms map[Tid]CommGroup
	Total AllocSize
}

type AllocValue struct {
	Size      AllocSize
	Timestamp Timestamp
}

func (df *DF) GroupAlloc() map[Pid]Result {
	allocs := make(map[AllocKey]AllocValue)
	names := make(map[AllocKey]Comm)

	for _, e := range df.Rows {
		k := AllocKey{Pid: e.Pid, Uid: e.Uid, Tid: e.Tid, DeviceID: e.DeviceID, Dptr: e.Dptr}
		allocs[k] = AllocValue{Size: allocs[k].Size + e.Size, Timestamp: e.Timestamp}
		names[k] = e.Comm
	}

	results := make(map[Pid]Result)

	for k, v := range allocs {
		r := results[k.Pid]
		if r.Pid == 0 {
			r = Result{Pid: k.Pid, Comms: make(map[Tid]CommGroup), Uid: k.Uid}
		}
		cg := r.Comms[k.Tid]
		if cg.Tid == 0 {
			cg = CommGroup{Comm: names[k], Tid: k.Tid}
		}
		cg.Ptrs = append(cg.Ptrs, PtrAllocateInfo{
			Dptr:      k.Dptr,
			DeviceID:  k.DeviceID,
			Size:      v.Size,
			Timestamp: v.Timestamp,
		})
		// You can add Timestamp to PtrInfo if needed
		r.Comms[k.Tid] = cg
		r.Total += v.Size
		results[k.Pid] = r
	}
	return results
}

var mux sync.Mutex

func PrintResults(results map[Pid]Result) {
	mux.Lock()
	defer mux.Unlock()
	for _, r := range results {
		fmt.Printf("PID: %d / UID: %d\n", r.Pid, r.Uid)

		// sort by TID for stable output
		tids := make([]Tid, 0, len(r.Comms))
		for tid := range r.Comms {
			tids = append(tids, tid)
		}
		slices.Sort(tids)
		for _, tid := range tids {
			cg := r.Comms[tid]
			fmt.Printf("  %s:%d\n", cg.Comm, cg.Tid)
			// sort pointers by timestamp
			sort.Slice(cg.Ptrs, func(i, j int) bool { return Timestamp(cg.Ptrs[i].Timestamp) > cg.Ptrs[j].Timestamp })
			for _, p := range cg.Ptrs {
				fmt.Printf("   [%s]  %-14s  %-6s  %8s\n",
					KtimeToTime(p.Timestamp).Format("2006-01-02T15:04:05.000000000Z07:00"),
					fmt.Sprintf("0x%012x", p.Dptr),
					fmt.Sprintf("gpu:%v", p.DeviceID),
					fmt.Sprintf("allocated size:%s", p.Size.HumanSize()),
				)
			}
		}
		fmt.Printf("\nTOTAL LEAKED: %s\n\n", r.Total.HumanSize())
	}
}
