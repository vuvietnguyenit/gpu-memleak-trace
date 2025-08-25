package main

import (
	"fmt"
	"slices"
	"sort"
	"sync"
)

type Row struct {
	DeviceID DeivceID
	Pid      Pid
	Uid      Uid
	Comm     Comm
	Dptr     Dptr
	Tid      Tid
	Sid      StackID
	Size     AllocSize
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

type PtrInfo struct {
	Dptr     Dptr
	DeviceID DeivceID
	Size     AllocSize
}

type CommGroup struct {
	Comm Comm
	Tid  Tid
	Ptrs []PtrInfo
}

type Result struct {
	Pid   Pid
	Uid   Uid
	Comms map[Tid]CommGroup
	Total AllocSize
}

func (df *DF) GroupAlloc() map[Pid]Result {
	allocs := make(map[AllocKey]AllocSize)
	names := make(map[AllocKey]Comm)

	for _, e := range df.Rows {
		k := AllocKey{Pid: e.Pid, Uid: e.Uid, Tid: e.Tid, DeviceID: e.DeviceID, Dptr: e.Dptr}
		allocs[k] += e.Size
		names[k] = e.Comm
	}

	results := make(map[Pid]Result)

	for k, size := range allocs {
		r := results[k.Pid]
		if r.Pid == 0 {
			r = Result{Pid: k.Pid, Comms: make(map[Tid]CommGroup), Uid: k.Uid}
		}
		cg := r.Comms[k.Tid]
		if cg.Tid == 0 {
			cg = CommGroup{Comm: names[k], Tid: k.Tid}
		}
		cg.Ptrs = append(cg.Ptrs, PtrInfo{
			Dptr:     k.Dptr,
			DeviceID: k.DeviceID,
			Size:     size,
		})
		r.Comms[k.Tid] = cg
		r.Total += size
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
			// sort pointers by address
			sort.Slice(cg.Ptrs, func(i, j int) bool { return cg.Ptrs[i].Size > cg.Ptrs[j].Size })
			for _, p := range cg.Ptrs {
				fmt.Printf("    0x%x:gpu_%d: %s\n",
					p.Dptr, p.DeviceID, p.Size.HumanSize())
			}
		}
		fmt.Printf("\nTOTAL LEAKED: %s\n\n", r.Total.HumanSize())
	}
}
