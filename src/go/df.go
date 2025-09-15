package main

import (
	"fmt"
	"sort"
	"time"
)

type ThreadGroup struct {
	Tid     Tid
	Name    Comm
	Total   AllocSize
	Count   uint64
	SizeMap map[AllocSize]uint64 // Size -> number of blocks
}

type Result struct {
	Pid     Pid
	Uid     Uid
	Comm    Comm
	Threads map[Tid]*ThreadGroup
	Total   AllocSize
}

func (t *AllocTable) Aggregate() map[Pid]*Result {
	t.mu.RLock()
	defer t.mu.RUnlock()

	results := make(map[Pid]*Result)

	for key, entry := range t.data {
		p, ok := results[key.Pid]
		if !ok {
			p = &Result{
				Pid:     key.Pid,
				Uid:     key.Uid,
				Comm:    key.Comm,
				Total:   entry.TotalSize, // Total size for this PID
				Threads: make(map[Tid]*ThreadGroup),
			}
			results[key.Pid] = p
		}

		tg, ok := p.Threads[key.Tid]
		if !ok {
			tg = &ThreadGroup{
				Tid:     key.Tid,
				Name:    key.Comm,
				Total:   entry.TotalSize, // Total size for this TID
				Count:   1,
				SizeMap: make(map[AllocSize]uint64),
			}
			p.Threads[key.Tid] = tg
		}

		tg.Count++
		tg.SizeMap[entry.Size]++
		p.Total += entry.Size
	}

	return results
}

func PrintResults(results map[Pid]*Result, topThreadN, topSizeN int) {
	now := time.Now()
	fmt.Printf("-------------------- %s --------------------\n", now.Format(time.RFC3339))

	for _, r := range results {
		fmt.Printf("PID: %d (%s) / UID: %d / GPU:0\n\n", r.Pid, r.Comm, r.Uid)
		fmt.Printf("TOTAL LEAKED: %s (%d allocations)\n", r.Total.HumanSize(), totalAllocCount(r))
		fmt.Printf("Leak Rate: %s bytes/sec\n\n", r.Total.HumanSize()) // placeholder

		// Sort threads by total descending
		threads := make([]*ThreadGroup, 0, len(r.Threads))
		for _, tg := range r.Threads {
			threads = append(threads, tg)
		}
		sort.Slice(threads, func(i, j int) bool {
			return threads[i].Total > threads[j].Total
		})

		fmt.Printf("Top %d Leaked Groups (by TID, sorted by total leaked size):\n\n", topThreadN)
		for i, tg := range threads {
			if i >= topThreadN {
				fmt.Printf("  [... + %d more for this process]\n", len(threads)-topThreadN)
				break
			}

			fmt.Printf("  TID %d (%s):\n", tg.Tid, tg.Name)
			fmt.Printf("    Total Leaked: %s, %d blocks\n", r.Total.HumanSize(), tg.Count)
			fmt.Printf("    Groups:\n")

			// Sort size groups descending
			type kv struct {
				Size  AllocSize
				Count uint64
			}
			sizes := make([]kv, 0, len(tg.SizeMap))
			for sz, count := range tg.SizeMap {
				sizes = append(sizes, kv{Size: sz, Count: count})
			}
			sort.Slice(sizes, func(i, j int) bool { return sizes[i].Size > sizes[j].Size })

			for j, s := range sizes {
				if j >= topSizeN {
					fmt.Printf("      [... + %d more for this thread]\n", len(sizes)-topSizeN)
					break
				}
				fmt.Printf("      Size=%d bytes,   Blocks=%d,  Total=%s\n",
					s.Size, s.Count, (s.Size * AllocSize(s.Count)).HumanSize())
			}
			fmt.Println()
		}

		// // Compute largest block for PID
		// var largestSize AllocSize
		// var largestCount uint64
		// for _, tg := range threads {
		// 	for sz, count := range tg.SizeMap {
		// 		if sz > largestSize {
		// 			largestSize = sz
		// 			largestCount = count
		// 		}
		// 	}
		// }
		// totalLeaked := largestSize * AllocSize(largestCount)
		// percent := float64(totalLeaked) / float64(r.Total) * 100.0

		// fmt.Printf("Largest Block Size Summary (PID level):\n")
		// fmt.Printf("  size_bytes      = %d   // %s per block\n", largestSize, largestSize.HumanSize())
		// fmt.Printf("  blocks_count    = %d   // %d allocations of this size\n", largestCount, largestCount)
		// fmt.Printf("  total_leaked    = %d   // %s in total\n", totalLeaked, totalLeaked.HumanSize())
		// fmt.Printf("  percent_of_leak = %.1f%%   // contributes ~%.1f%% of all leaks\n", percent, percent)
		// fmt.Printf("  freed_ratio     = %.1f   // none of these were freed\n", 0.0)
		// fmt.Printf("  note            = \"%s\"\n\n", "This is the largest leaked allocation group observed")
	}
}

// helper to get total number of allocations per PID
func totalAllocCount(r *Result) int {
	var cnt int
	for _, tg := range r.Threads {
		cnt += int(tg.Count)
	}
	return cnt
}

// type Row struct {
// 	Timestamp Timestamp
// 	DeviceID  DeviceID
// 	Pid       Pid
// 	Uid       Uid
// 	Comm      Comm
// 	Tid       Tid
// 	Size      AllocSize
// }

// type Header string

// type DF struct {
// 	Headers []Header
// 	Rows    []Row
// }

// func (df *DF) InitHeader(headers []Header) {
// 	df.Headers = headers
// }

// func (df *DF) Insert(r Row) {
// 	df.Rows = append(df.Rows, r)
// }

// type PtrAllocateInfo struct {
// 	Dptr      Dptr
// 	DeviceID  DeviceID
// 	Size      AllocSize
// 	Timestamp Timestamp
// }

// type ThreadGroup struct {
// 	Comm Comm
// 	Tid  Tid
// 	Ptrs []PtrAllocateInfo
// }

// type Result struct {
// 	Pid     Pid
// 	Uid     Uid
// 	Comm    Comm
// 	Threads map[Tid]ThreadGroup
// 	Total   AllocSize
// }

// type AllocValue struct {
// 	Size      AllocSize
// 	Timestamp Timestamp
// }

// func (df *DF) GroupAlloc() map[Pid]Result {
// 	allocs := make(map[AllocKey]AllocValue)
// 	names := make(map[AllocKey]Comm)

// 	for _, e := range df.Rows {
// 		k := AllocKey{Pid: e.Pid, Uid: e.Uid, Tid: e.Tid, DeviceID: e.DeviceID}
// 		allocs[k] = AllocValue{Size: allocs[k].Size + e.Size, Timestamp: e.Timestamp}
// 		names[k] = e.Comm
// 	}

// 	results := make(map[Pid]Result)

// 	for k, v := range allocs {
// 		r := results[k.Pid]
// 		if r.Pid == 0 {
// 			r = Result{Pid: k.Pid, Threads: make(map[Tid]ThreadGroup), Uid: k.Uid}
// 		}
// 		cg := r.Threads[k.Tid]
// 		if cg.Tid == 0 {
// 			cg = ThreadGroup{Comm: names[k], Tid: k.Tid}
// 		}
// 		cg.Ptrs = append(cg.Ptrs, PtrAllocateInfo{
// 			DeviceID:  k.DeviceID,
// 			Size:      v.Size,
// 			Timestamp: v.Timestamp,
// 		})
// 		// You can add Timestamp to PtrInfo if needed
// 		r.Threads[k.Tid] = cg
// 		r.Total += v.Size
// 		results[k.Pid] = r
// 	}
// 	return results
// }

// var mux sync.Mutex

// func PrintResult(results map[Pid]Result, topN int) {
// 	now := time.Now()
// 	fmt.Printf("-------------------- %s --------------------\n", now.Format(time.RFC3339))

// 	for _, r := range results {
// 		fmt.Printf("PID: %d (%s) / UID: %d / GPU:0\n\n", r.Pid, r.Comm, r.Uid)

// 		fmt.Printf("TOTAL LEAKED: %s (%d allocations)\n", r.Total.HumanSize(), totalAllocCount(r))
// 		fmt.Printf("Leak Rate: %s bytes/sec\n\n", humanizeBytes(r.Total/1)) // placeholder for rate

// 		// Top N Threads by total leaked size
// 		fmt.Printf("Top %d Leaked Groups (by TID, sorted by total leaked size):\n\n", topN)

// 		threads := make([]ThreadGroup, 0, len(r.Threads))
// 		for _, tg := range r.Threads {
// 			threads = append(threads, tg)
// 		}

// 		// Sort by total descending
// 		sort.Slice(threads, func(i, j int) bool {
// 			return threads[i].Total > threads[j].Total
// 		})

// 		for i, tg := range threads {
// 			if i >= topN {
// 				fmt.Printf("  [... + %d more for this process]\n", len(threads)-topN)
// 				break
// 			}
// 			fmt.Printf("  TID %d (%s):\n", tg.Tid, tg.Name)
// 			fmt.Printf("    Total Leaked: %s, %d blocks\n", humanizeBytes(tg.Total), tg.Count)
// 			fmt.Printf("    Groups:\n")

// 			// If you have per-size groups, sort and print them here
// 			sizeGroups := getSizeGroups(tg) // implement this to group by block size
// 			printSizeGroups(sizeGroups, 5)  // top 5 block sizes
// 		}

// 		// Largest Block Size Summary (PID level)
// 		largest := getLargestBlock(r) // implement logic to find largest allocation block
// 		fmt.Printf("\nLargest Block Size Summary (PID level):\n")
// 		fmt.Printf("  size_bytes      = %d   // %s per block\n", largest.Size, humanizeBytes(largest.Size))
// 		fmt.Printf("  blocks_count    = %d   // %d allocations of this size\n", largest.Count, largest.Count)
// 		fmt.Printf("  total_leaked    = %d   // %s in total\n", largest.Total, humanizeBytes(largest.Total))
// 		fmt.Printf("  percent_of_leak = %.1f%%   // contributes ~%.1f%% of all leaks\n", largest.Percent, largest.Percent)
// 		fmt.Printf("  freed_ratio     = %.1f   // %.1f%% of these were freed\n", largest.FreedRatio, largest.FreedRatio*100)
// 		fmt.Printf("  note            = \"%s\"\n", largest.Note)

// 		fmt.Println()
// 	}
// }

// func PrintResults(results map[Pid]Result) {
// 	mux.Lock()
// 	defer mux.Unlock()
// 	for _, r := range results {
// 		fmt.Printf("PID: %d / UID: %d\n", r.Pid, r.Uid)

// 		// sort by TID for stable output
// 		tids := make([]Tid, 0, len(r.Threads))
// 		for tid := range r.Threads {
// 			tids = append(tids, tid)
// 		}
// 		slices.Sort(tids)
// 		for _, tid := range tids {
// 			cg := r.Threads[tid]
// 			fmt.Printf("  %s:%d\n", cg.Comm, cg.Tid)
// 			// sort pointers by timestamp
// 			sort.Slice(cg.Ptrs, func(i, j int) bool { return Timestamp(cg.Ptrs[i].Timestamp) > cg.Ptrs[j].Timestamp })
// 			// for _, p := range cg.Ptrs {
// 			// 	fmt.Printf("   [%s]  %-14s  %-6s  %8s\n",
// 			// 		KtimeToTime(p.Timestamp).Format("2006-01-02T15:04:05.000000000Z07:00"),
// 			// 		fmt.Sprintf("0x%012x", p.Dptr),
// 			// 		fmt.Sprintf("gpu:%v", p.DeviceID),
// 			// 		fmt.Sprintf("allocated size:%s", p.Size.HumanSize()),
// 			// 	)
// 			// }
// 		}
// 		fmt.Printf("\nTOTAL LEAKED: %s\n\n", r.Total.HumanSize())
// 	}
// }
