package main

import (
	"fmt"
	"log/slog"
	"maps"
	"os"
	"os/user"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"
)

type AllocMap struct {
	mu   sync.Mutex
	data map[uint32]map[uint64]uint64 // PID -> (ptr -> size)
}

func NewAllocMap() *AllocMap {
	return &AllocMap{
		data: make(map[uint32]map[uint64]uint64),
	}
}

func (a *AllocMap) AddAlloc(pid uint32, ptr uint64, size uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if _, ok := a.data[pid]; !ok {
		a.data[pid] = make(map[uint64]uint64)
	}
	if oldSize, exists := a.data[pid][ptr]; exists {
		t := fmt.Sprintf("PID %d already has ptr %d (old size: %d, new size: %d)", pid, ptr, oldSize, size)
		slog.Warn(t)
	}

	a.data[pid][ptr] = size
}

func (a *AllocMap) FreeAlloc(pid uint32, ptr uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	allocs, ok := a.data[pid]
	if !ok {
		t := fmt.Sprintf("PID %d not found when freeing ptr %d", pid, ptr)
		slog.Warn(t)
		return
	}

	if _, exists := allocs[ptr]; !exists {
		t := fmt.Sprintf("PID %d has no record for ptr %d", pid, ptr)
		slog.Warn(t)
		return
	}

	delete(allocs, ptr)
}

func (a *AllocMap) String() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return fmt.Sprintf("%+v", a.data)
}

func (a *AllocMap) Snapshot() map[uint32]map[uint64]uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	copyData := make(map[uint32]map[uint64]uint64)
	for pid, allocs := range a.data {
		copyData[pid] = make(map[uint64]uint64)
		maps.Copy(copyData[pid], allocs)
	}
	return copyData
}

func (a *AllocMap) CleanupExited() {
	for {
		time.Sleep(1 * time.Second)
		slog.Debug("Checking for exited PIDs...", "data", a.String())
		a.mu.Lock()
		for pid := range a.data {
			if !pidExists(pid) || len(a.data[pid]) == 0 {
				delete(a.data, pid)
				t := fmt.Sprintf("PID %d exited — removing from allocs map", pid)
				slog.Debug(t)
			}
			if len(a.data[pid]) == 0 {
				delete(a.data, pid)
				t := fmt.Sprintf("PID %d has no allocations left — removing from allocs map", pid)
				slog.Debug(t)
			}
		}
		a.mu.Unlock()
	}
}

func pidExists(pid uint32) bool {
	_, err := os.Stat("/proc/" + strconv.FormatUint(uint64(pid), 10))
	return err == nil
}

func humanSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func getProcessInfo(pid uint32) (string, string, error) {
	// 1. Get COMM (process name) from /proc/<pid>/comm
	commPath := filepath.Join("/proc", strconv.Itoa(int(pid)), "comm")
	commBytes, err := os.ReadFile(commPath)
	if err != nil {
		return "", "", fmt.Errorf("read comm: %w", err)
	}
	comm := strings.TrimSpace(string(commBytes))

	// 2. Get UID from stat info
	procPath := filepath.Join("/proc", strconv.Itoa(int(pid)))
	stat, err := os.Stat(procPath)
	if err != nil {
		return "", "", fmt.Errorf("stat proc: %w", err)
	}

	sys := stat.Sys().(*syscall.Stat_t)
	uid := sys.Uid

	// 3. Convert UID to username
	u, err := user.LookupId(fmt.Sprintf("%d", uid))
	if err != nil {
		return "", comm, fmt.Errorf("lookup user: %w", err)
	}

	return u.Username, comm, nil
}

type PIDMem struct {
	PID   uint32
	Total uint64
}

func (a *AllocMap) aggregateResourcePID() []PIDMem {
	snapshot := a.Snapshot()
	slog.Debug("Current allocs_data", "data", snapshot)
	result := make(map[uint32]uint64)

	for pid, allocs := range snapshot {
		var totalPerPID uint64
		for _, size := range allocs {
			totalPerPID += size
		}
		result[pid] = totalPerPID
	}
	sorted := make([]PIDMem, 0, len(result))
	for pid, total := range result {
		sorted = append(sorted, PIDMem{PID: pid, Total: total})
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Total > sorted[j].Total
	})
	// Add cut-off if neccessary
	return sorted
}

func (a *AllocMap) printAllocMapPeriodically() {
	ticker := time.NewTicker(FlagPrintInterval)
	defer ticker.Stop()

	for range ticker.C {
		result := a.aggregateResourcePID()
		// Clear screen before printing (like htop)
		fmt.Print("\033[H\033[2J")
		fmt.Printf("Time: %s\n", time.Now().Format(time.RFC3339))
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "PID\tUSER\tCOMM\tLEAKED")

		for _, proc := range result {
			user, comm, err := getProcessInfo(proc.PID)
			if err != nil {
				user = "?"
				comm = "?"
			}
			fmt.Fprintf(w, "%d\t%s\t%s\t%s\n", proc.PID, user, comm, humanSize(int64(proc.Total)))
		}
		fmt.Fprintln(w, "----\t----\t----\t-----------")
		w.Flush()
	}

}
