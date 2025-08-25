#!/usr/bin/env python
"""
gpu_mem_threads_leak.py
Multithreaded PyTorch script that randomly mallocs/frees GPU memory to generate
allocation/free patterns for tracers or stress testing.

Notes:
- Uses THREADS (not processes) to avoid CUDA fork issues.
- Each thread randomly allocates/deletes tensors on the selected CUDA device.
- Optional memory ceiling to avoid OOM.
- Periodic stats printout: allocated, reserved, active tensors.
"""

import argparse
import random
import threading
import time
import signal
import sys
from typing import List, Optional

import torch

def human_mb(x_bytes: int) -> float:
    return x_bytes / (1024.0 * 1024.0)

def make_tensor(byte_size: int, device: torch.device) -> torch.Tensor:
    # Allocate exact number of bytes using uint8
    n = max(1, byte_size)
    return torch.empty(n, dtype=torch.uint8, device=device, requires_grad=False)

class Worker(threading.Thread):
    def __init__(
        self,
        tid: int,
        device: torch.device,
        max_tensor_mb: int,
        alloc_prob: float,
        stop_event: threading.Event,
        stats: dict,
        memory_cap_mb: Optional[int] = None,
        sync_every: int = 25,
        empty_cache_every: int = 200,
        min_tensor_kb: int = 64,
    ):
        super().__init__(daemon=True)
        self.tid = tid
        self.device = device
        self.max_tensor_mb = max_tensor_mb
        self.alloc_prob = alloc_prob
        self.stop_event = stop_event
        self.memory_cap_mb = memory_cap_mb
        self.sync_every = sync_every
        self.empty_cache_every = empty_cache_every
        self.min_tensor_kb = min_tensor_kb
        self.tensors: List[torch.Tensor] = []
        self.iter = 0
        self.stats = stats  # shared dict: {'allocs': int, 'frees': int}

    def can_alloc(self, bytes_needed: int) -> bool:
        if self.memory_cap_mb is None:
            return True
        allocated = torch.cuda.memory_allocated(self.device)
        will_be = allocated + bytes_needed
        return human_mb(will_be) <= self.memory_cap_mb

    def run(self):
        gen = random.Random(self.tid ^ int(time.time()))
        while not self.stop_event.is_set():
            self.iter += 1
            # Decide alloc or free
            do_alloc = gen.random() < self.alloc_prob or len(self.tensors) == 0
            if do_alloc:
                # Random tensor size between min_tensor_kb and max_tensor_mb
                size_kb = gen.randint(self.min_tensor_kb, self.max_tensor_mb * 1024)
                byte_size = size_kb * 1024
                if self.can_alloc(byte_size):
                    try:
                        t = make_tensor(byte_size, self.device)
                        self.tensors.append(t)
                        self.stats['allocs'] += 1
                    except RuntimeError as e:
                        # Allocation failed, likely OOM; back off
                        time.sleep(0.01)
                else:
                    # Can't allocate under cap, free instead if possible
                    if self.tensors:
                        idx = gen.randrange(len(self.tensors))
                        self.tensors[idx] = self.tensors[-1]
                        self.tensors.pop()
                        self.stats['frees'] += 1
            else:
                if self.tensors:
                    idx = gen.randrange(len(self.tensors))
                    self.tensors[idx] = self.tensors[-1]
                    self.tensors.pop()
                    self.stats['frees'] += 1

            # Periodic sync to flush work
            if self.iter % self.sync_every == 0:
                torch.cuda.synchronize(self.device)

            # Periodically release cached blocks so external tracers can see churn
            if self.empty_cache_every and self.iter % self.empty_cache_every == 0:
                torch.cuda.empty_cache()

            # Small jitter to vary timing
            time.sleep(gen.uniform(1, 5))

def print_stats_loop(device: torch.device, workers: List[Worker], stop_event: threading.Event, interval: float):
    last = time.time()
    while not stop_event.is_set():
        time.sleep(interval)
        now = time.time()
        try:
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            free, total = torch.cuda.mem_get_info(device)
        except Exception:
            allocated = reserved = 0
            free = total = 0

        active_tensors = sum(len(w.tensors) for w in workers)
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"allocated={human_mb(allocated):.2f}MB "
              f"reserved={human_mb(reserved):.2f}MB "
              f"free={human_mb(free):.2f}MB/{human_mb(total):.2f}MB "
              f"active_tensors={active_tensors} "
              f"allocs={sum(w.stats['allocs'] for w in workers)} "
              f"frees={sum(w.stats['frees'] for w in workers)}")
        sys.stdout.flush()
        last = now

def main():
    parser = argparse.ArgumentParser(description="Multithreaded GPU malloc/free stress for PyTorch")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g., cuda:0")
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    parser.add_argument("--max-tensor-mb", type=int, default=64, help="Max tensor size per allocation (MB)")
    parser.add_argument("--alloc-prob", type=float, default=0.6, help="Probability of allocation vs free per step")
    parser.add_argument("--memory-cap-mb", type=int, default=None, help="Soft cap on allocated memory (MB)")
    parser.add_argument("--stats-interval", type=float, default=1.0, help="Stats print interval (s)")
    parser.add_argument("--seed", type=int, default=None, help="Global RNG seed for reproducibility")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Please install a CUDA-enabled PyTorch build and a CUDA-capable GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)

    # Force CUDA context creation in main thread
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    stop_event = threading.Event()

    def handle_sigint(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    workers: List[Worker] = []
    for i in range(args.threads):
        w = Worker(
            tid=i,
            device=device,
            max_tensor_mb=args.max_tensor_mb,
            alloc_prob=args.alloc_prob,
            stop_event=stop_event,
            stats={'allocs': 0, 'frees': 0},
            memory_cap_mb=args.memory_cap_mb,
        )
        workers.append(w)
        w.start()

    stats_thread = threading.Thread(target=print_stats_loop, args=(device, workers, stop_event, args.stats_interval), daemon=True)
    stats_thread.start()

    # Wait for duration then stop
    t0 = time.time()
    while time.time() - t0 < args.duration and not stop_event.is_set():
        time.sleep(0.1)

    stop_event.set()
    for w in workers:
        w.join()
    stats_thread.join(timeout=1.0)

    # Final stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print("Finished.")
    print(f"Final allocated={human_mb(allocated):.2f}MB reserved={human_mb(reserved):.2f}MB")

if __name__ == "__main__":
    main()
