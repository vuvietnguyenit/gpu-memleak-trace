#!/usr/bin/env python3
import argparse
import math
import os
import sys
import threading
import time
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print("This script requires PyTorch. Please install it with `pip install torch`.", file=sys.stderr)
    raise


def mb_to_numel(mb: float, dtype: torch.dtype = torch.float32) -> int:
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    return max(1, int((mb * 1024 * 1024) // bytes_per_elem))


class TinyModel(nn.Module):
    """A tiny model to exercise GPU a bit beyond raw allocation."""

    def __init__(self, hidden: int = 1024):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.seq(x)


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def log_cuda_mem(prefix: str, device: torch.device):
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        print(f"[{prefix}] allocated={format_bytes(alloc)}, reserved={format_bytes(reserved)}, peak={format_bytes(peak)}", flush=True)


def worker(
    tid: int,
    barrier: threading.Barrier,
    args: argparse.Namespace,
    stop_flag: threading.Event,
    hold_refs_lists: List[List[torch.Tensor]],
    device: torch.device,
    model: Optional[nn.Module],
):
    local_refs = hold_refs_lists[tid]
    if model is not None:
        model = model.to(device)

    for it in range(args.iterations):
        if stop_flag.is_set():
            break

        # Align threads at batch boundary
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            break

        size_mb = args.normal_mb
        is_anomaly = (tid == args.anomaly_thread and it == args.anomaly_iter)
        if is_anomaly:
            size_mb = args.anomaly_mb

        numel = mb_to_numel(size_mb, dtype=torch.float32)

        try:
            # Allocate a large tensor on GPU
            x = torch.empty(numel, dtype=torch.float32, device=device)
            # Optionally do a small compute step to make it more realistic
            if model is not None:
                # Reshape to [batch, hidden] if possible; else just do a simple op
                hidden = args.model_hidden
                batch = max(1, numel // hidden)
                x2 = x[: batch * hidden].view(batch, hidden)
                y = model(x2)
                # keep a tiny reference to prevent complete optimization-away
                if args.keep_intermediate:
                    local_refs.append(y)
            else:
                # simple op to touch memory
                x.mul_(1.0001).add_(0.0001)

            # Keep references to simulate leaks/retention
            if args.hold_refs > 0:
                local_refs.append(x)
                # Bound the number of kept references to avoid unbounded growth unless requested
                while len(local_refs) > args.hold_refs:
                    ref = local_refs.pop(0)
                    # Drop reference explicitly
                    del ref

            # Report after allocation
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                log_cuda_mem(
                    f"T{tid} iter {it} ({'ANOMALY' if is_anomaly else 'normal'} {size_mb} MB)", device)
                # Reset peak if requested
                if args.reset_peak_each_iter:
                    torch.cuda.reset_peak_memory_stats(device)

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        except RuntimeError as e:
            # Catch CUDA OOM and keep going
            msg = str(e)
            print(f"[T{tid} iter {it}] RuntimeError: {msg}",
                  file=sys.stderr, flush=True)
            if "out of memory" in msg.lower():
                # optionally free some refs
                if local_refs:
                    drop_n = max(1, len(local_refs) // 2)
                    for _ in range(drop_n):
                        ref = local_refs.pop(0)
                        del ref
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # If configured to stop on OOM, set flag
                if args.stop_on_oom:
                    stop_flag.set()
            else:
                # unexpected runtime error; propagate
                stop_flag.set()

        except KeyboardInterrupt:
            stop_flag.set()
            break

    # final barrier to let main know we're done (non-fatal if broken)
    try:
        barrier.wait(timeout=1.0)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(
        description="GPU allocation anomaly simulator with PyTorch")
    p.add_argument("--device", default="cuda:0",
                   help="Device to use, e.g., cuda:0 or cpu")
    p.add_argument("--threads", type=int, default=4,
                   help="Number of worker threads (simulated concurrent streams)")
    p.add_argument("--iterations", type=int, default=20,
                   help="Iterations (batches) per thread")
    p.add_argument("--normal-mb", type=float, default=128,
                   help="Normal per-iteration allocation size in MB")
    p.add_argument("--anomaly-mb", type=float, default=4096,
                   help="Anomalous allocation size in MB for one thread/iter")
    p.add_argument("--anomaly-thread", type=int, default=0,
                   help="Thread index that will perform the anomalous allocation")
    p.add_argument("--anomaly-iter", type=int, default=5,
                   help="Iteration at which anomaly happens (0-based)")
    p.add_argument("--hold-refs", type=int, default=3,
                   help="How many allocations to keep referenced per thread to simulate retention/leaks (0 disables)")
    p.add_argument("--sleep-ms", type=int, default=50,
                   help="Sleep between iterations to control pacing")
    p.add_argument("--use-model", action="store_true",
                   help="Run a tiny NN on the allocated tensor to add compute load")
    p.add_argument("--model-hidden", type=int, default=1024,
                   help="Hidden size for the tiny model (only if --use-model)")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Keep model outputs referenced (increases retention)")
    p.add_argument("--reset-peak-each-iter", action="store_true",
                   help="Reset CUDA peak memory stats after each iteration")
    p.add_argument("--stop-on-oom", action="store_true",
                   help="Stop all threads when an OOM occurs")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                "CUDA is not available. Use --device cpu or install CUDA-enabled PyTorch.", file=sys.stderr)
            sys.exit(1)
        # Warm up context
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")

    model = TinyModel(args.model_hidden) if args.use_model else None

    # +1 for main thread to synchronize starts
    barrier = threading.Barrier(args.threads + 1)
    stop_flag = threading.Event()
    hold_refs_lists: List[List[torch.Tensor]] = [[]
                                                 for _ in range(args.threads)]

    threads = []
    for tid in range(args.threads):
        t = threading.Thread(
            target=worker,
            args=(tid, barrier, args, stop_flag,
                  hold_refs_lists, device, model),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Run iterations with barrier to mark batch boundaries
    try:
        for it in range(args.iterations):
            # Let workers start this batch
            barrier.wait()
            if stop_flag.is_set():
                break
            # Optionally, main thread could do something here (e.g., log overall stats)
            if device.type == "cuda":
                # Give workers a bit of time to allocate this batch
                time.sleep(max(0.0, args.sleep_ms / 1000.0) * 0.5)
                log_cuda_mem(f"MAIN after starting batch {it}", device)

        # Final sync to let threads exit cleanly
        try:
            barrier.wait(timeout=2.0)
        except Exception:
            pass

    except KeyboardInterrupt:
        stop_flag.set()

    for t in threads:
        t.join(timeout=5.0)

    # Final memory stats
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print("Final CUDA memory stats:")
        log_cuda_mem("FINAL", device)


if __name__ == "__main__":
    main()
