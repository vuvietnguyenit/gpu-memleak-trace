#!/usr/bin/env python3
# Python program generate multiple processes those use GPU resources and make allocate/free memory (can simulator multiple train machine learning jobs)
# Example: 4 processes over GPUs 0 and 1 for 60s, 32–512 MiB blocks,
# ~10% leak rate, tiny compute each iter, 25ms pacing.
# NOTICE: Need very careful when use this script, it can make interrupt another processes
# python gpu_mem_stressor.py \
#   --procs 4 --gpus 0,1 --duration 60 \
#   --min-mb 32 --max-mb 512 \
#   --sleep-ms 25 --leak-prob 0.1 --compute
import argparse
import os
import random
import signal
import sys
import time
from multiprocessing import Process, Event, current_process
from multiprocessing import get_context


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-process GPU memory stressor (PyTorch) to simulate ML jobs.")
    p.add_argument("--procs", type=int, default=2,
                   help="Number of worker processes to spawn.")
    p.add_argument("--gpus", type=str, default="",
                   help="Comma-separated GPU IDs to use (e.g., '0,1'). If empty, use all visible.")
    p.add_argument("--duration", type=int, default=60,
                   help="How long to run (seconds).")
    p.add_argument("--min-mb", type=int, default=16,
                   help="Minimum allocation size per block (MiB).")
    p.add_argument("--max-mb", type=int, default=256,
                   help="Maximum allocation size per block (MiB).")
    p.add_argument("--sleep-ms", type=int, default=25,
                   help="Sleep between iterations per worker (milliseconds).")
    p.add_argument("--leak-prob", type=float, default=0.0,
                   help="Probability [0..1] of *not* freeing an allocation this iter (simulated leak).")
    p.add_argument("--compute", action="store_true",
                   help="Do a tiny matmul on the device per iter to simulate compute work.")
    p.add_argument("--report-interval", type=float, default=5.0,
                   help="Seconds between memory usage reports per worker.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base RNG seed. Each worker derives from this.")
    return p.parse_args()


def has_cuda_torch():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def bytes_from_mib(mib: int) -> int:
    return mib * 1024 * 1024


def worker(stop_ev: Event, gpu_id: int, args, worker_idx: int):
    name = current_process().name
    try:
        import torch
    except Exception as e:
        print(f"[{name}] PyTorch import failed: {e}", flush=True)
        return

    if not torch.cuda.is_available():
        print(f"[{name}] CUDA not available, exiting.", flush=True)
        return

    # Bind to a specific device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    rng = random.Random(args.seed + worker_idx * 131)

    # Storage for intentionally leaked allocations
    leaked = []

    start = time.time()
    last_report = start

    # Pre-warm CUDA context
    torch.cuda.synchronize(device)
    _ = torch.tensor([1.0], device=device)

    print(f"[{name}] Started on GPU {gpu_id}. Duration={args.duration}s, "
          f"alloc_range=[{args.min_mb},{args.max_mb}] MiB, leak_prob={args.leak_prob}", flush=True)

    iters = 0
    try:
        while not stop_ev.is_set():
            now = time.time()
            if now - start >= args.duration:
                break

            # Pick a random size
            sz_mib = rng.randint(args.min_mb, max(args.min_mb, args.max_mb))
            nbytes = bytes_from_mib(sz_mib)

            # Allocate a flat tensor of uint8 to control bytes exactly
            # (uint8 uses 1 byte per element)
            try:
                t = torch.empty(nbytes, dtype=torch.uint8, device=device)
            except RuntimeError as e:
                # OOM or allocator error; report and back off a bit
                print(
                    f"[{name}] Allocation of {sz_mib} MiB failed: {e}", flush=True)
                time.sleep(args.sleep_ms / 1000.0)
                continue

            # Optionally do a compute op using a small matmul
            if args.compute:
                # Create small matrices to keep runtime light but non-zero
                a = torch.randn((512, 256), device=device, dtype=torch.float32)
                b = torch.randn((256, 512), device=device, dtype=torch.float32)
                c = a @ b  # GEMM
                # Prevent dead code elimination
                _ = c.sum()

            # Decide to leak or free
            if rng.random() < args.leak_prob:
                leaked.append(t)  # keep reference -> no free
            else:
                # Explicitly drop the reference so it can be freed
                del t

            # Periodic reporting
            if (now - last_report) >= args.report_interval:
                try:
                    alloc = torch.cuda.memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)
                except Exception:
                    alloc = reserved = -1
                print(f"[{name}] GPU{gpu_id} iters={iters} "
                      f"allocated={alloc/1024/1024:.1f} MiB reserved={reserved/1024/1024:.1f} MiB "
                      f"leaked_blocks={len(leaked)}", flush=True)
                last_report = now

            # Small sleep to modulate pressure
            time.sleep(args.sleep_ms / 1000.0)
            iters += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Optional: free leaked tensors at shutdown to avoid persistent pressure
        # Comment out the following two lines to keep memory "leaked" after exit (for testing).
        # leaked.clear()
        # torch.cuda.empty_cache()
        try:
            alloc = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        except Exception:
            alloc = reserved = -1
        print(f"[{name}] Exiting. Final allocated={alloc/1024/1024:.1f} MiB "
              f"reserved={reserved/1024/1024:.1f} MiB leaked_blocks={len(leaked)}", flush=True)


def main():
    args = parse_args()

    # Basic validation
    if args.procs <= 0:
        print("--procs must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.min_mb <= 0 or args.max_mb <= 0 or args.min_mb > args.max_mb:
        print("--min-mb and --max-mb must be positive and min <= max", file=sys.stderr)
        sys.exit(2)
    if args.leak_prob < 0.0 or args.leak_prob > 1.0:
        print("--leak-prob must be in [0,1]", file=sys.stderr)
        sys.exit(2)

    if not has_cuda_torch():
        print("CUDA not available via PyTorch on this host. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Select GPUs
    try:
        import torch
        num_total = torch.cuda.device_count()
    except Exception as e:
        print(f"Failed to query CUDA devices: {e}", file=sys.stderr)
        sys.exit(1)

    if num_total == 0:
        print("No CUDA devices found.", file=sys.stderr)
        sys.exit(1)

    if args.gpus.strip():
        gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpu_list = list(range(num_total))

    # Re-map to valid indices
    gpu_list = [g for g in gpu_list if 0 <= g < num_total]
    if not gpu_list:
        print("No valid GPU IDs selected.", file=sys.stderr)
        sys.exit(2)

    # Handle SIGINT/SIGTERM for graceful stop
    stop_ev = Event()

    def _handle_signal(signum, frame):
        stop_ev.set()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ctx = get_context("spawn")   # <<< use spawn instead of fork
    procs = []
    for i in range(args.procs):
        gpu_id = gpu_list[i % len(gpu_list)]
        p = ctx.Process(target=worker,
                        args=(stop_ev, gpu_id, args, i),
                        name=f"gpu-worker-{i}")
        p.daemon = False
        p.start()
        procs.append(p)

    # Wait for workers or duration timeout (workers also self-stop by time)
    start = time.time()
    try:
        while time.time() - start < args.duration and any(p.is_alive() for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_ev.set()

    # Ensure stop signaled
    stop_ev.set()
    for p in procs:
        p.join()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
