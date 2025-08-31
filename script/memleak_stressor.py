#!/usr/bin/env python3
"""
memleak_stressor.py — simple memory leak generator for testing eBPF-based detectors.

Supported modes:
  - python-heap : allocate Python bytearrays and keep references
  - libc        : call libc malloc() via ctypes and never free
  - torch       : allocate CUDA tensors with PyTorch and never free (requires CUDA + torch)
  - cupy        : allocate CUDA memory with CuPy and never free (requires CUDA + cupy)
  - pycuda      : allocate CUDA memory with PyCUDA and never free (requires CUDA + pycuda)

Examples:
  # 1) Leak on CPU heap, 4 MiB per alloc, every 100ms for ~10s
  python3 memleak_stressor.py --mode python-heap --size 4MiB --interval 0.1 --duration 10

  # 2) Leak native heap using malloc, 1 MiB per alloc until killed
  python3 memleak_stressor.py --mode libc --size 1MiB

  # 3) Leak on GPU 0 using PyCUDA (64 MiB per alloc)
  python3 memleak_stressor.py --mode pycuda --size 64MiB --device 0

  # 4) Leak GPU memory with PyTorch (256 MiB per alloc, stop after 4 GiB)
  python3 memleak_stressor.py --mode torch --size 256MiB --device 0 --max-bytes 4GiB
"""

import argparse
import ctypes
import os
import re
import signal
import sys
import time
from typing import Optional

# -------- Utilities --------

MULTIPLIERS = {
    "B": 1,
    "KB": 1000, "K": 1000,
    "MB": 1000**2, "M": 1000**2,
    "GB": 1000**3, "G": 1000**3,
    "TB": 1000**4, "T": 1000**4,
    "KIB": 1024, "KI": 1024,
    "MIB": 1024**2, "MI": 1024**2,
    "GIB": 1024**3, "GI": 1024**3,
    "TIB": 1024**4, "TI": 1024**4,
}

def parse_bytes(s: str) -> int:
    s = s.strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d+)([KMGT]?I?B?)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid size: {s}")
    n = int(m.group(1))
    unit = m.group(2) or "B"
    unit = {"": "B", "K": "KB", "M": "MB", "G": "GB", "T": "TB",
            "KI": "KIB", "MI": "MIB", "GI": "GIB", "TI": "TIB",
            "B": "B", "KB": "KB", "MB": "MB", "GB": "GB", "TB": "TB",
            "KIB": "KIB", "MIB": "MIB", "GIB": "GIB", "TIB": "TIB"}[unit]
    return n * MULTIPLIERS[unit]

def human_bytes(n: int) -> str:
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TiB"

# -------- Leak implementations --------

class Leaker:
    def __init__(self):
        self.total_bytes = 0
        self._keep = []  # keep references to avoid frees

    def alloc(self, size: int):
        raise NotImplementedError

    def keep_and_count(self, obj, size: int):
        self._keep.append(obj)
        self.total_bytes += size

class PythonHeapLeaker(Leaker):
    def alloc(self, size: int):
        # Allocate and keep a bytearray of 'size' bytes
        obj = bytearray(size)
        self.keep_and_count(obj, size)

class LibcLeaker(Leaker):
    def __init__(self):
        super().__init__()
        self.libc = ctypes.CDLL(None)
        self.libc.malloc.argtypes = [ctypes.c_size_t]
        self.libc.malloc.restype = ctypes.c_void_p

    def alloc(self, size: int):
        ptr = self.libc.malloc(size)
        if not ptr:
            raise MemoryError("malloc returned NULL")
        # Keep the integer address so nothing calls free()
        self.keep_and_count(int(ptr), size)

class TorchCudaLeaker(Leaker):
    def __init__(self, device: int = 0):
        super().__init__()
        import torch  # type: ignore
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for torch")
        torch.cuda.set_device(device)
        self.device = device

    def alloc(self, size: int):
        # Allocate a CUDA tensor of 'size' bytes
        numel = size  # use bytes as elements of uint8
        t = self.torch.empty((numel,), dtype=self.torch.uint8, device=f"cuda:{self.device}")
        self.keep_and_count(t, size)

class CuPyLeaker(Leaker):
    def __init__(self, device: int = 0):
        super().__init__()
        import cupy as cp  # type: ignore
        self.cp = cp
        self.device = device
        self.cp.cuda.Device(device).use()

    def alloc(self, size: int):
        mem = self.cp.cuda.alloc(size)
        self.keep_and_count(mem, size)

class PyCUDALeaker(Leaker):
    def __init__(self, device: int = 0):
        super().__init__()
        import pycuda.autoinit  # noqa: F401  # sets up context on device 0 by default
        import pycuda.driver as cuda  # type: ignore
        self.cuda = cuda
        # If desired device != 0, switch context
        if device != 0:
            # create context on requested device
            cuda.Context.pop()  # remove autoinit's context
            cuda.init()
            dev = cuda.Device(device)
            self.ctx = dev.make_context()
        else:
            # use autoinit's context
            self.ctx = None

    def alloc(self, size: int):
        buf = self.cuda.mem_alloc(size)
        self.keep_and_count(buf, size)

# -------- Main runner --------

def build_leaker(mode: str, device: int) -> Leaker:
    if mode == "python-heap":
        return PythonHeapLeaker()
    elif mode == "libc":
        return LibcLeaker()
    elif mode == "torch":
        return TorchCudaLeaker(device=device)
    elif mode == "cupy":
        return CuPyLeaker(device=device)
    elif mode == "pycuda":
        return PyCUDALeaker(device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main(argv=None):
    p = argparse.ArgumentParser(description="Allocate memory repeatedly and never free (for memleak testing).")
    p.add_argument("--mode", choices=["python-heap", "libc", "torch", "cupy", "pycuda"], default="python-heap",
                   help="Allocation backend.")
    p.add_argument("--size", type=parse_bytes, default=parse_bytes("4MiB"),
                   help="Bytes per allocation, e.g. 4MiB, 1GB.")
    p.add_argument("--interval", type=float, default=0.0,
                   help="Seconds to sleep between allocations (can be fractional).")
    p.add_argument("--count", type=int, default=None,
                   help="Number of allocations to perform (default: infinite).")
    p.add_argument("--duration", type=float, default=None,
                   help="Stop after N seconds (wall clock).")
    p.add_argument("--max-bytes", type=parse_bytes, default=None,
                   help="Stop after leaking this many bytes in total.")
    p.add_argument("--device", type=int, default=0, help="GPU device index for CUDA-backed modes.")
    p.add_argument("--report-every", type=int, default=10, help="Print progress every N allocations.")
    args = p.parse_args(argv)

    # Set up leaker
    try:
        leaker = build_leaker(args.mode, args.device)
    except Exception as e:
        print(f"Failed to initialize mode '{args.mode}': {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[memleak] PID={os.getpid()} mode={args.mode} size={human_bytes(args.size)} interval={args.interval}s")
    if args.duration:
        print(f"[memleak] Will stop after duration={args.duration}s")
    if args.count:
        print(f"[memleak] Will stop after count={args.count}")
    if args.max_bytes:
        print(f"[memleak] Will stop after max-bytes={human_bytes(args.max_bytes)}")

    start = time.time()
    i = 0

    def should_stop() -> bool:
        if args.count is not None and i >= args.count:
            return True
        if args.duration is not None and (time.time() - start) >= args.duration:
            return True
        if args.max_bytes is not None and leaker.total_bytes >= args.max_bytes:
            return True
        return False

    def handle_sigint(signum, frame):
        print(f"\n[memleak] Caught signal {signum}, exiting... total leaked ~ {human_bytes(leaker.total_bytes)}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_sigint)

    try:
        while True:
            leaker.alloc(args.size)
            i += 1
            if i % max(1, args.report_every) == 0:
                elapsed = time.time() - start
                rate = leaker.total_bytes / elapsed if elapsed > 0 else 0
                print(f"[memleak] iters={i} leaked={human_bytes(leaker.total_bytes)} elapsed={elapsed:.1f}s rate~{human_bytes(int(rate))}/s")
            if should_stop():
                break
            if args.interval > 0:
                time.sleep(args.interval)
    except MemoryError as e:
        print(f"[memleak] MemoryError after {i} allocs, leaked ~ {human_bytes(leaker.total_bytes)}: {e}")
        sys.exit(1)

    print(f"[memleak] Done. Total leaked ~ {human_bytes(leaker.total_bytes)} in {i} allocations.")
    # Keep process alive a bit so tools can observe the leaked state if duration/count ended.
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
