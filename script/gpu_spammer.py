#!/usr/bin/env python3
import time
import argparse
import torch


def gpu_malloc_free(events_per_sec=1000, alloc_size=1_000_000, duration=5, device=0, hold_time=0.0):
    """
    GPU malloc/free stress test with PyTorch.

    :param events_per_sec: malloc/free events per second
    :param alloc_size: allocation size in bytes
    :param duration: run duration in seconds
    :param device: GPU device id
    :param hold_time: seconds to hold allocation before freeing
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Run on a machine with GPU + PyTorch + CUDA.")

    dev = f"cuda:{device}"
    interval = 1.0 / events_per_sec
    start = time.perf_counter()
    next_time = start
    count = 0

    print(f"[+] Running {duration}s on {dev}, "
          f"{events_per_sec} events/sec, {alloc_size} bytes per alloc, hold={hold_time}s")

    references = []  # keep references if hold_time > 0

    while True:
        now = time.perf_counter()
        if now - start >= duration:
            break

        # wait until it's time for the next event
        if now < next_time:
            time.sleep(next_time - now)

        try:
            # allocate
            t = torch.empty(alloc_size, dtype=torch.uint8, device=dev)
            # touch memory
            t[0] = 1
            if alloc_size > 1:
                t[-1] = 1

            if hold_time > 0:
                references.append((t, time.perf_counter() + hold_time))
            else:
                del t
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"[!] OOM: {e}")
            torch.cuda.empty_cache()
            time.sleep(0.01)

        # free expired allocations
        if hold_time > 0:
            now_ts = time.perf_counter()
            still_alive = []
            for tensor, expire in references:
                if now_ts >= expire:
                    del tensor
                else:
                    still_alive.append((tensor, expire))
            references = still_alive
            torch.cuda.empty_cache()

        count += 1
        next_time += interval

    print(f"[+] Finished {count} malloc/free events")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU malloc/free stress test (PyTorch)")
    parser.add_argument("--events-per-sec", type=float, default=1000,
                        help="malloc/free events per second (default=1000)")
    parser.add_argument("--alloc-size", type=int, default=1_000_000,
                        help="allocation size in bytes (default=1MB)")
    parser.add_argument("--duration", type=float, default=5,
                        help="run duration in seconds (default=5)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device id (default=0)")
    parser.add_argument("--hold-time", type=float, default=0.0,
                        help="seconds to hold allocation before freeing (default=0)")

    args = parser.parse_args()
    gpu_malloc_free(events_per_sec=args.events_per_sec,
                    alloc_size=args.alloc_size,
                    duration=args.duration,
                    device=args.device,
                    hold_time=args.hold_time)
