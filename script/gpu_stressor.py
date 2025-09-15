#!/usr/bin/env python3
import argparse
import cupy as cp
import time
import random


def main():
    parser = argparse.ArgumentParser(
        description="GPU malloc/free stress generator")
    parser.add_argument("--throughput", type=int, default=1000,
                        help="Number of malloc/free events per second (default: 1000)")
    parser.add_argument("--allocate-size", type=int, default=8,
                        help="Size of each allocation in bytes (default: 8)")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of mallocs that will later be freed (0.0 - 1.0)")
    args = parser.parse_args()

    throughput = args.throughput
    alloc_size = args.allocate_size
    ratio = args.ratio

    # Store references to allocated GPU blocks
    allocated = []
    interval = 1.0 / throughput

    print(f"Running GPU stress generator: {throughput} events/sec, "
          f"alloc_size={alloc_size} bytes, free ratio={ratio}")

    try:
        while True:
            if not allocated or random.random() < ratio:
                # malloc on GPU
                n_elems = (alloc_size + 3) // 4  # round to float32 size
                arr = cp.empty(n_elems, dtype=cp.float32)
                allocated.append(arr)
            else:
                # free a random block
                idx = random.randrange(len(allocated))
                del allocated[idx]
                cp._default_memory_pool.free_all_blocks()

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExiting stress generator")


if __name__ == "__main__":
    main()
