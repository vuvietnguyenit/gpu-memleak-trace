#!/usr/bin/env python3
"""
pid_monitor_csv.py

Monitor CPU, memory, and disk I/O usage for specific PIDs.
Prints results as CSV lines (PID, name, CPU%, memory_MB, memory%, disk_read_KB, disk_write_KB).
"""

import argparse
import time
import psutil
import os


HEADER = "timestamp,pid,name,cpu_percent,memory_mb,memory_percent,disk_read_kb,disk_write_kb"


def print_header(file):
    """Write header only if file is empty."""
    if os.stat(file.name).st_size == 0:
        file.write(HEADER + "\n")
        file.flush()


def print_metrics(pids, interval, file=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for pid in pids:
        try:
            p = psutil.Process(pid)
            cpu = p.cpu_percent(interval=interval)  # accurate, like `top`
            mem = p.memory_info().rss / (1024 * 1024)
            mem_percent = p.memory_percent()
            io = p.io_counters()
            line = (
                f"{ts},{pid},{p.name()},{cpu:.1f},"
                f"{mem:.1f},{mem_percent:.1f},"
                f"{io.read_bytes/1024:.1f},{io.write_bytes/1024:.1f}"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            line = f"{ts},{pid},N/A,0,0,0,0,0"

        print(line)
        if file:
            file.write(line + "\n")
            file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CPU, memory, and disk usage for specific PIDs (CSV output)"
    )
    parser.add_argument("--pids", type=int, nargs="+",
                        required=True, help="List of PIDs to monitor")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Refresh interval in seconds")
    parser.add_argument("--output", type=str,
                        help="Optional file to save CSV results")
    args = parser.parse_args()

    file = None
    if args.output:
        # Open in append mode
        file = open(args.output, "a")
        print_header(file)

    try:
        while True:
            print_metrics(args.pids, args.interval, file)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if file:
            file.close()


if __name__ == "__main__":
    main()
