#!/usr/bin/env python3
"""
pid_monitor_csv.py

Monitor CPU, memory, and disk I/O usage for specific PIDs or processes by name.
Appends results as CSV lines (with optional description column).
"""

import argparse
import time
import psutil
import os
import sys
from datetime import datetime, timedelta

BASE_HEADER = "timestamp,pid,name,cpu_percent,memory_mb,memory_percent,disk_read_kb,disk_write_kb"


def build_header(with_desc: bool) -> str:
    if with_desc:
        return "timestamp,description,pid,name,cpu_percent,memory_mb,memory_percent,disk_read_kb,disk_write_kb"
    return BASE_HEADER


def print_header(file, with_desc: bool):
    """Write header only if file is empty."""
    if os.stat(file.name).st_size == 0:
        file.write(build_header(with_desc) + "\n")
        file.flush()


def print_metrics(pids, interval, desc, file=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # with microseconds
    for pid in pids:
        try:
            p = psutil.Process(pid)
            cpu = p.cpu_percent(interval=interval)  # accurate, like `top`
            mem = p.memory_info().rss / (1024 * 1024)
            mem_percent = p.memory_percent()
            io = p.io_counters()

            if desc:
                line = (
                    f"{ts},{desc},{pid},{p.name()},{cpu:.1f},"
                    f"{mem:.1f},{mem_percent:.1f},"
                    f"{io.read_bytes/1024:.1f},{io.write_bytes/1024:.1f}"
                )
            else:
                line = (
                    f"{ts},{pid},{p.name()},{cpu:.1f},"
                    f"{mem:.1f},{mem_percent:.1f},"
                    f"{io.read_bytes/1024:.1f},{io.write_bytes/1024:.1f}"
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            if desc:
                line = f"{ts},{desc},{pid},N/A,0,0,0,0,0"
            else:
                line = f"{ts},{pid},N/A,0,0,0,0,0"

        print(line)
        if file:
            file.write(line + "\n")
            file.flush()


def find_pids_by_name(name: str):
    """Return list of PIDs matching the process name (case-insensitive)."""
    pids = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() == name.lower():
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CPU, memory, and disk usage for specific PIDs or process name (CSV output)"
    )
    parser.add_argument("--pids", type=int, nargs="+",
                        help="List of PIDs to monitor")
    parser.add_argument("--name", type=str,
                        help="Process name to monitor (all matching processes)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Refresh interval in seconds")
    parser.add_argument("--duration", type=int, default=0,
                        help="How long to run in seconds (0 = run forever)")
    parser.add_argument("--output", type=str,
                        help="Optional file to save CSV results")
    parser.add_argument("--desc", type=str, default="",
                        help="Optional description to include in CSV rows")
    args = parser.parse_args()

    if not args.pids and not args.name:
        print("[ERROR] You must specify either --pids or --name")
        sys.exit(1)

    # Resolve PIDs
    pids = args.pids or find_pids_by_name(args.name)
    if not pids:
        print(f"[ERROR] No processes found for: {args.name or args.pids}")
        sys.exit(1)

    file = None
    if args.output:
        file = open(args.output, "a")
        print_header(file, with_desc=bool(args.desc))

    end_time = None
    if args.duration > 0:
        end_time = datetime.now() + timedelta(seconds=args.duration)

    try:
        while True:
            # Stop if duration expired
            if end_time and datetime.now() >= end_time:
                print("[INFO] Duration finished, exiting.")
                break

            if args.name:  # refresh PIDs to catch restarts
                pids = find_pids_by_name(args.name)
                if not pids:
                    print(f"[WARN] No processes found for name '{args.name}'")
                    time.sleep(args.interval)
                    continue

            print_metrics(pids, args.interval, args.desc, file)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if file:
            file.close()


if __name__ == "__main__":
    main()
