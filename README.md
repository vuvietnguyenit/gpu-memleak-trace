# GPU memleak trace

GPU Memory Leak Diagnostic & Monitoring Tool based on eBPF

This tool helps diagnose GPU-related issues by tracing memory allocations and deallocations in real time. It detects potential leaks, correlates them with process metadata (PID, user, command), and periodically reports statistics such as total leaked memory.

## Features
- Multi-GPU aware: Traces allocations across all GPUs on a node, not just a single device, making it suitable for modern multi-GPU servers.
- Real-time leak detection with process context.

## Build
```shell
root@gpu1 ~/gpu-memleak-trace (main)# make build 
>> Generate eBPF code from .bpf.c
go generate ./src/go
>> Building gpu-memleak-trace
go build  -ldflags "-s -w" -o ./bin/gpu-memleak-trace ./src/go
root@gpu1 ~/gpu-memleak-trace (main)# 
```

## Trace memory leaked by command

In this case, we can trace total leaked bytes in each GPU
First, need to run trace tool:

```shell
root@gpu1 ~/gpu-memleak-trace (main)# ./bin/gpu-memleak-trace --trace-print
time=2025-09-14T15:51:22.213+07:00 level=INFO msg="eBPF program running... Press Ctrl+C to exit."
time=2025-09-14T15:51:22.213+07:00 level=INFO msg="Running in DEBUG mode: ignoring --trace-print and --export-metrics"
```
When it run, if we have any leak allocated, the leaked result will be print to console.
***Example:***
```text
--- Scan Time: 2025-09-14 15:51:52 ---

PID=603409 TID=603409 UID=0 DEV=0 Comm=python -> TotalSize=9.44 MB LastTs=2025-09-14 15:51:52.213767602
Top allocations for TID=603409:
  Size=512.00 B, Ptr=0x0000735467000000
  Size=512.00 B, Ptr=0x0000735467000200
  Size=512.00 B, Ptr=0x0000735467000400
  Size=512.00 B, Ptr=0x0000735467000600
  Size=512.00 B, Ptr=0x0000735467000800
```
If you want to have a better overall view of the trace tool, you need to see some test cases at: [Experimental](docs/experimentals.md)



