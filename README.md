# GPU memleak trace

GPU Memory Leak Diagnostic & Monitoring Tool based on eBPF

This tool helps diagnose GPU-related issues by tracing memory allocations and deallocations in real time. It detects potential leaks, correlates them with process metadata (PID, user, command), and periodically reports statistics such as total leaked memory. The collected data can also be exported as Prometheus metrics.

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
time=2025-08-25T16:25:23.756+07:00 level=INFO msg="eBPF program running... Press Ctrl+C to exit."
time=2025-08-25T16:25:23.757+07:00 level=INFO msg="Running in DEBUG mode: ignoring --trace-print and --export-metrics"
-------------------- 2025-08-25T16:25:25+07:00 --------------------
NO EVENT.
...
```
When it run, if we have any leak allocated, the leaked result will be print to console.
***Example:***
```text
-------------------- 2025-08-25T15:45:33+07:00 --------------------
PID: 2166793 / UID: 0
  python:2166793
    0x7d332a000000:gpu_0: 20.00 MB
    0x7d332cc00000:gpu_0: 16.00 MB
    0x7d332bc00000:gpu_0: 16.00 MB
    0x7d331d400000:gpu_0: 8.00 MB
    0x7d3343000000:gpu_0: 2.00 MB
    0x7d331d200400:gpu_0: 128.00 KB
    0x7d331d200000:gpu_0: 1.00 KB
  pt_autograd_0:2167089
    0x7d332b400000:gpu_0: 8.00 MB
    0x7d331d220800:gpu_0: 128.00 KB
    0x7d331d220400:gpu_0: 1.00 KB

TOTAL LEAKED: 70.25 MB // Allocated total 70.25 MB in process 2166793 on GPU 0 (in this result doesn't include pointer from another GPU)
-------------------- 2025-08-25T15:45:35+07:00 --------------------
PID: 2166793 / UID: 0
  python:2166793
    0x7d332a000000:gpu_0: 20.00 MB
    0x7d332bc00000:gpu_0: 16.00 MB
    0x7d332cc00000:gpu_0: 16.00 MB
    0x7d331d400000:gpu_0: 8.00 MB
    0x7d3343000000:gpu_0: 2.00 MB
    0x7d331d200400:gpu_0: 128.00 KB
    0x7d331d200000:gpu_0: 1.00 KB
  pt_autograd_0:2167089
    0x7d332b400000:gpu_0: 8.00 MB
    0x7d331d220800:gpu_0: 128.00 KB
    0x7d331d220400:gpu_0: 1.00 KB

TOTAL LEAKED: 70.25 MB

-------------------- 2025-08-25T15:45:37+07:00 --------------------
PID: 2166793 / UID: 0
  python:2166793
    0x7d332a000000:gpu_0: 20.00 MB
    0x7d332cc00000:gpu_0: 16.00 MB
    0x7d331d400000:gpu_0: 8.00 MB
    0x7d3343000000:gpu_0: 2.00 MB
    0x7d331d200400:gpu_0: 128.00 KB
    0x7d331d200000:gpu_0: 1.00 KB
  pt_autograd_0:2167089
    0x7d332b400000:gpu_0: 8.00 MB
    0x7d331d220800:gpu_0: 128.00 KB
    0x7d331d220400:gpu_0: 1.00 KB

TOTAL LEAKED: 54.25 MB // Freed 16.00 MB at pointer 0x7d332bc00000 (GPU 0), thread 2166793, at 2025-08-25T15:45:37+07:00
```
In the above example, we have many records, each corresponding to one PID (Process ID) in Linux. Basically, the record can be described as:

```text
PID: 2166793 / UID: 0 // PID/UID
  python:2166793 // COMM:ThreadID
    0x7d332a000000:gpu_0: 20.00 MB // Pointet:DeviceID:SizeAllocated
    0x7d332cc00000:gpu_0: 16.00 MB
    0x7d331d400000:gpu_0: 8.00 MB
    0x7d3343000000:gpu_0: 2.00 MB
    0x7d331d200400:gpu_0: 128.00 KB
    0x7d331d200000:gpu_0: 1.00 KB
  pt_autograd_0:2167089
    0x7d332b400000:gpu_0: 8.00 MB
    0x7d331d220800:gpu_0: 128.00 KB
    0x7d331d220400:gpu_0: 1.00 KB

TOTAL LEAKED: 54.25 MB // Total size leaked by process

```
If you want to have a better overall view of the trace tool, you need to see some test cases at: [Experimental](docs/experimentals.md)



