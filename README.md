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
-------------------- 2025-09-09T17:17:26+07:00 --------------------
PID: 1551384 / UID: 0
  python:1551384 // **Followed by format:** `COMM:ThreadID`
   [2025-09-09T17:17:26.310983268+07:00]  0x761564000000  gpu:0   allocated size:256.00 MB // **Followed by format:** `timestamp_allocated pointer device_id allocated_size`
   [2025-09-09T17:17:25.310191069+07:00]  0x761574000000  gpu:0   allocated size:256.00 MB
   [2025-09-09T17:17:24.309494050+07:00]  0x761584000000  gpu:0   allocated size:256.00 MB
   [2025-09-09T17:17:23.308732492+07:00]  0x761594000000  gpu:0   allocated size:256.00 MB
   [2025-09-09T17:17:22.308044564+07:00]  0x7615a4000000  gpu:0   allocated size:256.00 MB
   [2025-09-09T17:17:21.307175606+07:00]  0x7615b4000000  gpu:0   allocated size:256.00 MB
   [2025-09-09T17:17:20.306554487+07:00]  0x7615c4000000  gpu:0   allocated size:256.00 MB

TOTAL LEAKED: 1.75 GB // Allocated total 1.75 GB in process 1551384 on GPU 0 (in this result doesn't include pointer from another GPU)
```
If you want to have a better overall view of the trace tool, you need to see some test cases at: [Experimental](docs/experimentals.md)



