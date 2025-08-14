# GPU memleak trace

GPU Memory Leak Diagnostic & Monitoring Tool

This tool helps diagnose GPU-related issues by tracing memory allocations and deallocations in real time. It detects potential leaks, correlates them with process metadata (PID, user, command), and periodically reports statistics such as total leaked memory. The collected data can also be exported as Prometheus metrics, enabling integration with dashboards like Grafana for visualization and long-term analysis.

## CLI

```shell
Usage of gpu-trace:
      --export-metrics             Export metrics as Prometheus exporter
      --interval duration          Trace print interval (default 2s)
      --libcuda-path string        Path to libcuda.so (default "/usr/lib/x86_64-linux-gnu/libcuda.so")
      --log-verbose string         Log verbosity level (DEBUG, INFO, WARN, ERROR) (default "INFO")
      --trace-print                Enable periodic printing of allocation map
      --update-interval duration   Metrics update interval (default 2s)

```
## Trace memory leaked

```shell
root@gpu1 . ~# ./gpu-memleak-trace --trace-print

Time: 2025-08-13T11:45:00+07:00
PID   USER     COMM        LEAKED
1234  root     python3     12.3MB
5678  vu       myapp       4.7MB
----     ----  ----    -----------

```

## Expose metrics as Promtheus exporter

```shell

root@gpu1 . ~# ./gpu-memleak-trace --export-metrics --port 8080

```

## Build
