# Experimentals

The following test scenarios are simulated with Python scripts to evaluate the behavior of the GPU memory tracer.  
These cases help validate correctness of allocation tracking, detection of leaks, and handling of high-load conditions.

## Envinroment

The following environment was used for all experiments:

- **Operating System**: Ubuntu 24.04.2 LTS
- **CUDA Driver**: 12.8
- **PyTorch**: 2.8.0
- **Python**: 3.12
- **GPU Model**: NVIDIA GeForce RTX 5090
- **GPU Driver Version**: 570.133.07

## Test malloc/free actions
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python script/train_simulator.py
```
Result
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

TOTAL LEAKED: 70.25 MB
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

TOTAL LEAKED: 54.25 MB // Freed 16.00 MB at 0x7d332bc00000 (gpu 0) thread 2166793
```


## Test memory allocated by multi-process - each process has only one thread
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python scripts/gpu_mem_stressor.py --procs 4 --duration 120 --min-mb 2 --max-mb 10 --sleep-ms 25 --leak-prob 0.1 --compute 

```
Result
```text
-------------------- 2025-08-25T14:24:06+07:00 --------------------
NO EVENT.
-------------------- 2025-08-25T14:24:08+07:00 --------------------
PID: 2082548 / UID: 0
  python:2082548
    0x725540400000:gpu_0: 20.00 MB
    0x725522000000:gpu_0: 10.00 MB
    0x72553b800000:gpu_0: 8.00 MB
    0x72553b400000:gpu_0: 2.00 MB
    0x72553b000000:gpu_0: 2.00 MB
    0x72553b600400:gpu_0: 128.00 KB
    0x72553b600000:gpu_0: 1.00 KB

TOTAL LEAKED: 42.13 MB

PID: 2082547 / UID: 0
  python:2082547
    0x74d3a4400000:gpu_0: 20.00 MB
    0x74d386a00000:gpu_0: 20.00 MB
    0x74d387e00000:gpu_0: 20.00 MB
    0x74d386000000:gpu_0: 10.00 MB
    0x74d39f800000:gpu_0: 8.00 MB
    0x74d39f400000:gpu_0: 2.00 MB
    0x74d39f000000:gpu_0: 2.00 MB
    0x74d39f600400:gpu_0: 128.00 KB
    0x74d39f600000:gpu_0: 1.00 KB

TOTAL LEAKED: 82.13 MB

PID: 2082550 / UID: 0
  python:2082550
    0x7083f0400000:gpu_0: 20.00 MB
    0x7083d2000000:gpu_0: 20.00 MB
    0x7083eb800000:gpu_0: 8.00 MB
    0x7083eb000000:gpu_0: 2.00 MB
    0x7083eb400000:gpu_0: 2.00 MB
    0x7083eb600400:gpu_0: 128.00 KB
    0x7083eb600000:gpu_0: 1.00 KB

TOTAL LEAKED: 52.13 MB

PID: 2082549 / UID: 0
  python:2082549
    0x7ab4faa00000:gpu_0: 20.00 MB
    0x7ab518400000:gpu_0: 20.00 MB
    0x7ab4fa000000:gpu_0: 10.00 MB
    0x7ab513800000:gpu_0: 8.00 MB
    0x7ab513400000:gpu_0: 2.00 MB
    0x7ab513000000:gpu_0: 2.00 MB
    0x7ab513600400:gpu_0: 128.00 KB
    0x7ab513600000:gpu_0: 1.00 KB

TOTAL LEAKED: 62.13 MB
```

## Test malloc/free actions by multi-threading
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python script/gpu_mem_threads_leak.py --device cuda:0 --threads 5 --duration 60 --max-tensor-mb 10 --alloc-prob 0.65 --stats-interval 1 --memory-cap-mb 4096
```
Result:

```text
-------------------- 2025-08-25T16:16:27+07:00 --------------------
PID: 2181102 / UID: 0
  python:2181147
    0x78c246000000:gpu_0: 20.00 MB
    0x78c282400000:gpu_0: 20.00 MB
  python:2181148
    0x78c23e000000:gpu_0: 20.00 MB
  python:2181150
    0x78c252000000:gpu_0: 20.00 MB
    0x78c247400000:gpu_0: 2.00 MB
  python:2181151
    0x78c23c000000:gpu_0: 20.00 MB

TOTAL LEAKED: 102.00 MB

-------------------- 2025-08-25T16:16:29+07:00 --------------------
PID: 2181102 / UID: 0
  python:2181147
    0x78c282400000:gpu_0: 20.00 MB
    0x78c246000000:gpu_0: 20.00 MB
  python:2181148
    0x78c23e000000:gpu_0: 20.00 MB
  python:2181149
    0x78c23a000000:gpu_0: 20.00 MB
  python:2181150
    0x78c252000000:gpu_0: 20.00 MB
    0x78c238000000:gpu_0: 20.00 MB
    0x78c247400000:gpu_0: 2.00 MB
  python:2181151
    0x78c23c000000:gpu_0: 20.00 MB

TOTAL LEAKED: 142.00 MB
```
