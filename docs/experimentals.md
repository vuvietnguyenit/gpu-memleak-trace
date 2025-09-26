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
Basic case use pytorch lib to help simulate allocate operator
Run experimental test: 

```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python script/train_simulator.py
```
gpu-memleak-trace result:
```text
--- Scan Time: 2025-09-26 10:25:48 ---                                                                                               
                                                                                                                                                                                                                                                                           
PID=2821094 TID=2821094 UID=0 DEV=0 Comm=python -> TotalSize=30.13 MB LastTs=2025-09-26 10:25:48.356790850                                                                                                                                                                 
Top allocations for TID=2821094:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x0000738608000000                                                                                                                                                                                                                                    
  Size=8.00 MB, Ptr=0x00007385fd400000                                                                                                                                                                                                                                     
  Size=2.00 MB, Ptr=0x0000738621000000                                                                                                                                                                                                                                     
  Size=128.00 KB, Ptr=0x00007385fd200400                                                                                                                                                                                                                                   
  Size=1.00 KB, Ptr=0x00007385fd200000                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                           
--- Scan Time: 2025-09-26 10:25:50 ---                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                           
PID=2821094 TID=2821094 UID=0 DEV=0 Comm=python -> TotalSize=46.13 MB LastTs=2025-09-26 10:25:50.513837333                                                                                                                                                                 
PID=2821094 TID=2821381 UID=0 DEV=0 Comm=pt_autograd_0 -> TotalSize=8.13 MB LastTs=2025-09-26 10:25:49.850939944 // we got more TID                                                                                                                                            
Top allocations for TID=2821094:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x0000738608000000                                                                                              
  Size=16.00 MB, Ptr=0x0000738609c00000                                                                                                                                                                                                                                    
  Size=8.00 MB, Ptr=0x00007385fd400000                                                                                                                                                                                                                                     
  Size=2.00 MB, Ptr=0x0000738621000000                                                                                                                                                                                                                                     
  Size=128.00 KB, Ptr=0x00007385fd200400                                                                                                                                                                                                                                   
Top allocations for TID=2821381:                                                                                                                                                                                                                                           
  Size=8.00 MB, Ptr=0x0000738609400000                                                                                                                                                                                                                                     
  Size=128.00 KB, Ptr=0x00007385fd220800                                                                                                                                                                                                                                   
  Size=1.00 KB, Ptr=0x00007385fd220400
```


## Test memory allocated by multi-process - each process has only one thread
This test case will be generate 4 proccesses (one thread per proc), run in 120 seconds, size allocated generated in range [2, 10] MB, each allocate request will sleep in 25ms with 10% leak probaly
Run experimental test: 
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python scripts/gpu_mem_stressor.py --procs 4 --duration 120 --min-mb 2 --max-mb 10 --sleep-ms 25 --leak-prob 0.1 --compute 

```
gpu-memleak-trace result:

```text
--- Scan Time: 2025-09-26 10:14:50 ---                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                           
PID=2816887 TID=2816887 UID=0 DEV=0 Comm=python -> TotalSize=532.13 MB LastTs=2025-09-26 10:14:50.537371934                                                                                                                                                                
PID=2816888 TID=2816888 UID=0 DEV=0 Comm=python -> TotalSize=492.13 MB LastTs=2025-09-26 10:14:50.537722275                                                                                                                                                                
PID=2816889 TID=2816889 UID=0 DEV=0 Comm=python -> TotalSize=472.13 MB LastTs=2025-09-26 10:14:50.487077475                                                                                                                                                                
PID=2816890 TID=2816890 UID=0 DEV=0 Comm=python -> TotalSize=452.13 MB LastTs=2025-09-26 10:14:49.626359419                                                                                                                                                                
Top allocations for TID=2816888:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x000075081dc00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x000075081aa00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x000075081c800000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x0000750838400000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x000075081f000000                                                                                                                                                                                                                                    
Top allocations for TID=2816887:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x0000797dfd200000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x0000797dfaa00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x0000797dfbe00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x0000797e18400000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x0000797dff000000                                                                                                                                                                                                                                    
Top allocations for TID=2816890:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x00006ffc05200000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00006ffc03e00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00006ffc02000000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00006ffc20400000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00006ffc06600000                                                                                                                                                                                                                                    
Top allocations for TID=2816889:                                                                                                                                                                                                                                           
  Size=20.00 MB, Ptr=0x00007c7edd200000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00007c7edaa00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00007c7edbe00000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00007c7ef8400000                                                                                                                                                                                                                                    
  Size=20.00 MB, Ptr=0x00007c7ede600000 
```

## Test malloc/free actions by multi-threading
Run experimental test: 
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python script/gpu_mem_threads_leak.py --device cuda:0 --threads 5 --duration 60 --max-tensor-mb 10 --alloc-prob 0.65 --stats-interval 1 --memory-cap-mb 4096
```
gpu-memleak-trace result:

```text
--- Scan Time: 2025-09-26 10:31:54 ---

PID=2823326 TID=2823368 UID=0 DEV=0 Comm=python -> TotalSize=100.00 MB LastTs=2025-09-26 10:31:45.154295469
PID=2823326 TID=2823370 UID=0 DEV=0 Comm=python -> TotalSize=62.00 MB LastTs=2025-09-26 10:31:52.568443898
PID=2823326 TID=2823371 UID=0 DEV=0 Comm=python -> TotalSize=20.00 MB LastTs=2025-09-26 10:31:04.762561961
PID=2823326 TID=2823369 UID=0 DEV=0 Comm=python -> TotalSize=20.00 MB LastTs=2025-09-26 10:31:28.925112202
Top allocations for TID=2823368:
  Size=20.00 MB, Ptr=0x0000715b0c400000
  Size=20.00 MB, Ptr=0x0000715ad2000000
  Size=20.00 MB, Ptr=0x0000715aca000000
  Size=20.00 MB, Ptr=0x0000715ac6000000
  Size=20.00 MB, Ptr=0x0000715ac0000000
Top allocations for TID=2823370:
  Size=20.00 MB, Ptr=0x0000715ade000000
  Size=20.00 MB, Ptr=0x0000715ac4000000
  Size=20.00 MB, Ptr=0x0000715abe000000
  Size=2.00 MB, Ptr=0x0000715ad3400000
Top allocations for TID=2823371:
  Size=20.00 MB, Ptr=0x0000715ac8000000
Top allocations for TID=2823369:
  Size=20.00 MB, Ptr=0x0000715ac2000000
```


## Test anomaly allocation
This experiment simulates an anomaly allocation by running a process with multiple threads and specifying one thread as the anomaly thread. That thread will generate an anomalous allocation. The anomaly allocation has a very large size, and we can observe it when running a trace tool.
The script and arguments above, each normal allocation is 10 MB. However, one thread produces the anomaly allocation. This anomaly appears in thread T0 at iteration 5, with a size of 2096 MB.
```shell
(.venv) root@gpu1 ~/gpu-memleak-trace (main)# python script/gpu_alloc_anomaly.py --device cuda:0 --threads 4 --iterations 20 --normal-mb 10 --anomaly-mb 2096 --anomaly-thread 0 --anomaly-iter 5
```
gpu-memleak-trace result:

```text
--- Scan Time: 2025-09-26 10:50:36 ---

PID=2830725 TID=2830758 UID=0 DEV=0 Comm=python -> TotalSize=2.09 GB LastTs=2025-09-26 10:50:35.567509175
PID=2830725 TID=2830759 UID=0 DEV=0 Comm=python -> TotalSize=40.00 MB LastTs=2025-09-26 10:50:35.462147309
PID=2830725 TID=2830760 UID=0 DEV=0 Comm=python -> TotalSize=40.00 MB LastTs=2025-09-26 10:50:35.462600987
PID=2830725 TID=2830761 UID=0 DEV=0 Comm=python -> TotalSize=40.00 MB LastTs=2025-09-26 10:50:35.462843205
Top allocations for TID=2830761:
  Size=10.00 MB, Ptr=0x000077c5a5000000
  Size=10.00 MB, Ptr=0x000077c58ac00000
  Size=10.00 MB, Ptr=0x000077c58de00000
  Size=10.00 MB, Ptr=0x000077c591000000
Top allocations for TID=2830759:
  Size=10.00 MB, Ptr=0x000077c5aa400000
  Size=10.00 MB, Ptr=0x000077c58b600000
  Size=10.00 MB, Ptr=0x000077c58d400000
  Size=10.00 MB, Ptr=0x000077c58f200000
Top allocations for TID=2830760:
  Size=10.00 MB, Ptr=0x000077c5aae00000
  Size=10.00 MB, Ptr=0x000077c58c000000
  Size=10.00 MB, Ptr=0x000077c58e800000
  Size=10.00 MB, Ptr=0x000077c590600000
Top allocations for TID=2830758:
  Size=2.05 GB, Ptr=0x000077c4e4000000
  Size=10.00 MB, Ptr=0x000077c58a000000
  Size=10.00 MB, Ptr=0x000077c589200000
  Size=10.00 MB, Ptr=0x000077c58ca00000
  Size=10.00 MB, Ptr=0x000077c58fc00000

```


## Test allocation only

```shell
(.venv) root@gpu1 ~/g/script (dev)# python memleak_stressor.py --mode torch --size 256MiB --device 0 --max-bytes 8GiB --interval 1

[memleak] PID=976561 mode=torch size=256 MiB interval=1.0s
[memleak] Will stop after max-bytes=8 GiB
[memleak] iters=10 leaked=2 GiB elapsed=9.0s rate~284 MiB/s
[memleak] iters=20 leaked=5 GiB elapsed=19.0s rate~269 MiB/s
[memleak] iters=30 leaked=8 GiB elapsed=29.0s rate~265 MiB/s
[memleak] Done. Total leaked ~ 8 GiB in 32 allocations.

```
gpu-memleak-trace result:

```text
--- Scan Time: 2025-09-26 10:52:30 ---                                                                                               
                                                                  
PID=2831498 TID=2831498 UID=0 DEV=0 Comm=python -> TotalSize=6.00 GB LastTs=2025-09-26 10:52:30.121510906                                                                                                                                                                  
Top allocations for TID=2831498:                                  
  Size=256.00 MB, Ptr=0x000070ff70000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff60000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff50000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff40000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff30000000                                                                                             
                                                                  
--- Scan Time: 2025-09-26 10:52:32 ---                                                                                               
                                                                  
PID=2831498 TID=2831498 UID=0 DEV=0 Comm=python -> TotalSize=6.50 GB LastTs=2025-09-26 10:52:32.125966389                                                                                                                                                                  
Top allocations for TID=2831498:                                  
  Size=256.00 MB, Ptr=0x000070ff70000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff60000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff50000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff40000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff30000000                                                                                             
                                                                  
--- Scan Time: 2025-09-26 10:52:34 ---                                                                                               
                                                                  
PID=2831498 TID=2831498 UID=0 DEV=0 Comm=python -> TotalSize=7.00 GB LastTs=2025-09-26 10:52:34.127126012                                                                                                                                                                  
Top allocations for TID=2831498:                                  
  Size=256.00 MB, Ptr=0x000070ff70000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff60000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff50000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff40000000                                                                                             
  Size=256.00 MB, Ptr=0x000070ff30000000
```
