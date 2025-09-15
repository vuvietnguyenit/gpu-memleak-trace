#!/bin/bash

# Run gpu_stressor in the background
python gpu_stressor.py --throughput 4000 --allocate-size 2048 --ratio 1.0 &

# Run pid_monitor in the background
python pid_monitor_csv.py \
  --pids $(ps aux | grep nvidia-smi | grep -v grep | awk '{print $2}') \
         $(ps aux | grep trace-print | grep -v grep | awk '{print $2}') \
  --interval 0.2 \
  --duration 120 \
  --output ../sample/metrics.csv \
  --desc "--throughput 4000 --allocate-size 2048 --free-ratio 0.0 (disable console printer)" &

# Wait for both to finish
wait


