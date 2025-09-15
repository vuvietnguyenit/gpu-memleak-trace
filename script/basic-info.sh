#!/bin/bash
# basic-info.sh
# Collects basic Linux system information
# Generated on 2024-09-11 by ChatGPT

echo "=============================="
echo " 🖥️  System & Kernel Info"
echo "=============================="
uname -a
echo
hostnamectl 2>/dev/null || echo "hostnamectl not available"

echo
echo "=============================="
echo " 📦 Distribution Info"
echo "=============================="
if [ -f /etc/os-release ]; then
    cat /etc/os-release
else
    echo "/etc/os-release not found"
fi

echo
echo "=============================="
echo " 💾 Hardware Info"
echo "=============================="
lscpu | grep -E 'Model name|Socket|Thread|CPU\(s\)|Arch'
echo
free -h
echo
lsblk
echo
df -h --total | grep -E 'Filesystem|total'

echo
echo "=============================="
echo " 📡 Network Info"
echo "=============================="
ip a
echo
ip r
echo
ss -tulnp | head -20

echo
echo "=============================="
echo " 👤 Users & Processes"
echo "=============================="
whoami
id
echo
w
echo
ps aux --sort=-%mem | head -10

echo
echo "=============================="
echo " 🔐 Security Info"
echo "=============================="
if command -v getenforce &>/dev/null; then
    getenforce
else
    echo "SELinux not installed"
fi


echo
echo "=============================="
echo " 🎮 GPU Info"
echo "=============================="
if command -v nvidia-smi &>/dev/null; then
    echo "[NVIDIA GPUs detected]"
    nvidia-smi -L
    echo
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
elif command -v rocm-smi &>/dev/null; then
    echo "[AMD ROCm GPUs detected]"
    rocm-smi
else
    echo "No GPU info tool found (install nvidia-smi or rocm-smi)"
fi