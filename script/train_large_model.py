#!/usr/bin/env python3
"""
train_large_model.py

Simulates a real-world ML training job:
- Loads a large model once
- Runs continuously (while True)
- Training loop with fake data
- Sometimes allocates/free memory
- Sometimes leaks memory (simulate real leak case)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import random
import gc
import time


# -----------------------------
# Large model
# -----------------------------
class LargeModel(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=8, output_size=10):
        super(LargeModel, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Resource monitor
# -----------------------------
def print_resources(tag=""):
    cpu_mem = psutil.Process().memory_info().rss / 1024**2
    msg = f"[{tag}] CPU RAM: {cpu_mem:.2f} MB"
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        msg += f" | GPU Alloc: {allocated:.2f} MB | GPU Reserved: {reserved:.2f} MB"
    else:
        msg += " | GPU: not available"
    print(msg)


# -----------------------------
# Memory allocation cases
# -----------------------------
leak_container = []  # holds leaked tensors


def maybe_allocate(leak_prob=0.3):
    """Sometimes allocate/free memory, sometimes leak"""
    size = random.choice([128, 256, 512])
    tensor = torch.randn(
        1024, size, device="cuda" if torch.cuda.is_available() else "cpu")
    if random.random() < leak_prob:
        # simulate leak by keeping reference
        leak_container.append(tensor)
        print(f"  ⚠️ Leaked tensor of shape {tensor.shape}")
    else:
        # free immediately
        del tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  ✅ Allocated and freed tensor")


# -----------------------------
# Continuous training job
# -----------------------------
def train_job(batch_size=32, batches_per_epoch=50, leak_prob=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model once
    model = LargeModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print_resources("After model load")

    epoch = 0
    while True:  # infinite loop
        epoch += 1
        print(f"\nEpoch {epoch} (continuous job)")
        for batch in range(batches_per_epoch):
            # fake data
            inputs = torch.randn(batch_size, 2048, device=device)
            labels = torch.randint(0, 10, (batch_size,), device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # simulate random allocation/free/leak
            if batch % 10 == 0:
                maybe_allocate(leak_prob=leak_prob)

            if batch % 10 == 0:
                print(
                    f"  Batch {batch}/{batches_per_epoch}, Loss: {loss.item():.4f}")
                print_resources(f"Epoch {epoch} Batch {batch}")

        # optional small sleep so it doesn’t hog 100% CPU forever
        time.sleep(0.5)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train_job(batch_size=32, batches_per_epoch=30, leak_prob=0.3)
