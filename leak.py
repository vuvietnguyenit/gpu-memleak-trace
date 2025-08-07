import time
import random

# Storage for leaked memory (so it's not garbage collected)
leaks = []

def leak_memory():
    while True:
        # Random size between 1 KB and 1 MB
        size = random.randint(1024, 1024 * 1024)

        # Allocate a bytes object (won't be freed)
        block = bytearray(size)
        leaks.append(block)

        print(f"[leakmem] Allocated {size / 1024:.2f} KB - Total blocks: {len(leaks)}")

        # Sleep a bit to simulate real process behavior
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        leak_memory()
    except KeyboardInterrupt:
        print("\nStopped leaking memory.")
