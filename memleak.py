import random
import time

allocated = []

def allocate_memory():
    # Allocate a random size between 1KB and 1MB
    size_kb = random.randint(1, 1024)
    block = bytearray(size_kb * 1024)
    allocated.append(block)
    print(f"[+] Allocated {size_kb} KB (total blocks: {len(allocated)})")

def free_memory():
    if allocated:
        idx = random.randint(0, len(allocated) - 1)
        del allocated[idx]
        print(f"[-] Freed block at index {idx} (remaining blocks: {len(allocated)})")
    else:
        print("[!] No memory to free")

def main():
    try:
        while True:
            action = random.choice(['alloc', 'free'])
            if action == 'alloc':
                allocate_memory()
            else:
                free_memory()

            sleep_time = random.uniform(0.1, 1.0)  # sleep between 100ms to 1s
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Exiting and freeing all memory.")
        allocated.clear()

if __name__ == "__main__":
    main()
