import ctypes
import random
import time
import os

libc = ctypes.CDLL("libc.so.6")

# Correct return type for malloc: void*
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]

allocated = {}

def allocate_memory():
    size = random.randint(1, 1024 * 1024)
    ptr = libc.malloc(size)
    if not ptr:
        print("malloc failed")
        return
    allocated[ptr] = size
    print(f"[+] malloc: ptr=0x{ptr:x} size={size} bytes (total: {len(allocated)})")
    return ptr

def free_memory():
    if not allocated:
        print("[!] No allocations to free")
        return
    ptr = random.choice(list(allocated.keys()))
    libc.free(ptr)
    del allocated[ptr]
    print(f"[-] free: ptr=0x{ptr:x} (remaining: {len(allocated)})")

def main():
    print(f"{os.getpid()} - Memory Leak Simulation Started")
    try:
        while True:
            if random.choice(["alloc", "free"]) == "alloc":
                allocate_memory()
            else:
                free_memory()
            time.sleep(random.uniform(3, 4.5))
    except KeyboardInterrupt:
        print("Exiting, cleaning up...")
        for ptr in allocated:
            libc.free(ptr)
        allocated.clear()

if __name__ == "__main__":
    main()
