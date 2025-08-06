#ifndef __EVENT_H__
#define __EVENT_H__
#include <linux/ptrace.h>

struct alloc_info_t
{
    __u64 size;
    __u64 timestamp_ns;
};

struct memleak_event_t
{
    __u32 pid;
    __u64 unfreed_bytes;
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);   // TID
    __type(value, __u64); // size param
} tmp_alloc_size SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1000000);
    __type(key, __u64); // ptr
    __type(value, struct alloc_info_t);
} allocs SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 409600);
    __type(key, __u32);   // PID
    __type(value, __u64); // unfreed total bytes
} unfreed_bytes SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MB
} events SEC(".maps");

#endif /* __EVENT_H__ */