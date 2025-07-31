#ifndef __EVENT_H__
#define __EVENT_H__
#include <linux/ptrace.h>

enum event_type_t
{
    EVENT_MALLOC,
    EVENT_FREE,
};

struct event_t
{
    __u32 pid;
    enum event_type_t type;
    __u32 _pad;
    union
    {
        __u64 size;
        __u64 ptr;
    };
};

struct alloc_info_t
{
    __u64 size;
    __u64 timestamp_ns;
};

// Hash map: key = malloc ptr, value = alloc_info
struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);
    __type(value, struct alloc_info_t);
    __uint(max_entries, 100000);
} allocs SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MB
} events SEC(".maps");

// Save malloc size before return (per-TID)
struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);   // tid
    __type(value, __u64); // size
    __uint(max_entries, 1024);
} temp_size SEC(".maps");

#endif /* __EVENT_H__ */