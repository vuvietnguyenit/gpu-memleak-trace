
// go:build ignore
// SPDX-License-Identifier: MIT

/*
 *
 * Trace malloc/free with per-PID allocation counting.
 *
 * This eBPF program attaches uprobes to malloc and free to track
 * allocation behavior for leak detection.
 *
 * Author: Vu Nguyen (vunv)
 * Date: 2025-08-06
 */

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/ptrace.h>

char LICENSE[] SEC("license") = "GPL";

struct
{
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} pid_allocs SEC(".maps");

enum event_type
{
    EVENT_MALLOC = 0,
    EVENT_FREE,
};

// Define the event struct
struct event
{
    __u32 pid;
    __u64 tid;
    enum event_type type;
    union
    {
        __u64 ptr;
        __u64 size;
    };
    __u8 is_ret;
};

struct
{
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 25); // 16 MB buffer
} events SEC(".maps");

SEC("uprobe/malloc")
int trace_malloc_entry(struct pt_regs *ctx)
{
    struct event *e;
    __u64 size = PT_REGS_PARM1(ctx); // size_t size
    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
    {
        bpf_printk("uprobe/malloc reserve failed");
        return 0;
    }
    bpf_printk("malloc entry: size=%llu", size);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    e->pid = pid_tgid >> 32;
    e->tid = pid_tgid;
    e->type = EVENT_MALLOC;
    e->size = size;
    e->is_ret = 0;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uretprobe/malloc")
int trace_malloc_return(struct pt_regs *ctx)
{
    struct event *e;
    __u64 ret_ptr = PT_REGS_RC(ctx); // return value

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
    {
        bpf_printk("uretprobe/malloc reserve failed");
        return 0;
    }
    bpf_printk("malloc return: %llu", ret_ptr);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    e->pid = pid_tgid >> 32;
    e->tid = pid_tgid;
    e->type = EVENT_MALLOC;
    e->ptr = ret_ptr;
    e->is_ret = 1;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uprobe/free")
int trace_free(struct pt_regs *ctx)
{
    struct event *e;
    __u64 ptr = PT_REGS_PARM1(ctx); // void *ptr

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
    {
        bpf_printk("uprobe/free reserve failed");
        return 0;
    }

    bpf_printk("free: ptr=%llu", ptr);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    e->pid = pid_tgid >> 32;
    e->tid = pid_tgid;
    e->type = EVENT_FREE;
    e->ptr = ptr;
    e->is_ret = 0; // only entry, free has no useful return

    bpf_ringbuf_submit(e, 0);
    return 0;
}
