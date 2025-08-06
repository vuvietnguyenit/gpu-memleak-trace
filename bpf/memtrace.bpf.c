
// go:build ignore

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/ptrace.h>
#include "events.h"

char LICENSE[] SEC("license") = "GPL";
int PID_TEST = 117243;

SEC("uprobe/malloc")
int trace_malloc_enter(struct pt_regs *ctx)
{
    __u64 size = PT_REGS_PARM1(ctx);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;
    if (pid != PID_TEST)
        return 0; // filter by TID
    bpf_map_update_elem(&tmp_alloc_size, &tid, &size, BPF_ANY);
    // bpf_printk("malloc: tid=%llu size=%llu", tid, size);
    return 0;
}

SEC("uretprobe/malloc")
int malloc_return(struct pt_regs *ctx)
{
    __u64 ptr = PT_REGS_RC(ctx);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;
    if (pid != PID_TEST)
        return 0; // filter by TID

    __u64 *size = bpf_map_lookup_elem(&tmp_alloc_size, &tid);
    if (!size || ptr == 0)
    {
        bpf_printk("malloc tmp_alloc_size: pid=%d tid=%d ptr=0x%llx size=%llu not found", pid, tid, ptr, size ? *size : 0);
        return 0;
    }
    struct alloc_info_t info = {
        .size = *size,
        .timestamp_ns = bpf_ktime_get_ns(),
    };
    bpf_map_update_elem(&allocs, &ptr, &info, BPF_ANY);
    bpf_map_delete_elem(&tmp_alloc_size, &tid);

    __u64 zero = 0;
    __u64 *total = bpf_map_lookup_elem(&unfreed_bytes, &pid);
    __u64 s = *size;
    if (!total)
    {
        // initialize first
        bpf_map_update_elem(&unfreed_bytes, &pid, &s, BPF_ANY);
        total = bpf_map_lookup_elem(&unfreed_bytes, &pid);
        if (!total)
        {
            bpf_printk("malloc total: pid=%d ptr=0x%llx size=%llu total_size not found", pid, ptr, *size);
            return 0;
        }
    }

    __sync_fetch_and_add(total, *size);
    struct memleak_event_t *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->unfreed_bytes = *total;

    bpf_ringbuf_submit(e, 0);
    bpf_printk("malloc: pid=%d ptr=0x%llx size+=%llu total_size=%llu", pid, ptr, *size, *total);
    return 0;
}

SEC("uprobe/free")
int trace_free(struct pt_regs *ctx)
{
    __u64 ptr = PT_REGS_PARM1(ctx);
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    if (pid != PID_TEST)
        return 0; // filter by TID

    struct alloc_info_t *info = bpf_map_lookup_elem(&allocs, &ptr);
    if (!info)
    {
        bpf_printk("free: pid=%d ptr=0x%llx not found", pid, ptr);
        return 0;
    }
    bpf_map_delete_elem(&allocs, &ptr);

    __u64 size = info->size;
    __u64 *total = bpf_map_lookup_elem(&unfreed_bytes, &pid);
    __u64 total_val = 0;
    if (total)
    {
        __sync_fetch_and_sub(total, size);
        total_val = *total;
    }
    bpf_printk("free success: pid=%d ptr=0x%llx size-=%llu total_size=%llu", pid, ptr, size, total_val);
    struct memleak_event_t *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = pid;
    e->unfreed_bytes = total_val;

    bpf_ringbuf_submit(e, 0);
    return 0;
}