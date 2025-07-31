
// go:build ignore

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/ptrace.h>
#include "events.h"

char LICENSE[] SEC("license") = "GPL";

SEC("uprobe/malloc")
int trace_malloc_enter(struct pt_regs *ctx)
{
    __u64 size = PT_REGS_PARM1(ctx);
    __u64 tid = bpf_get_current_pid_tgid();
    bpf_map_update_elem(&temp_size, &tid, &size, BPF_ANY);
    bpf_printk("malloc: tid=%llu size=%llu", tid, size);
    return 0;
}

SEC("uretprobe/malloc")
int trace_malloc_ret(struct pt_regs *ctx)
{
    __u64 tid = bpf_get_current_pid_tgid();
    __u64 *size_ptr = bpf_map_lookup_elem(&temp_size, &tid);
    if (!size_ptr)
        return 0;

    __u64 ptr = PT_REGS_RC(ctx);
    bpf_map_delete_elem(&temp_size, &tid);

    if (ptr == 0)
        return 0;

    struct alloc_info_t info = {
        .size = *size_ptr,
        .timestamp_ns = bpf_ktime_get_ns(),
    };
    bpf_map_update_elem(&allocs, &ptr, &info, BPF_ANY);

    struct event_t *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;
    bpf_printk("malloc ret: tid=%llu ptr=0x%llx size=%llu", tid, ptr, *size_ptr);
    e->pid = tid >> 32;
    e->type = EVENT_MALLOC;
    e->size = *size_ptr;
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uprobe/free")
int trace_free(struct pt_regs *ctx)
{
    __u64 ptr = PT_REGS_PARM1(ctx);
    if (ptr == 0)
        return 0;

    bpf_map_delete_elem(&allocs, &ptr);
    struct event_t *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;
    bpf_printk("free: ptr=0x%llx", ptr);
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    e->pid = pid;
    e->type = EVENT_FREE;
    e->ptr = ptr;
    bpf_ringbuf_submit(e, 0);
    return 0;
}