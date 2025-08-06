
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

char LICENSE[] SEC("license") = "MIT";

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} pid_allocs SEC(".maps");

SEC("uretprobe/malloc")
int trace_malloc_ret(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    void *ret = (void *)PT_REGS_RC(ctx);
    if (!ret)
        return 0;

    __u64 init = 0;
    __u64 *count = bpf_map_lookup_elem(&pid_allocs, &pid);
    if (!count)
    {
        bpf_map_update_elem(&pid_allocs, &pid, &init, BPF_NOEXIST);
        count = bpf_map_lookup_elem(&pid_allocs, &pid);
        if (!count)
            return 0;
    }
    __sync_fetch_and_add(count, 1);
    bpf_printk("malloc: pid=%d, ret %p, count %llu", pid, ret, *count);
    return 0;
}
SEC("uprobe/free")
int trace_free(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    __u64 *count = bpf_map_lookup_elem(&pid_allocs, &pid);
    if (count && *count > 0)
    {
        __sync_fetch_and_sub(count, 1);
    }
    bpf_printk("free: pid=%d, count %llu", pid, count ? *count : 0);
    return 0;
}