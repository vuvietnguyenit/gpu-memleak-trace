
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

// clang-format off
#include <asm-generic/int-ll64.h>
#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <linux/types.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <cuda.h>
// clang-format on

char LICENSE[] SEC("license") = "GPL";

struct alloc_info_t {
  __u64 size;
  CUdeviceptr dptr_addr;
};

enum event_type {
  EVENT_MALLOC = 0,
  EVENT_FREE = 1,
};

struct alloc_event {
  __u32 pid;
  __u32 tid; // Thread ID
  __u32 uid;
  __s32 stack_id; // user stack id
  __u64 size;
  __u64 dptr;
  char comm[16]; // <- command of proc
  enum event_type event_type;
  int retval;
};

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __type(key, __u64); // pid_tgid
  __type(value, struct alloc_info_t);
  __uint(max_entries, 1024);
} inflight SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_STACK_TRACE);
  __uint(key_size, sizeof(__u32));
  __uint(value_size, 127 * sizeof(__u64));
  __uint(max_entries, 8192);
} stack_traces SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 25); // 16 MB buffer
} events SEC(".maps");

SEC("uprobe/cuMemAlloc")
int trace_cu_mem_alloc_entry(struct pt_regs *ctx) {
  __u64 pid_tgid = bpf_get_current_pid_tgid();
  CUdeviceptr *dptr_ptr = (CUdeviceptr *)PT_REGS_PARM1(ctx);
  __u64 size = PT_REGS_PARM2(ctx);
  if (bpf_map_lookup_elem(&inflight, &pid_tgid)) {
    bpf_printk("cuMemAlloc entry: pid_tgid=%llu already has inflight alloc",
               pid_tgid);
    return 0;
  }
  bpf_map_update_elem(
      &inflight, &pid_tgid,
      &(struct alloc_info_t){.size = size, .dptr_addr = (CUdeviceptr)dptr_ptr},
      BPF_ANY);
  return 0;
}

SEC("uretprobe/cuMemAlloc")
int trace_malloc_return(struct pt_regs *ctx) {
  __u64 pid_tgid = bpf_get_current_pid_tgid();
  __u32 pid = pid_tgid >> 32;
  __u32 tid = (__u32)pid_tgid;
  __u64 uid_gid = bpf_get_current_uid_gid();
  __u32 uid = (__u32)uid_gid;
  __u32 sid = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);

  int retval = PT_REGS_RC(ctx);
  struct alloc_info_t *info;
  struct alloc_event *event;
  info = bpf_map_lookup_elem(&inflight, &pid_tgid);
  if (!info) {
    bpf_printk("cuMemAlloc return: pid_tgid=%llu no inflight alloc", pid_tgid);
    return 0;
  }
  // Read value from *dptr
  CUdeviceptr real_dptr;
  bpf_probe_read_user(&real_dptr, sizeof(real_dptr),
                      (const void *)info->dptr_addr);
  bpf_printk("cuMemAlloc return: pid=%u, sid=%d size=%llu, dptr_addr=0x%llx",
             pid, sid, info->size, real_dptr);
  event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
  if (!event)
    goto cleanup;

  event->pid = pid;
  event->tid = tid;
  event->uid = uid;
  event->stack_id = sid;
  event->size = info->size;
  event->dptr = real_dptr; // CUdeviceptr dptr
  event->retval = PT_REGS_RC(ctx);
  event->event_type = EVENT_MALLOC;
  bpf_get_current_comm(&event->comm, sizeof(event->comm));
  bpf_ringbuf_submit(event, 0);

cleanup:
  bpf_map_delete_elem(&inflight, &pid_tgid);
  return 0;
}

SEC("uprobe/cuMemFree")
int trace_cuMemFree(struct pt_regs *ctx) {
  struct alloc_event *e;
  __u64 pid_tgid = bpf_get_current_pid_tgid();
  __u64 uid_gid = bpf_get_current_uid_gid();

  __u64 dptr = PT_REGS_PARM1(ctx);
  __u32 pid = pid_tgid >> 32;
  __u32 tid = (__u32)pid_tgid;
  __u32 uid = (__u32)uid_gid;

  bpf_printk("cuMemFree: pid=%d tid=%d uid=%d dptr_addr=0x%llx\n", pid, tid,
             uid, dptr);

  e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
  if (!e) {
    return 0;
  }
  e->pid = pid;
  e->dptr = dptr; // CUdeviceptr dptr
  e->event_type = EVENT_FREE;
  e->tid = tid;
  e->uid = uid;
  e->retval = -1;
  bpf_get_current_comm(&e->comm, sizeof(e->comm));
  bpf_ringbuf_submit(e, 0);
  return 0;
}