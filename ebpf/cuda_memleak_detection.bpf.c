
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
  __s32 device;
  __u32 uid;
  __s32 stack_id; // user stack id
  __u64 size;
  __u64 dptr;
  char comm[16]; // <- command of proc
  enum event_type event_type;
  int retval;
};

struct ctx_new_tmp {
  __u64 pctx_ptr; // user pointer to CUcontext variable
  __s32 dev;      // CUdevice (ordinal)
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
  __uint(type, BPF_MAP_TYPE_LRU_HASH);
  __uint(max_entries, 32768);
  __type(key, __u32); // TID
  __type(value, struct ctx_new_tmp);
} tmp_ctx SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_LRU_HASH);
  __uint(max_entries, 65536);
  __type(key, __u32);   // TID
  __type(value, __u64); // current CUcontext handle
} tid2ctx SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_LRU_HASH);
  __uint(max_entries, 65536);
  __type(key, __u64);   // CUcontext handle
  __type(value, __s32); // CUdevice (GPU ordinal)
} ctx2dev SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 25); // 16 MB buffer
} events SEC(".maps");

static __always_inline __u32 get_tid(void) {
  return (__u32)bpf_get_current_pid_tgid();
}
static __always_inline __u32 get_pid(void) {
  return (__u32)(bpf_get_current_pid_tgid() >> 32);
}

// ----- Context management probes -----

// CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
SEC("uprobe/cuCtxCreate")
int BPF_KPROBE(up_cuCtxCreate, __u64 pctx, __u32 flags, __s32 dev) {
  __u32 tid = get_tid();
  bpf_printk("up_cuCtxCreate %d", tid);

  struct ctx_new_tmp tmp = {.pctx_ptr = pctx, .dev = dev};
  bpf_map_update_elem(&tmp_ctx, &tid, &tmp, BPF_ANY);
  return 0;
}

SEC("uretprobe/cuCtxCreate")
int BPF_KRETPROBE(ur_cuCtxCreate, long ret) {
  __u32 tid = get_tid();
  bpf_printk("ur_cuCtxCreate ret %d", tid);

  struct ctx_new_tmp *t = bpf_map_lookup_elem(&tmp_ctx, &tid);
  if (!t)
    return 0;

  // Read CUcontext value written to *pctx
  __u64 ctx_handle = 0;
  bpf_probe_read_user(&ctx_handle, sizeof(ctx_handle),
                      (const void *)t->pctx_ptr);

  if (ret == 0 && ctx_handle) {
    // Map ctx -> dev, and set current ctx for this thread
    bpf_map_update_elem(&ctx2dev, &ctx_handle, &t->dev, BPF_ANY);
    bpf_map_update_elem(&tid2ctx, &tid, &ctx_handle, BPF_ANY);
  }
  bpf_map_delete_elem(&tmp_ctx, &tid);
  return 0;
}

// CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
SEC("uprobe/cuDevicePrimaryCtxRetain")
int BPF_KPROBE(up_cuDevicePrimaryCtxRetain, __u64 pctx, __s32 dev) {
  __u32 tid = get_tid();
  bpf_printk("up_cuDevicePrimaryCtxRetain %d", tid);

  struct ctx_new_tmp tmp = {.pctx_ptr = pctx, .dev = dev};
  bpf_map_update_elem(&tmp_ctx, &tid, &tmp, BPF_ANY);
  return 0;
}

SEC("uretprobe/cuDevicePrimaryCtxRetain")
int BPF_KRETPROBE(ur_cuDevicePrimaryCtxRetain, long ret) {
  __u32 tid = get_tid();
  bpf_printk("ur_cuDevicePrimaryCtxRetain ret %d", tid);

  struct ctx_new_tmp *t = bpf_map_lookup_elem(&tmp_ctx, &tid);
  if (!t)
    return 0;

  __u64 ctx_handle = 0;
  bpf_probe_read_user(&ctx_handle, sizeof(ctx_handle), (void *)t->pctx_ptr);

  if (ret == 0 && ctx_handle) {
    bpf_map_update_elem(&ctx2dev, &ctx_handle, &t->dev, BPF_ANY);
    bpf_map_update_elem(&tid2ctx, &tid, &ctx_handle, BPF_ANY);
  }
  bpf_map_delete_elem(&tmp_ctx, &tid);
  return 0;
}

// CUresult cuCtxSetCurrent(CUcontext ctx);
SEC("uprobe/cuCtxSetCurrent")
int BPF_KPROBE(up_cuCtxSetCurrent, __u64 ctx_handle) {
  __u32 tid = get_tid();
  bpf_printk("cuCtxSetCurrent %d", tid);
  bpf_map_update_elem(&tid2ctx, &tid, &ctx_handle, BPF_ANY);
  return 0;
}

// CUresult cuCtxPushCurrent(CUcontext ctx);
SEC("uprobe/cuCtxPushCurrent")
int BPF_KPROBE(up_cuCtxPushCurrent, __u64 ctx_handle) {
  __u32 tid = get_tid();
  bpf_printk("up_cuCtxPushCurrent ret %d", tid);
  bpf_map_update_elem(&tid2ctx, &tid, &ctx_handle, BPF_ANY);
  return 0;
}

// CUresult cuCtxPopCurrent(CUcontext *pctx);
SEC("uretprobe/cuCtxPopCurrent")
int BPF_KRETPROBE(ur_cuCtxPopCurrent, long ret) {
  if (ret != 0)
    return 0;
  __u32 tid = get_tid();
  bpf_printk("ur_cuCtxPopCurrent ret %d", tid);
  __u64 pctx_ptr = PT_REGS_PARM1(ctx);
  if (!pctx_ptr)
    return 0;

  __u64 ctx_handle = 0;
  bpf_probe_read_user(&ctx_handle, sizeof(ctx_handle), (void *)pctx_ptr);
  if (ctx_handle) {
    bpf_map_update_elem(&tid2ctx, &tid, &ctx_handle, BPF_ANY);
  }
  return 0;
}

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
  if (retval != 0) {
    goto cleanup;
    return 0;
  }
  struct alloc_info_t *info;
  struct alloc_event *event;
  info = bpf_map_lookup_elem(&inflight, &pid_tgid);
  if (!info) {
    bpf_printk("cuMemAlloc return: pid_tgid=%llu no inflight alloc", pid_tgid);
    return 0;
  }
  __s32 dev = -1;
  __u64 *ctx_handle = bpf_map_lookup_elem(&tid2ctx, &tid);
  if (ctx_handle) {
    __s32 *pdev = bpf_map_lookup_elem(&ctx2dev, ctx_handle);
    if (pdev)
      dev = *pdev;
  }

  // Read value from *dptr
  CUdeviceptr real_dptr;
  bpf_probe_read_user(&real_dptr, sizeof(real_dptr),
                      (const void *)info->dptr_addr);
  bpf_printk("cuMemAlloc return: pid=%u, sid=%d, device=%d, size=%llu, "
             "dptr_addr=0x%llx",
             pid, sid, dev, info->size, real_dptr);
  event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
  if (!event)
    goto cleanup;

  event->pid = pid;
  event->tid = tid;
  event->device = dev;
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