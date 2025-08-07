from bcc import BPF
import ctypes

# Replace with actual libc path
libc_path = "/lib/x86_64-linux-gnu/libc.so.6"

bpf_text = """
#include <uapi/linux/ptrace.h>

struct alloc_info_t {
    u32 pid;
    u32 tid;
    u64 size;
};

BPF_HASH(allocs, u64, struct alloc_info_t);

int trace_malloc(struct pt_regs *ctx, size_t size) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = pid_tgid;

    bpf_trace_printk("malloc start: pid=%d, size=%lu\\n", pid, size);
    return 0;
}

int trace_malloc_ret(struct pt_regs *ctx) {
    u64 ptr = PT_REGS_RC(ctx);
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = pid_tgid;

    struct alloc_info_t info = {};
    info.pid = pid;
    info.tid = tid;
    info.size = 0;  // Cannot get size here unless stored in entry

    allocs.update(&ptr, &info);
    bpf_trace_printk("malloc ret: pid=%d, ptr=0x%lx\\n", pid, ptr);
    return 0;
}

int trace_free(struct pt_regs *ctx, void *ptr) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;

    struct alloc_info_t *info = allocs.lookup(&ptr);
    if (info != 0) {
        bpf_trace_printk("free: pid=%d, ptr=0x%lx\\n", pid, ptr);
        allocs.delete(&ptr);
    }
    return 0;
}
"""

b = BPF(text=bpf_text)
b.attach_uprobe(name=libc_path, sym="malloc", fn_name="trace_malloc")
b.attach_uretprobe(name=libc_path, sym="malloc", fn_name="trace_malloc_ret")
b.attach_uprobe(name=libc_path, sym="free", fn_name="trace_free")

print("Monitoring malloc/free... Ctrl-C to exit.\n")
b.trace_print()
