#!/usr/bin/env python3
"""
Roofline baseline: measure this machine's GPU and CPU memory bandwidth.

WHY THIS MATTERS
LLM decoding is memory-bound, not compute-bound. Producing one token requires
reading EVERY model weight from memory exactly once, and each weight is used in
just one multiply-accumulate (at batch size 1):

    arithmetic intensity = 2 FLOP / 1 byte     -> extremely low

For comparison, a modern GPU can sustain hundreds of TFLOPS but only reads a
fraction of a TB/s from memory. The compute units starve waiting on memory.

The practical consequence:

    decode_speed (tok/s) ~= memory_bandwidth (GB/s) / model_size (GB)

This script measures that bandwidth term. Every performance prediction we make
is anchored to these numbers, so we measure them instead of guessing.
"""

import time

import numpy as np
import torch


def bench_gpu_copy(dtype=torch.float16, size_mb=1024, iters=50):
    """GPU VRAM bandwidth via a large device-to-device copy (1 read + 1 write)."""
    n = size_mb * 1024 * 1024 // dtype.itemsize
    src = torch.empty(n, dtype=dtype, device="cuda")
    dst = torch.empty(n, dtype=dtype, device="cuda")
    src.uniform_()

    for _ in range(5):  # warmup: let clocks boost and caches settle
        dst.copy_(src)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_moved = src.nbytes * 2 * iters  # a copy touches memory twice
    return bytes_moved / elapsed / 1e9


def bench_gpu_read(dtype=torch.float16, size_mb=1024, iters=50):
    """Read-only bandwidth. Closer to decoding, which reads weights but rarely writes."""
    n = size_mb * 1024 * 1024 // dtype.itemsize
    buf = torch.empty(n, dtype=dtype, device="cuda").uniform_()

    for _ in range(5):
        buf.sum()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        buf.sum()
    torch.cuda.synchronize()
    return buf.nbytes * iters / (time.perf_counter() - start) / 1e9


def bench_cpu_copy(size_mb=2048, iters=10):
    """CPU DRAM bandwidth via a plain memcpy, staying clear of threaded BLAS paths."""
    n = size_mb * 1024 * 1024 // 2
    src = np.ones(n, dtype=np.float16)
    dst = np.empty_like(src)

    np.copyto(dst, src)  # warmup
    start = time.perf_counter()
    for _ in range(iters):
        np.copyto(dst, src)
    elapsed = time.perf_counter() - start

    return src.nbytes * 2 * iters / elapsed / 1e9


def bench_pcie(size_mb=512, iters=20):
    """PCIe bandwidth, the host<->device link used by hybrid offload.

    llama.cpp uploads weights once at load time, but per-layer activations cross
    this link on every token. Small compared to weight traffic, but not zero.
    """
    n = size_mb * 1024 * 1024 // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device="cuda")

    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    h2d = host.nbytes * iters / (time.perf_counter() - start) / 1e9

    start = time.perf_counter()
    for _ in range(iters):
        host.copy_(dev, non_blocking=True)
    torch.cuda.synchronize()
    d2h = host.nbytes * iters / (time.perf_counter() - start) / 1e9

    return h2d, d2h


# Candidate quantizations of Qwen3.8-27B, as shipped by unsloth/Qwen3.8-27B-GGUF.
QUANTS = [
    ("IQ2_XXS", 9.01),
    ("UD-IQ3_XXS", 11.91),
    ("UD-Q3_K_XL", 13.44),
    ("IQ4_XS", 15.71),
    ("Q4_K_M", 17.11),
]

# VRAM actually available to weights after the driver, display server and CUDA
# context take their cut. Measured with `llama-cli --list-devices`.
USABLE_VRAM_GB = 7.3


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU         : {props.name}")
    print(f"SM count    : {props.multi_processor_count}")
    print(f"Compute cap : sm_{props.major}{props.minor}")
    print(f"VRAM        : {props.total_memory / 1e9:.2f} GB")
    print()

    gpu_copy = bench_gpu_copy()
    gpu_read = bench_gpu_read()
    cpu_bw = bench_cpu_copy()
    h2d, d2h = bench_pcie()

    print(f"GPU VRAM (copy R+W) : {gpu_copy:7.1f} GB/s")
    print(f"GPU VRAM (read-only): {gpu_read:7.1f} GB/s   <-- the one that governs decode")
    print(f"CPU DRAM (copy R+W) : {cpu_bw:7.1f} GB/s")
    print(f"PCIe host->device   : {h2d:7.1f} GB/s")
    print(f"PCIe device->host   : {d2h:7.1f} GB/s")
    print()
    print(f"GPU/CPU bandwidth ratio: {gpu_read / cpu_bw:.1f}x")
    print()

    print("=== PREDICTED DECODE SPEED UNDER HYBRID OFFLOAD ===")
    print(f"(assuming {USABLE_VRAM_GB} GB of VRAM is available for weights)")
    print()
    header = f"{'quant':<14}{'size':>8}{'on GPU':>9}{'on CPU':>9}{'GPU ms':>9}{'CPU ms':>9}{'tok/s':>9}"
    print(header)
    print("-" * len(header))
    for name, size_gb in QUANTS:
        on_gpu = min(size_gb, USABLE_VRAM_GB)
        on_cpu = size_gb - on_gpu
        t_gpu = on_gpu / gpu_read * 1000
        t_cpu = on_cpu / cpu_bw * 1000
        tps = 1000.0 / (t_gpu + t_cpu)
        print(
            f"{name:<14}{size_gb:7.2f}G{on_gpu:8.1f}G{on_cpu:8.1f}G"
            f"{t_gpu:9.1f}{t_cpu:9.1f}{tps:9.1f}"
        )
    print()
    print("NOTE: these are upper bounds. Real llama.cpp numbers land lower because")
    print("      of kernel launch overhead, synchronization and dequantization cost.")
    print("      The point is to see how close the measured throughput gets to this")
    print("      ceiling -- that gap is what optimization work has to close.")


if __name__ == "__main__":
    main()
