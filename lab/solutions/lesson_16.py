# Lesson 16 - reference solution.

import time

import torch


def benchmark_matmul(n, dtype=torch.float32, iters=5):
    a = torch.randn(n, n, dtype=dtype)
    b = torch.randn(n, n, dtype=dtype)
    _ = a @ b  # warm-up, not timed
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    seconds = (time.perf_counter() - t0) / iters
    flops = 2 * n ** 3
    return {
        "n": n,
        "dtype": str(dtype),
        "seconds": seconds,
        "gflops": flops / seconds / 1e9,
    }
