# Lesson 16 - Why it is slow: dtypes, kernels, and torch.compile
# Read lessons/module-06/lesson-01.md before filling this in.

import time

import torch


def benchmark_matmul(n, dtype=torch.float32, iters=5):
    """Measure this machine's throughput on an (n, n) @ (n, n) matmul.

    Returns {"n": int, "dtype": str, "seconds": float, "gflops": float}
    where seconds is the time for ONE iteration and gflops is derived from it.

    A square matmul is 2 * n**3 floating-point operations. Do one untimed
    warm-up multiply before the timed loop, or the first call's setup cost
    lands in your measurement.
    """
    # TODO: allocate, warm up, time `iters` iterations, convert to GFLOP/s.
    #one untimed:
    a, b = torch.randn(n, n, dtype = dtype), torch.randn(n, n, dtype= dtype)
    a @ b

    t0 = time.perf_counter()
    for _ in range(iters): 
        c = a @ b
    
    flops = 2 * n ** 3
    seconds = (time.perf_counter() - t0) / iters
    return {"n": n, "dtype": type, "seconds": seconds, "gflops": flops / seconds / 1e9 }
