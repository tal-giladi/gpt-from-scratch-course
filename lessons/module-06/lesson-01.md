# 16 - Why it is slow: dtypes, kernels, and torch.compile

Upstream trains this model on an H100 at 40-50% MFU. The CPU fork gets about 4%. Some of
that is simply "a CPU is not a GPU", but most of it is not - it is three specific things the
port gave up, and understanding them is understanding what a training stack actually spends
its time on.

## 1. The arithmetic is not the bottleneck

Measure your machine's peak first. A plain `4096x1024 @ 1024x1024` float32 matmul inside the
container reaches about **30 GFLOP/s** on 16 threads. The model, doing the same arithmetic
inside a real training step, reaches about **9 GFLOP/s** - a third of a peak that is itself
modest.

The gap is not FLOPs. It is everything around them:

- **Small matrices.** At `n_embd = 256`, a matmul is `(2048x256) @ (256x256)`. That is
  0.27 GFLOP - a few milliseconds of work with a fixed per-operation overhead that does not
  shrink with it. GPUs have the same problem, which is why batch size matters so much.
- **Memory traffic.** RMS norm, the residual adds, ReLU², rotary - each reads a whole
  activation tensor from RAM, does one cheap thing, and writes it back. These are
  **memory-bound**: their cost is bytes moved, not FLOPs. A transformer has a lot of them.
- **Eager mode.** Every one of those is a separate PyTorch operator dispatch, a separate
  kernel launch, a separate round trip to memory.

## 2. What `torch.compile` would fix

`torch.compile` traces the model into a graph and generates fused kernels. The win is
almost entirely **fusion**: `rms_norm -> multiply -> add` becomes one pass over memory
instead of three. On the memory-bound operations that is a 2-3x improvement, and it is why
upstream compiles both the model and the two optimizer steps.

The fork turns it off on CPU, for two honest reasons: the generated code needs a C++
toolchain that is not in the slim image, and the warm-up compilation costs tens of seconds -
against a 600-second budget where the first two steps are already untimed. On a five-minute
GPU run that trade is obviously worth it; on a ten-minute CPU run it is not. `AR_COMPILE=1`
re-enables it if you install a compiler and want to measure the difference yourself.

This is also why upstream goes to such lengths with those 0-D CPU tensors in the optimizer:

    self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

Passing a plain Python float that changes every step would make `torch.compile` re-trace the
kernel on every step. Passing a 0-D tensor whose *value* changes but whose *type* does not
means it compiles once. That is what the comment "avoid recompilation when values change"
is protecting.

## 3. bf16 is a GPU win and a CPU trap

Upstream keeps the embeddings, the rotary tables and Muon's orthogonalisation in bfloat16,
and wraps the forward pass in `autocast`. On an H100 that roughly doubles throughput,
because the tensor cores are twice as fast in bf16 and every tensor moved is half the bytes.

On most x86 CPUs there is no bf16 matmul instruction at all (you need AVX512-BF16 or AMX).
PyTorch emulates it: convert to fp32, multiply, convert back - *slower* than just using
fp32, plus you lose precision. So the port introduced `LOW_DTYPE`, which is `bfloat16` on
CUDA and `float32` on CPU, and dropped `autocast` on CPU entirely.

The lesson generalises: **a dtype is not "more efficient" in the abstract.** It is efficient
if the hardware has an instruction for it.

## 4. What FlashAttention was doing

That one is big enough to be its own lesson - next.

## Do this

1. In the lab shell, measure your own machine:

       bash lab/lab.sh shell

       import time, torch
       a, b = torch.randn(4096, 1024), torch.randn(1024, 1024)
       for _ in range(2):
           t0 = time.time()
           for _ in range(5): c = a @ b
           dt = (time.time() - t0) / 5
           print(f"{2*4096*1024*1024/dt/1e9:.1f} GFLOP/s")

       torch.get_num_threads()
       torch.__config__.parallel_info().splitlines()[:5]

   Run it for `bfloat16` too, and for a small matrix (`128x128`), and watch what happens.

2. Fill in `lab/exercises/lesson_16.py`: `benchmark_matmul(n, dtype, iters)`.

3. Grade it:

       bash lab/lab.sh check 16

## Hints

- A square `(n, n) @ (n, n)` matmul is `2 * n**3` FLOPs: one multiply and one add per
  element of the sum, `n` of them per output element, `n*n` outputs.
- **Warm up first.** The first call allocates buffers and picks a kernel; timing it makes
  every number wrong. Run the multiply once outside the timed loop.
- Time `iters` iterations and divide, rather than timing one - a single small matmul is
  shorter than the clock's noise.
- Return both the seconds per iteration and the GFLOP/s, so the check can verify you
  computed one from the other.
- `torch.randn(n, n, dtype=dtype)` works for `float32` and `bfloat16` alike.

## Solution

    import time
    import torch

    def benchmark_matmul(n, dtype=torch.float32, iters=5):
        a = torch.randn(n, n, dtype=dtype)
        b = torch.randn(n, n, dtype=dtype)
        _ = a @ b                              # warm-up, not timed
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

## Summary

A training step is not one big matmul; it is a long chain of small, memory-bound operations
around some medium-sized matmuls, and that chain is what `torch.compile` fuses away. bf16
helps only where the hardware has an instruction for it. Both are things the CPU port gave
up on purpose, and both are measurable on your own machine in five lines. Next: the one
kernel the port could not simply switch off.
