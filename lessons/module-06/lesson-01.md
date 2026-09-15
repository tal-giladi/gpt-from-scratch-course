# 16 - Why it is slow: dtypes, kernels, and torch.compile

A GPU run of this model does ~500 million tokens in five minutes (lesson 15). The CPU fork does
about a million in ten. Some of that gap is simply "a CPU is not a GPU". But a surprising
amount is not - it is specific, measurable things, and understanding them is understanding what a
training step actually spends its time on.

This lesson's numbers were measured inside the lab container on the machine the course was
written on (an Intel Core Ultra 7 laptop chip, 16 logical CPUs). **Yours will differ** - the point
is the pattern, and the Do-this section has you measure your own.

## FLOPs per second is not the whole story

Lesson 14 measured MFU from FLOPs. It is tempting to think time is just FLOPs divided by speed.
It is not, because operations are limited by different things:

- **Compute-bound**: a big matrix multiply. Almost all the time goes into arithmetic, and it
  gets faster with a faster processor.
- **Memory-bound**: operations like the residual add, RMS norm, ReLU² and rotary. Each does
  almost no arithmetic per number - one add, one multiply - but has to read a whole activation
  tensor out of memory and write a whole new one back. Their cost is **bytes moved**, not FLOPs.
- **Overhead**: work that is neither. Python calling into PyTorch, PyTorch choosing a kernel,
  starting it, waking up worker threads. It costs roughly the same for a tiny tensor as for a big
  one.

A transformer step is a few big multiplies surrounded by dozens of memory-bound and overhead-heavy
small operations. Here is how that shows up.

## 1. Small multiplies waste most of their time

Timing a single square matrix multiply at different sizes (8 threads, float32):

    size         FLOPs per call     speed
    64 x 64            0.5 M         71 GFLOP/s
    256 x 256         34   M        189 GFLOP/s
    1024 x 1024      2.1   G        279 GFLOP/s

Same processor, same operation, four times faster per FLOP at the large size. A `64 x 64` multiply
finishes its arithmetic in well under a millisecond, and the fixed cost of launching it is a big
share of that. A `1024 x 1024` multiply runs long enough for the launch cost to disappear into the
noise. GPUs have exactly the same problem, much worse - which is why GPU training uses batches as
big as memory allows.

At this model's size the matmuls are medium: the stream projection for one micro-batch is
`(2048 x 256) @ (256 x 256)`, about 0.27 GFLOP, a few milliseconds. The small operations around it
are where time leaks.

## 2. Too many threads made every step 8x slower

PyTorch on CPU splits big operations across worker threads, and by default it uses one thread per
logical CPU - 16 here. That sounds like the fastest choice. Measured, for one forward+backward pass
of the real 4-layer training model (`B = 8`, `T = 256`):

    threads     time per forward+backward
       1              1.13 s
       4              0.54 s
       8              0.46 s
      12              0.63 s
      16              3.98 s        <- the default

**The default is eight times slower than 8 threads.** And it is not the big multiplies that suffer:
at 16 threads a `64 x 64` matmul dropped to 0.1 GFLOP/s, and a plain `x + x` on one 2 MB activation
took ~7 ms instead of ~0.05 ms.

Why? Every operation hands its work to all 16 threads and waits for the slowest one. Laptop chips
like this have a mix of fast and slow cores, and 16 logical CPUs include hyper-threads sharing the
same physical core; inside Docker on Windows the operating system is also juggling a virtual
machine. For big multiplies the extra threads still roughly pay for themselves. For the many tiny
operations in a transformer, the cost of starting 16 threads and waiting for all of them swamps the
work, hundreds of times over.

This was the single biggest reason this fork's first CPU training runs took ~8 seconds per step.
Both `docker-compose.yml` files now set `OMP_NUM_THREADS: "8"`, and the same 2-minute run went from
16 steps and `val_bpb` 2.61 to ~100 steps and 2.28 - with no change to the model. But the
right thread count is a measurement, not a guess - which is exactly what the Do-this section has you
do, and why the lessons quote numbers from the 8-thread runs.

## 3. What `torch.compile` would fix

In normal ("eager") mode, PyTorch runs each operation the moment Python reaches it. RMS norm then a
multiply then an add is three separate kernels, each reading the whole tensor from memory and writing
a whole new tensor:

    y = rms_norm(x)       # pass 1 over memory
    y = y * w             # pass 2
    y = y + z             # pass 3

`torch.compile` traces the model into a graph first, then generates **fused** kernels: the three
steps become one loop that reads each number once and writes the result once. For memory-bound
operations that means roughly a third of the bytes moved, and fewer kernel launches - often a 2-3x
speedup on those operations. That is why upstream compiles both the model and the two optimizer
steps.

The fork turns it off on CPU, for two reasons: the code it generates needs a C++ compiler that the
slim Docker image does not have, and compiling takes tens of seconds before the first step - a large
bite out of a short budget. On a GPU the trade is clearly worth it; here it is not. `AR_COMPILE=1`
turns it back on if you install a compiler and want to measure the difference yourself.

This also explains a strange-looking detail in the optimizer:

    self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

Why store a learning rate as a 0-dimensional tensor instead of a plain Python number? A compiled
kernel is specialised to the Python values it was traced with. If `lr` were a float that changes
every step (and it does - lesson 12), `torch.compile` would regenerate the kernel every step. A 0-D
tensor whose *contents* change but whose *type* does not gets compiled once. That is what the code
comment about avoiding recompilation is protecting.

## 4. bfloat16: a GPU win, not a CPU one

A float32 number takes 32 bits; **bfloat16** takes 16. bf16 keeps float32's huge *range* (it can
still hold `1e30`) but gives up most of its *precision* - it only has 7 bits for the digits after
the leading one:

    value      stored as float32     stored as bfloat16
    1.001      1.001                 1.0          the 0.001 is lost
    1.005      1.005                 1.0078       nearest step is 1/128 away
    3.14159    3.14159               3.140625

On an H100 that is a good trade. The GPU has dedicated circuits for 16-bit multiplies that are
roughly twice as fast, and every tensor is half the bytes to move - so upstream keeps the
embeddings, the rotary tables and Muon's orthogonalisation in bf16 and wraps the forward pass in
`autocast`. Training tolerates the lost precision in most places (and lesson 09's `.float()` upcast
protects the one place it does not).

A CPU only gains if it has instructions for 16-bit float maths (`AVX512-BF16` or `AMX` on some
Intel server chips). This machine has neither. Measured at 8 threads:

    size          float32         bfloat16
    64 x 64       71 GFLOP/s      26 GFLOP/s     2.7x slower
    256 x 256    189              192            the same
    1024 x 1024  279              297            the same

That is a single matmul. A whole training step is far worse: run with `AR_BF16=1` (autocast plus
bf16 embeddings, rotary tables and Muon), one step of the 4-layer model took about **190 seconds**
instead of about 1.2 - over 150 times slower. Since the big matmuls above ran at about the same speed in
either format, nearly all of that cost is in the operations around them.

No reliable win, sometimes a large loss, and always less precision. So the port introduced
`LOW_DTYPE`: `bfloat16` when `USE_BF16` (the GPU default), `float32` on CPU, and it skips `autocast`
entirely on CPU.

The general lesson: **a number format is not "more efficient" in the abstract. It is efficient if the
hardware has circuits for it.**

## 5. What FlashAttention was doing

That one is big enough to be its own lesson - next.

## Do this

1. In the lab shell, measure your own machine:

       bash lab/lab.sh shell

       import time, torch
       import os
       torch.get_num_threads(), os.cpu_count()    # what the lab pins (8), and how many CPUs you have

       def gflops(n, dtype=torch.float32, iters=20):
           a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
           a @ b                                  # warm-up
           t0 = time.perf_counter()
           for _ in range(iters): a @ b
           return 2 * n**3 / ((time.perf_counter() - t0) / iters) / 1e9

       [round(gflops(n), 1) for n in (64, 256, 1024)]
       [round(gflops(n, torch.bfloat16), 1) for n in (64, 256, 1024)]

2. Find your best thread count on the real model:

       from lib.common import build_model, toy_config, get_batch
       m = build_model(toy_config(n_layer=4, n_embd=256, sequence_len=256))
       x, y = get_batch(B=8, T=256)
       for th in (1, 2, 4, 8, os.cpu_count()):   # os.cpu_count() is PyTorch's own default
           torch.set_num_threads(th)
           m(x, y).backward(); m.zero_grad()      # warm-up at this thread count
           t0 = time.perf_counter(); m(x, y).backward(); m.zero_grad()
           print(th, round(time.perf_counter() - t0, 2), "s")

   If a smaller number wins by a lot, that is a real speedup for every training run on your
   machine - worth writing down for the capstone.

3. Fill in `lab/exercises/lesson_16.py`: `benchmark_matmul(n, dtype, iters)`.

4. Grade it:

       bash lab/lab.sh check 16

## Hints

- A square `(n, n) @ (n, n)` matmul is `2 * n**3` FLOPs. Each of the `n * n` output numbers is a
  sum of `n` products: `n` multiplies and about `n` adds, so `2 * n` FLOPs each.
- **Warm up first.** The first call allocates memory and picks a kernel; timing it makes every
  number wrong. Run the multiply once outside the timed loop.
- Time `iters` iterations and divide, rather than timing one - a single small matmul can be shorter
  than the clock's resolution.
- Use `time.perf_counter()`, which is made for measuring short intervals, rather than `time.time()`.
- Return both the seconds per iteration and the GFLOP/s, so the check can verify you computed one
  from the other.
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

A training step is not one big matmul. It is a few medium matmuls surrounded by many small,
memory-bound operations, and each of those pays a fixed launch cost. That is why small operations
are slow per FLOP, why too many threads can make a step several times slower, and why
`torch.compile` - which fuses those small operations - helps so much on a GPU. bfloat16 halves the
bytes and trades away precision, which pays only on hardware with circuits for it. The port gave up
compile and bf16 on purpose; the thread count is something to measure on your own machine. Next:
the one kernel the port could not simply switch off.
