# Module 06 - Speed, and reading a port

**Lessons 16-17.** Where the time actually goes, and how to read a fork for what it gave up.

## Why this module exists

The learner arrived here because they wanted to run a GPU-only repo on a CPU. That port is
the best available teaching material about performance: every difference between the two is
a decision with a stated reason, and several of them invert the usual advice.

## Learning objectives

After this module a learner can:

1. Measure their own machine's matmul throughput and compare it against what the model
   achieves, and explain the gap in terms of small matrices, memory-bound operations and
   eager-mode dispatch.
2. Explain what `torch.compile` does (fusion, primarily) and why the port turned it off,
   including the 0-D-tensor trick that exists to avoid recompilation.
3. Explain why bf16 is a large win on an H100 and a measurable loss on a typical x86 CPU,
   and back it with their own measurement.
4. Explain what FlashAttention computes and why it is fast - tiling and online softmax,
   not fewer FLOPs - and why that makes it a CUDA kernel rather than a formula.
5. Reproduce the sliding-window mask exactly, count its entries by formula, and compute the
   score-tensor memory an explicit mask forces SDPA to materialise.
6. Read a fork's change list and classify each item as *forced* or *a trade*.

## Dependencies

Modules 01-05. Lesson 17 depends on lesson 05's attention and lesson 14's FLOPs split.

## Misconceptions this module is written to break

- *"Lower precision is faster."* Only where the hardware has an instruction for it. The
  check measures bf16 at about 0.78x of fp32 on this machine.
- *"The matmuls are the bottleneck."* At this scale the memory-bound elementwise operations
  around them are a large share, which is exactly what fusion addresses.
- *"FlashAttention is an approximation."* It is exact. Online softmax is an algebraic
  rearrangement, not an approximation.
- *"Sliding windows are always a saving."* On a GPU with FA3, yes. In the CPU port they
  force an explicit mask and the general math kernel, which costs memory the full-causal
  path avoids.
- *"A port is a mechanical translation."* Two of this fork's five significant changes were
  forced; the other three were judgement calls, and a maintainer has to be able to tell
  which is which.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 16 | `benchmark_matmul` | you can measure throughput correctly - warm-up, repetition, `2n^3` - and you have your machine's real numbers for fp32 and bf16 |
| 17 | `window_mask`, `mask_memory_bytes` | your mask matches the port's at four shapes and matches a closed-form count, and you can compute the `O(T^2)` memory FlashAttention avoids |

## Time

About two hours, including running the measurements yourself.
