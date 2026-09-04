# Module 06 quiz - Speed, and reading a port

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** Your machine does 30 GFLOP/s on a large fp32 matmul, but the training step achieves
9 GFLOP/s of useful arithmetic. Name three distinct causes of the gap and say which one
`torch.compile` addresses.

**2.** A colleague sets `AR_BF16=1` on your CPU to "speed things up." Predict the result and
say how you would settle the argument in 30 seconds.

**3.** Why does `MuonAdamW` keep learning rates in 0-D tensors on the CPU rather than as
Python floats? What breaks if you use floats, and does that breakage apply to the CPU fork?

**4.** Explain, without using the word "approximate", why FlashAttention is faster than the
naive implementation.

**5.** The port replaced FA3 with SDPA plus an explicit mask. For the *full-causal* case it
passes `is_causal=True` instead of a mask. Why does that distinction matter, and what does it
cost in the windowed case?

**6.** Of the five things the CPU port gave up - FlashAttention, bf16, `torch.compile`,
512K-token batches, upstream's eval budget - which were forced and which were judgement
calls? For one judgement call, describe the experiment that would settle it.

---

## Answers

**1.** (a) **Small matrices**: at `n_embd = 256` each matmul is a fraction of a GFLOP, and
the fixed per-operation overhead does not shrink with it. (b) **Memory-bound elementwise
ops**: RMS norm, the residual adds, ReLU², rotary - each reads and writes a whole activation
tensor to do very little arithmetic. (c) **Eager-mode dispatch**: every one of those is a
separate operator call and kernel launch. `torch.compile` addresses (b) and (c) together, by
fusing chains of elementwise ops into single passes over memory - it does not make the
matmuls faster.

**2.** Prediction: it will be *slower*, because most x86 CPUs have no bf16 matmul instruction
(you need AVX512-BF16 or AMX) and PyTorch emulates it by converting to fp32 and back. Settle
it by running the lesson-16 benchmark for both dtypes: on this machine bf16 measures about
0.78x of fp32. Measure, do not argue.

**3.** Because those functions are wrapped in `@torch.compile(fullgraph=True)`. A Python
float is baked into the traced graph as a constant, so a value that changes every step (the
learning rate follows a schedule) would trigger a **recompilation every step**, which is
catastrophically slow. A 0-D tensor is a graph *input*: its value can change freely without
retracing. It does not matter in the CPU fork, where compilation is off - but the code has to
stay correct for both, which is why the trick stays in.

**4.** It computes the same softmax-weighted sum without ever writing the `T x T` score
matrix to memory. It processes blocks of queries and keys in on-chip SRAM, maintaining a
running maximum and a running normalisation term so that partial results can be combined
exactly as more blocks arrive. The FLOP count is essentially unchanged; the **memory traffic**
drops from `O(T^2)` to `O(T)`, and memory traffic was the bottleneck.

**5.** `is_causal=True` is a flag, so SDPA can dispatch to a fused implementation that knows
the mask's structure and never materialises it - it simply does not compute the upper
triangle. An explicit `attn_mask` is opaque data, so SDPA falls back to its general math
kernel, which builds the full `(B, H, T, T)` score tensor and applies the mask to it. So in
the windowed case the port pays exactly the `O(T^2)` memory cost FlashAttention exists to
avoid - which means a sliding window, a pure saving on GPU, becomes a cost on CPU.

**6.** Forced: FlashAttention (no CPU build exists) and the eval budget (21M eval tokens
would take longer than the training run). Judgement calls: bf16 (measured, and the right
call), `torch.compile` (a toolchain in the image would change the answer), and batch size
(2048 tokens per step is a very noisy gradient - it was chosen to get more optimizer steps
into the budget). Settling the `torch.compile` one: add `g++` to the image, run the same
protocol twice with `AR_COMPILE=0` and `AR_COMPILE=1`, equal time budgets, and compare
`val_bpb` - the compile warm-up is paid out of the budget, so the comparison is fair by
construction.
