# Module 06 quiz - Speed, and reading a port

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** Your machine does ~150 GFLOP/s on a large fp32 matmul, but a training step achieves only
~9 GFLOP/s of useful arithmetic. After setting `OMP_NUM_THREADS=8` it achieves ~52. Name the
distinct causes of the remaining gap to 150, say which one the thread change addressed, and
which one `torch.compile` would address.

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

**1.** (a) **Thread overhead**: every operation is split across all worker threads and waits for
the slowest. With 16 threads on this laptop that cost swamped the many small operations - that is
the 9 -> 52 jump, and it is what `OMP_NUM_THREADS` addressed. (b) **Small matrices**: at
`n_embd = 256` each matmul is a fraction of a GFLOP, and its fixed launch cost does not shrink
with it. (c) **Memory-bound elementwise ops**: RMS norm, the residual adds, ReLU², rotary - each
reads and writes a whole activation tensor to do very little arithmetic. (d) **Eager-mode
dispatch**: every one of those is a separate operator call and kernel launch. `torch.compile`
addresses (c) and (d) together, by fusing chains of elementwise ops into single passes over
memory - it does not make the matmuls faster.

**2.** Prediction: no faster, and possibly slower. A CPU only gains from bf16 if it has
instructions for 16-bit float maths (AVX512-BF16 or AMX, on some Intel server chips); this
laptop has neither. You also lose precision (`1.001` is stored as `1.0`). Settle it by running
the lesson-16 benchmark for both dtypes. Measured here at 8 threads: 2.7x *slower* on a 64x64
matmul, and the same speed at 256x256 and 1024x1024. No win anywhere, a big loss on small
matrices. Measure, do not argue.

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

**5.** `is_causal=True` is a flag: SDPA knows the mask's shape without being handed one, and its
kernel needs no mask tensor at all. An explicit `attn_mask` is data the kernel has to carry
around. On PyTorch 2.9.1 on CPU, both calls still use the same CPU flash-style kernel
(`_scaled_dot_product_flash_attention_for_cpu`), so neither builds the full `(B, H, T, T)` score
tensor - but the masked call cost about ten times the extra memory of the causal one (179 MB vs
16 MB at `T = 4096`), and it computes every pair and then masks, so a shorter window saves no
compute (17 ms vs 14 ms at `T = 256`). On a GPU, FlashAttention-3's `window_size` skips
out-of-window tiles and the window is a pure saving; on this port it is a small cost. (Older
PyTorch versions fell back to the plain math kernel whenever a mask was passed, and materialised
the full score tensor - the naive version measured 1,072 MB.)

**6.** Forced: FlashAttention (no CPU build exists); the eval budget (21M eval tokens would take
longer than the training run); and 524K-token batches (at ~1,650 tok/s one such step would take
over five minutes, so a 2-minute run could not finish a single step). Judgement calls: bf16
(measured, and the right call), `torch.compile` (a toolchain in the image would change the
answer), and the *specific* small batch that replaced 524K - 2,048 tokens is a very noisy
gradient, chosen to get more optimizer steps into the budget. Settling the `torch.compile` one:
add `g++` to the image, run the same protocol twice with `AR_COMPILE=0` and `AR_COMPILE=1`, equal
time budgets, and compare `val_bpb` - the compile warm-up is paid out of the budget, so the
comparison is fair by construction.
