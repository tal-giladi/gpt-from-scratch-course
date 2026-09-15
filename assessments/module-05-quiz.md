# Module 05 quiz - Measuring

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** `evaluate_bpb` sums nats and sums bytes across all evaluation batches, then divides
once. Give a concrete two-batch example where averaging per-batch bpb instead would give a
different - and wrong - answer.

**2.** An agent lowers `EVAL_TOKENS` from 21M to 32,768 and reports a `val_bpb` improvement.
What is wrong, and what is the *one* circumstance under which changing `EVAL_TOKENS` is
legitimate?

**3.** Special tokens are excluded from both sums. Why would including them make the metric
easier to game, and in which direction?

**4.** Your first CPU run reports `mfu: 5.7%` and `tok/sec: 255`. After a one-line config change
the same machine reports `mfu: 34.4%` and `tok/sec: 1,544`, with an identical model. What kind
of change could that have been - a better model, better hardware, or a better use of the same
hardware - and what does the first MFU number tell you that `tok/sec` alone could not?

**5.** `estimate_flops` excludes `wte` and the value embeddings but not `lm_head`, even
though all three are `(vocab, n_embd)`-shaped. Justify the asymmetry.

**6.** For the 4-layer CPU training model at `sequence_len = 256`, the attention term is about 6%
of FLOPs per token; for an `n_embd = 512` model at 32,768 it is about 85%. Show the reasoning,
and say what that implies about which optimisation matters at which context length.

---

## Answers

**1.** Batch A: 1,000 nats over 1,000 bytes. Batch B: 100 nats over 10 bytes. Summed:
`1100 / (ln2 * 1010) = 1.57` bpb. Averaged per batch: `(1.44 + 14.43) / 2 = 7.9` bpb. The
average lets a tiny batch dominate the score. Real evaluation batches differ in byte count
because packing puts different tokens in them, so this is not a contrived case - it is a
smaller version of the same error.

**2.** The metric moved, not the model. A smaller eval set is noisier and will differ from
the 21M-token number in both directions; reporting the difference as an improvement is
measuring the ruler. The legitimate circumstance is a **whole-fork, declared** change - as
the CPU port did - where *every* run in that fork uses the new value, results are internally
comparable, and the README says loudly that they are not comparable to upstream's.

**3.** Special tokens are inserted by the pipeline, not by the data, and BOS appears at the
start of every document - a very predictable position. A model would learn to predict them
almost perfectly, so including them would add a large number of near-zero-loss terms to the
numerator while adding **nothing** to the denominator (their byte length is 0). The score
would fall - look better - purely as a function of how many documents got packed into the
eval set.

**4.** A better use of the same hardware. The model and the machine are unchanged - same FLOPs per
token, same peak - so the only thing that can move MFU six-fold is how much of the machine the run
actually uses. (In this fork it was exactly that: PyTorch's default 16 threads spent most of each
step waking and waiting for threads, and pinning 8 fixed it - lesson 16.) `tok/sec = 255` on its
own only says "slow"; it cannot distinguish slow hardware from wasted hardware. `mfu = 5.7%` says
the machine was idle 94% of the time, so the fix had to be in the software, not a bigger machine.
If MFU had already been ~35% at 255 tok/s, the right advice would have been a faster machine or a
smaller model.

**5.** The exclusion is not about shape, it is about arithmetic. `wte` and the value
embeddings are used as **lookups**: indexing row `i` of a table does no multiply-accumulate
work proportional to `n_embd`, so the `6N` rule does not apply to them. `lm_head` is used as
a **matmul** against the full vocabulary at every position - real arithmetic, `6 * V *
n_embd` FLOPs per token. At small scale that is a large share of the matmul term: 73% for the
course's toy model, 40% for the 4-layer CPU training model.

**6.** The parameter term is `6 * (params - embeddings)` and does not depend on sequence
length at all. The attention term is `12 * n_head * head_dim * min(window, t)` per layer -
**linear in `t` per token**, which means quadratic in `t` for a whole sequence. Scale `t` by
128x and the attention term scales by 128x while the parameter term is unchanged. Implication:
at short context, optimise the matmuls (fusion, dtype, `lm_head` size); at long context,
optimise attention (FlashAttention, sliding windows, sparsity) - the same change can be
irrelevant at one length and decisive at another.
