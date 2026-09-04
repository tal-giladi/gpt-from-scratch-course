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

**4.** Your run reports `mfu: 4.3%` and `tok/sec: 259`. Your colleague's run on a bigger
machine reports `mfu: 4.1%` and `tok/sec: 2,600`. Who has the better *implementation*, and
what would you tell each of them to work on?

**5.** `estimate_flops` excludes `wte` and the value embeddings but not `lm_head`, even
though all three are `(vocab, n_embd)`-shaped. Justify the asymmetry.

**6.** At `sequence_len = 256` the attention term is about 10% of FLOPs per token; at 32,768
it dominates. Show the reasoning, and say what that implies about which optimisation matters
at which context length.

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

**4.** Nobody's implementation is meaningfully better: both are at ~4% of their machine's
peak, so both are leaving the same fraction on the table. Your colleague is simply running on
hardware ~10x faster. Advice: both should work on **efficiency**, not on hardware - fusion,
larger effective matrices, fewer memory-bound ops. If instead your colleague reported 40% MFU
at 2,600 tok/s, then their implementation would be the better one and the gap would be
hardware.

**5.** The exclusion is not about shape, it is about arithmetic. `wte` and the value
embeddings are used as **lookups**: indexing row `i` of a table does no multiply-accumulate
work proportional to `n_embd`, so the `6N` rule does not apply to them. `lm_head` is used as
a **matmul** against the full vocabulary at every position - real arithmetic, `6 * V *
n_embd` FLOPs per token, and at small scale about half the total.

**6.** The parameter term is `6 * (params - embeddings)` and does not depend on sequence
length at all. The attention term is `12 * n_head * head_dim * min(window, t)` per layer -
**linear in `t` per token**, which means quadratic in `t` for a whole sequence. Scale `t` by
128x and the attention term scales by 128x while the parameter term is unchanged. Implication:
at short context, optimise the matmuls (fusion, dtype, `lm_head` size); at long context,
optimise attention (FlashAttention, sliding windows, sparsity) - the same change can be
irrelevant at one length and decisive at another.
