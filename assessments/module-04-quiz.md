# Module 04 quiz - Training

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** You add a print of `mlp.c_fc.weight.grad.abs().max()` after the first `backward()` of
a fresh model and it prints `0.0`. Is this a bug? Answer for step 0 and for step 1, and say
what would make it a real bug.

**2.** Someone removes `model.zero_grad(set_to_none=True)` from the training loop. The run
does not crash. Describe what happens to the effective step size over the first ten steps.

**3.** Explain in one sentence each why AdamW divides by `sqrt(exp_avg_sq)` and why the
weight decay is applied as `p.mul_(1 - lr*wd)` rather than by adding `wd*p` to the gradient.

**4.** Muon orthogonalises the update matrix with five Newton-Schulz iterations instead of
computing an SVD. Give the performance reason and the correctness caveat.

**5.** `EMBEDDING_LR = 0.6` and `UNEMBEDDING_LR = 0.004`. Both matrices are
`(vocab, n_embd)`. Explain the 150x difference.

**6.** You have `DEVICE_BATCH_SIZE = 8`, `MAX_SEQ_LEN = 256` and want `TOTAL_BATCH_SIZE =
2**19`. How many micro-batches per step? Your colleague's implementation forgets
`loss / grad_accum_steps`. What symptom would you see in the log, and what would you
mistakenly conclude?

---

## Answers

**1.** Not a bug at step 0: the gradient that reaches `c_fc` has to pass back through
`c_proj`, whose weight is exactly zero at initialisation, so `W.T @ grad_y` is zero. By step
1, `c_proj` has moved (its own gradient is non-zero, since it depends on the *input*
activation), so `c_fc` starts receiving gradient normally. It would be a real bug if it
persisted past the first step - that would mean `c_proj` is not learning either, which is
what you would get if `c_fc` had been the zero-initialised one instead.

**2.** Gradients accumulate across steps, so step `n`'s update uses the sum of all `n`
backward passes so far. The effective step size grows roughly linearly with the step number
(and the older gradients are stale, computed at parameters that have since moved). Adam
partially masks this - it normalises by the second moment, which is also inflated - so the
run often does not diverge; it just trains worse, which is the hardest kind of bug to notice.

**3.** Dividing by the square root of the smoothed squared gradient makes each coordinate's
step scale-invariant: a parameter whose gradients are consistently tiny gets the same
effective step as one whose gradients are huge. Decoupling the decay keeps it from passing
through that same normalisation - added into the gradient, weight decay would be scaled down
exactly for the parameters with the largest gradients, so the amount of decay would depend on
the loss surface rather than on the hyperparameter.

**4.** Performance: Newton-Schulz is only matrix multiplies, which run at peak throughput on
a GPU (and can be batched over a whole group of same-shaped parameters), while an SVD is
sequential, poorly parallelised, and would dominate the step. Caveat: it is an
*approximation* - five iterations get the singular values close to 1, not exactly 1 - and it
requires the input to be normalised first (`X / (||X|| * 1.02)`) for the iteration to
converge at all.

**5.** They are used differently. `wte` is a lookup: each step touches only the rows for the
tokens present in the batch, so any given row is updated rarely and can afford a large step.
`lm_head` is a dense projection: every row participates in every position of every batch, so
it receives a much larger accumulated update per step and needs a correspondingly smaller
learning rate. Same shape, opposite sparsity.

**6.** `2**19 / (8 * 256) = 256` micro-batches per step. Without the division, the summed
gradient is 256x too large. Symptom: the loss spikes or goes to `NaN` almost immediately and
the fast-fail (`train_loss_f > 100`) fires - or, with a small enough learning rate, the loss
just plateaus high and erratically. The tempting wrong conclusion is "this learning rate is
too high, let me lower it 256x", which papers over the bug and leaves a model that trains
with a badly mis-scaled effective batch.
