# Module 04 quiz - Training

Seven questions. Answers and explanations at the bottom - try all seven first.

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

**7.** A colleague says: "after Muon runs, the weight matrix `W` ends up with singular values
all equal to about 1." Is this accurate? If not, name the matrix whose singular values
actually get pushed toward 1, and say roughly how much `W` itself changes on one step.

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
effective step as one whose gradients are huge - squaring the gradient before smoothing it
throws away the sign, so this running estimate (`v`) tracks only how *big* the gradient
typically is, never which way it points, which is exactly why it is safe to divide by. Decoupling
the decay keeps it from passing through that same normalisation - added into the gradient,
weight decay would be scaled down exactly for the parameters with the largest gradients, so the
amount of decay would depend on the loss surface rather than on the hyperparameter.

**4.** Performance: Newton-Schulz is only matrix multiplies, which run at peak throughput on
a GPU (and can be batched over a whole group of same-shaped parameters), while an SVD is
sequential, poorly parallelised, and would dominate the step. Caveat: it is an
*approximation* - five iterations get the *singular values* close to 1, not exactly 1, and not
the matrix's own entries either (a pure rotation matrix like `[[0.6,-0.8],[0.8,0.6]]` has
singular values of exactly 1 with no entry anywhere near 1, which is the general relationship -
diagonal matrices are the special case where entries and singular values happen to coincide).
It also requires the input to be normalised first (`X / (||X|| * 1.02)`) for the iteration to
converge at all.

**5.** They are used differently. `wte` is a lookup: each step touches only the rows for the
tokens present in the batch, so any given row is updated rarely and can afford a large step.
`lm_head` is a dense projection: every row participates in every position of every batch, so
it receives a much larger accumulated update per step and needs a correspondingly smaller
learning rate. Same shape, opposite sparsity.

**6.** `2**19 / (8 * 256) = 256` micro-batches per step. Without the division, the summed
gradient is 256x too large - and in *this* repo you would see almost nothing in the log. AdamW
divides each gradient by its own running size and Muon rescales the gradient to a fixed norm
before orthogonalising, so a constant factor on every gradient mostly cancels: the loss curve
looks normal. That is exactly what makes it dangerous. The mistaken conclusion is "the division
does not matter" - which stays true only until someone swaps in plain SGD (the effective learning
rate jumps 256x and the fast-fail fires), adds gradient clipping (every step gets clipped), or
logs and compares gradient norms (all 256x off). The bug is silent here, not harmless.

**7.** Not accurate. Muon never inspects or targets `W`'s own singular values - it operates
entirely on the *gradient* `G` that `backward()` produces for `W` (and matrices derived from
it: the Nesterov-blended `direction`, then the orthogonalised, NorMuon-scaled update), pushing
*that* matrix's singular values toward 1. `W` itself is touched exactly once per step, as
`W -= lr * (the finished update)`, so `W` changes by roughly `lr` (`0.04` in this repo) each
step, not by jumping to singular values of 1 - it accumulates many small, well-conditioned
nudges over the course of training rather than being rewritten in one step.
