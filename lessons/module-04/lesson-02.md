# 11 - Two optimizers in one model: AdamW and Muon

Plain gradient descent is `p -= lr * g`. Nobody trains a transformer that way, because the
right step size differs by orders of magnitude between parameters and changes during
training. Every modern optimizer is an answer to "how big a step, per parameter, right
now?"

This repo answers it **twice**: AdamW for vectors and embeddings, Muon for the 2-D matrices
inside the blocks. That is unusual enough to be worth understanding properly - it is the
most modern thing in the file.

## AdamW, in the four lines that matter

    exp_avg.lerp_(grad, 1 - beta1)                   # m: smoothed gradient
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)       # v: smoothed squared gradient
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-lr / bias1)

- `exp_avg` (**momentum**) is an exponential moving average of the gradient. It smooths out
  the noise from one batch to the next.
- `exp_avg_sq` is an EMA of the *squared* gradient - a running estimate of each
  coordinate's typical magnitude.
- Dividing one by the square root of the other makes the step **scale-invariant per
  coordinate**: a parameter whose gradients are consistently tiny gets the same effective
  step as one whose gradients are huge. This is what makes Adam robust, and also what makes
  its steps roughly `lr`-sized regardless of the loss surface.
- `bias1`/`bias2` correct for the fact that both EMAs start at zero and are therefore
  biased toward zero for the first few steps. Without it the first step is far too small.

The **W** is decoupled weight decay, and it is the first line of the fused kernel:

    p.mul_(1 - lr * wd)

Shrink the parameter itself, *separately* from the gradient step. The older approach - add
`wd * p` to the gradient - interacts with Adam's per-coordinate scaling and effectively
decays big-gradient parameters less. Decoupling it fixes that. Here `weight_decay` is 0.0
for every AdamW group anyway; the machinery exists for the Muon groups.

## Muon: orthogonalise the update

Muon applies to exactly the 2-D matrices inside `transformer.h`. Its claim: for a matrix
parameter, the *direction* of the gradient matters more than its per-coordinate magnitudes,
and raw gradients are badly conditioned - a few singular directions dominate, so the update
mostly moves the matrix along one or two axes.

So Muon takes the momentum-smoothed gradient `G` and replaces it with the nearest
**semi-orthogonal** matrix - same singular vectors, all singular values set to 1:

    X = G / ||G||
    for a, b, c in polar_express_coeffs[:ns_steps]:
        A = X.T @ X
        B = b*A + c*(A @ A)
        X = a*X + X @ B

That loop is a **Newton-Schulz iteration** (here the "polar express" variant, five
hardcoded coefficient triples). It computes the orthogonal factor of a polar decomposition
using only matrix multiplies - no SVD, which would be far too slow and does not run well on
a GPU. Five iterations get close enough.

The effect: every direction in the matrix gets updated at a comparable rate, instead of the
update being dominated by whatever direction the gradient happened to be biggest in. In
practice it trains small transformers noticeably faster per step than Adam, which is why it
is here.

The rest of `muon_step_fused` is two refinements: **NorMuon** variance reduction (a
per-row/column second-moment scaling, Adam's idea applied to the orthogonalised update) and
**cautious weight decay** (`mask = (g * p) >= 0` - only decay a parameter when the update
agrees with its current sign).

## Why the split, and the learning rates

    param_groups = [
        adamw: lm_head        lr=0.004 * scale
        adamw: wte            lr=0.6   * scale
        adamw: value_embeds   lr=0.6   * scale
        adamw: resid_lambdas  lr=0.005
        adamw: x0_lambdas     lr=0.5
        muon:  every 2-D matrix in transformer.h, grouped by shape
    ]

Embeddings are lookup tables - each step touches only the rows for tokens in the batch, so
they can take much larger steps (0.6) than a dense matrix. The unembedding is dense and
touches everything, so it gets 0.004 - **150x smaller**. These are not arbitrary: they are
the kind of numbers a research loop finds and a reader should not casually change.

And this line matters more than it looks:

    dmodel_lr_scale = (model_dim / 768) ** -0.5

Learning rates are scaled by `1/sqrt(width)`, calibrated at `n_embd = 768`. It means the
hyperparameters transfer when you change model width - which is what makes "try depth 6
instead of 4" a one-line experiment rather than a re-tuning project. At the course's
`n_embd = 128` that factor is 2.45.

Muon groups are **by shape**, because `_step_muon` stacks all the parameters in a group into
one tensor and orthogonalises them in a single batched call. Same shape, one kernel.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs(); model = build_model()
       opt = model.setup_optimizer()
       for g in opt.param_groups:
           print(g["kind"], len(g["params"]), g["lr"], tuple(g["params"][0].shape))

       # what orthogonalisation does to a badly conditioned matrix
       G = torch.randn(64, 64) @ torch.diag(torch.linspace(1, 0.001, 64))
       torch.linalg.svdvals(G)[:5]                     # wildly unequal
       X = G / (G.norm() * 1.02 + 1e-6)
       for a, b, c in defs.polar_express_coeffs[:5]:
           A = X.mT @ X
           X = a * X + X @ (b * A + c * (A @ A))
       torch.linalg.svdvals(X)[:5]                     # all near 1

2. Fill in `lab/exercises/lesson_11.py`: `adamw_update(...)` and `nesterov_momentum(...)`.

3. Grade it:

       bash lab/lab.sh check 11

## Hints

- `adamw_update` mutates `p`, `exp_avg` and `exp_avg_sq` **in place**, in the order the
  kernel does: decay `p` first, then update both EMAs, then the parameter step. Getting the
  order wrong changes the answer by one step's worth of decay and the check will see it.
- `lerp_(other, w)` is `self = self + w * (other - self)`. For the EMA you want
  `exp_avg.lerp_(grad, 1 - beta1)`, which is the same as
  `exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)`. Either is fine.
- `step` counts from 1, not 0. `bias1 = 1 - beta1**step`.
- `eps` is added **after** the square root, not inside it.
- `nesterov_momentum` must not modify `grads`. `momentum_buffer` is updated in place;
  return the direction to step along.
- The name: `buf` is the ordinary momentum, and looking ahead one more `momentum`-weighted
  step (`grads.lerp(buf, momentum)`) is what makes it Nesterov rather than plain momentum.

## Solution

    import torch

    def adamw_update(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
        p.mul_(1 - lr * wd)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + eps
        p.add_(exp_avg / denom, alpha=-lr / bias1)
        return p

    def nesterov_momentum(momentum_buffer, grads, momentum):
        momentum_buffer.lerp_(grads, 1 - momentum)
        return torch.lerp(grads, momentum_buffer, momentum)

## Summary

AdamW normalises each coordinate by its own recent gradient magnitude and decays weights
separately from the gradient step. Muon replaces the update for 2-D matrices with its
nearest orthogonal matrix, computed by a few matmuls instead of an SVD, so no single
direction dominates. Which parameters get which - and at which learning rate - is a
deliberate table in `setup_optimizer`. Next: the schedules that move those learning rates
over the run, and how a big batch is faked on a small machine.
