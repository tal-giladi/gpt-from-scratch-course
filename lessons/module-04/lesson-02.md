# 11 - Two optimizers in one model: AdamW and Muon

Lesson 10 ended with a gradient for every parameter. The **optimizer** is the code that turns
those gradients into an actual change to the weights. The simplest possible optimizer is one
line:

    p -= lr * grad          # plain gradient descent

Nobody trains a transformer that way, and the reason is easiest to see with two numbers.
Suppose one weight has gradient `100` and another has gradient `0.001`, and `lr = 0.01`:

    step for weight 1:  0.01 * 100    = 1.0          huge - probably overshoots
    step for weight 2:  0.01 * 0.001  = 0.00001      tiny - effectively never learns

No single `lr` suits both. Different parameters need step sizes that differ by orders of
magnitude, and the right size changes as training goes on. Every modern optimizer is an answer
to one question: **how big a step, for this parameter, right now?**

This repo answers it **twice**: AdamW for embeddings and the few 1-D scalars, Muon for the 2-D
weight matrices inside the blocks. That is unusual, and it is the most modern thing in the
file.

## AdamW, one idea at a time

The repo's AdamW kernel is these lines:

    p.mul_(1 - lr * wd)                              # weight decay
    exp_avg.lerp_(grad, 1 - beta1)                   # m: smoothed gradient
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)       # v: smoothed squared gradient
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-lr / bias1)

It looks dense. It is four separate ideas.

### Idea 1: momentum - smooth out the noise

Each batch is a different handful of text, so each gradient is noisy: the true direction plus
batch-specific jitter. `exp_avg` (call it `m`) is a running average that remembers past
gradients and slowly forgets them:

    m = beta1 * m + (1 - beta1) * grad

(`lerp_(grad, 1 - beta1)` is exactly this formula.) With `beta1 = 0.8`, each step keeps 80% of
the old average and mixes in 20% of the new gradient. Feed it a noisy sequence:

    grad:   1       -1       1        1
    m:      0.200   -0.040   0.168    0.334

The `-1` barely dents it. Over several steps, directions that are consistent build up and
directions that flip back and forth cancel out.

### Idea 2: divide by the typical size - one step size for everyone

`exp_avg_sq` (call it `v`) is the same kind of running average, but of the **squared**
gradient. `sqrt(v)` is therefore a running estimate of how big this parameter's gradient
*usually* is.

The step is `m / sqrt(v)`: the smoothed gradient divided by its own typical size. Go back to
the two weights from the top of the lesson:

    weight 1: gradient always ~100     m ~ 100,    sqrt(v) ~ 100,    m/sqrt(v) ~ 1
    weight 2: gradient always ~0.001   m ~ 0.001,  sqrt(v) ~ 0.001,  m/sqrt(v) ~ 1

Both now take a step of about `lr`. The size of the raw gradient no longer matters, only its
direction and consistency. That per-parameter normalisation is what makes Adam robust. `eps`
(here `1e-10`) is added only to avoid dividing by zero.

### Idea 3: bias correction - fix the cold start

Both averages start at zero. After the first step `m = 0.2 * grad`, which is five times too
small, purely because of the zero it started from. `bias1 = 1 - beta1^step` undoes exactly
that:

    step 1:  bias1 = 1 - 0.8   = 0.2      m / bias1 = 0.2 * grad / 0.2 = grad
    step 10: bias1 = 1 - 0.8^10 = 0.89    barely any correction left

`bias2` does the same for `v`. The combined effect is neat: on step 1, every parameter moves
by exactly `lr` in the direction of its gradient's sign. Run it with gradients `100` and
`0.001` and both weights move from `1.0` to `0.99`.

### Idea 4: the W - decoupled weight decay

    p.mul_(1 - lr * wd)

Before the gradient step, shrink every weight slightly toward zero. With `lr = 0.01` and
`wd = 0.1`, that is `p *= 0.999`. This keeps weights from growing without limit.

The **W** in AdamW stands for doing this *separately* ("decoupled") from the gradient. The older
way added `wd * p` into the gradient - but then Idea 2 divides it by `sqrt(v)`, so parameters
with big gradients got almost no decay. Doing it as its own line gives every weight the same
relative shrink.

In this repo every AdamW group has `weight_decay = 0.0`, so the line does nothing for them;
decay is used by the Muon groups.

## Muon: make every direction of a matrix learn at the same speed

Muon is used for exactly the 2-D weight matrices inside `transformer.h`: `c_q`, `c_k`, `c_v`,
both `c_proj`s, `c_fc`, and `ve_gate`.

### The problem it solves

A weight matrix is not just a bag of independent numbers - it maps input directions to output
directions. Any matrix can be broken down (by the *singular value decomposition*, SVD) into a
set of independent directions, each with a strength called a **singular value**.

The gradient for a matrix is itself a matrix, and for transformers its singular values are
usually wildly unequal. A simple example:

    G = [[10,  0  ],
         [ 0,  0.1]]

This gradient wants to push hard along the first direction and barely at all along the second:
a 100-to-1 ratio. Take the step as-is and the matrix mostly learns along one or two dominant
directions, while the rest crawl.

### The fix: orthogonalise

Muon replaces the gradient with the closest matrix that has **all singular values equal to 1**:
same directions, equal strength (an *orthogonal* matrix, or semi-orthogonal when it is not
square). For the example above:

    G after Muon ~ [[0.91, 0   ],
                    [0,    1.00]]

The 100-to-1 ratio is gone. Every direction now gets a comparable update. In practice this
trains small transformers noticeably faster per step than Adam, which is why it is here.

### How, without an SVD

Computing an SVD directly is slow, especially on a GPU. Muon gets the same answer
approximately with a **Newton-Schulz iteration**, which uses only matrix multiplies:

    X = G / (||G|| * 1.02 + 1e-6)              # scale so the largest singular value < 1
    for a, b, c in polar_express_coeffs[:5]:
        A = X.T @ X
        B = b*A + c*(A @ A)
        X = a*X + X @ B

Each pass applies a polynomial that pushes every singular value toward 1, without ever
computing them. The five `(a, b, c)` triples are pre-tuned coefficients (the "polar express"
variant) that get close in only five passes. It is not exact: on a random badly conditioned
matrix most singular values land between roughly 0.75 and 1.15, and directions that were
essentially zero stay small. That is close enough, and far cheaper than an SVD.

### Three refinements

The rest of `muon_step_fused` adds:

- **Nesterov momentum** on the gradient before orthogonalising it - the momentum from Idea 1,
  plus a "look ahead": the direction used is a blend of the current gradient and the updated
  momentum buffer, which reacts a little faster to changes. The exercise has you write it.
- **NorMuon variance reduction** - Idea 2 again, applied per row (or column) of the
  orthogonalised update.
- **Cautious weight decay** - `mask = (g * p) >= 0`. When `g` and `p` have the same sign, the
  gradient step `-lr * g` is already moving that weight toward zero, and decay is applied
  there. Where the gradient wants the weight to *grow*, decay is skipped, so it never fights
  the gradient.

## Who gets which optimizer, and at what learning rate

With the values `train.py` passes in:

    adamw: lm_head        lr = 0.004 * scale
    adamw: wte            lr = 0.6   * scale
    adamw: value_embeds   lr = 0.6   * scale
    adamw: resid_lambdas  lr = 0.005
    adamw: x0_lambdas     lr = 0.5            betas (0.96, 0.95)
    muon:  every 2-D matrix in transformer.h, grouped by shape, lr = 0.04

**Why embeddings get a big learning rate.** An embedding table is a lookup: a batch only
*uses* the rows for the tokens that appear in it, so each row gets a gradient only
occasionally, and can afford a much bigger step when it does (`0.6`). `lm_head` is the same
shape, but it is a real matmul that scores *every* token at *every* position, so every row gets
a gradient on every step. It gets `0.004` - **150 times smaller**. These are the kind of
numbers a research loop finds; do not change them casually.

**Why the scale factor.**

    dmodel_lr_scale = (model_dim / 768) ** -0.5

The AdamW learning rates were tuned for a model of width 768. Wider models need smaller steps,
and this scales them by `1/sqrt(width / 768)`:

    n_embd = 768   ->  x 1.00
    n_embd = 256   ->  x 1.73      (the real CPU training run)
    n_embd = 128   ->  x 2.45      (the course's toy model)

This is what lets hyperparameters survive a change of width - "try depth 6 instead of 4" is a
one-line experiment instead of a re-tuning project.

**Why Muon groups by shape.** `_step_muon` stacks every matrix of the same shape into one 3-D
tensor and orthogonalises them all in a single batched call. At the course's scale that is 8
matrices of `(128, 128)`, 2 of `(512, 128)`, 2 of `(128, 512)`, and 1 `ve_gate` of `(2, 32)` -
four groups, four calls.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs(); model = build_model()
       opt = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.6,
                                   scalar_lr=0.5, matrix_lr=0.04, weight_decay=0.2)
       for g in opt.param_groups:
           print(g["kind"], len(g["params"]), round(g["lr"], 5), tuple(g["params"][0].shape))

   (Pass the values explicitly: `setup_optimizer()`'s own defaults are older numbers, not
   the ones `train.py` uses.)

   Adam's first step, on gradients 100,000 times apart:

       p = torch.tensor([1.0, 1.0]); g = torch.tensor([100.0, 0.001])
       m, v = torch.zeros(2), torch.zeros(2); T = torch.tensor
       defs.adamw_step_fused(p, g, m, v, T(1.), T(0.01), T(0.8), T(0.95), T(1e-10), T(0.))
       p                                               # tensor([0.99, 0.99]) - same step

   What orthogonalisation does to a badly conditioned matrix:

       torch.manual_seed(0)
       G = torch.randn(64, 64) @ torch.diag(torch.linspace(1, 0.001, 64))
       torch.linalg.svdvals(G)                         # from ~10 down to ~0.0001
       X = G / (G.norm() * 1.02 + 1e-6)
       for a, b, c in defs.polar_express_coeffs[:5]:
           A = X.mT @ X
           X = a * X + X @ (b * A + c * (A @ A))
       torch.linalg.svdvals(X)                         # mostly ~0.75 to ~1.15

2. Fill in `lab/exercises/lesson_11.py`: `adamw_update(...)` and `nesterov_momentum(...)`.

3. Grade it:

       bash lab/lab.sh check 11

## Hints

- `adamw_update` changes `p`, `exp_avg` and `exp_avg_sq` **in place** (the `_`-suffixed
  methods like `mul_`, `lerp_`, `add_`), in the order the kernel does: decay `p` first, then
  update both averages, then the parameter step. Getting the order wrong changes the answer by
  one step's worth of decay, and the check will see it.
- `a.lerp_(b, w)` means `a = a + w * (b - a)`, which is `(1 - w) * a + w * b`. So
  `exp_avg.lerp_(grad, 1 - beta1)` is `beta1 * exp_avg + (1 - beta1) * grad` - Idea 1 exactly.
- `step` counts from 1, not 0. `bias1 = 1 - beta1**step`; at step 0 it would be 0 and you would
  divide by zero.
- `eps` is added **after** the square root, not inside it.
- `nesterov_momentum` must not modify `grads`. `momentum_buffer` is updated in place;
  return the direction to step along.
- The two lines: first update the buffer as ordinary momentum,
  `momentum_buffer.lerp_(grads, 1 - momentum)`. Then look ahead by blending the raw gradient
  toward that updated buffer, `torch.lerp(grads, momentum_buffer, momentum)` - the non-`_`
  version, which returns a new tensor instead of changing `grads`.

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

Plain gradient descent fails because parameters need very different step sizes. AdamW fixes
that per number: smooth the gradient (momentum), divide by its typical size so every parameter
steps about `lr`, correct the zero start, and decay weights separately. Muon fixes it per
matrix: replace the gradient with the nearest matrix whose singular values are all about 1, so
every direction learns at a similar speed, using a few matrix multiplies instead of an SVD.
`setup_optimizer` decides who gets which, at learning rates scaled for the model's width. Next:
the schedules that move those learning rates over a run, and how a big batch is faked on a
small machine.
