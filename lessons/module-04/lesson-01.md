# 10 - Loss, gradients, and what backward() actually does

Modules 1-3 built a model that makes predictions. It starts out predicting at random. This
module is about how it gets better, and it starts with the core loop, which in `train.py` is
literally four lines:

    loss = model(x, y)      # forward:  how wrong are we?
    loss.backward()         # backward: which way should every parameter move?
    optimizer.step()        # update:   move them a little
    model.zero_grad(...)    # reset:    forget this step's gradients

Everything else in the file - the schedules, the accumulation, the timing - is bookkeeping
around those four lines.

## The loss: one number for "how wrong"

`model(x, y)` runs the forward pass and compares the predictions with the real next tokens
`y`. For each position it looks up the probability the model gave to the correct token and
takes `-ln` of it (cross-entropy, lesson 03), then averages over all positions:

    model gave the right token probability 0.9     ->  -ln(0.9)    = 0.105   good
    model gave it probability 0.1                  ->  -ln(0.1)    = 2.303   bad
    model gave it 1/8192 (a uniform guess)         ->  -ln(1/8192) = 9.011   untrained

The whole of training is: make this number smaller.

## What a gradient is, on one weight

Forget the model for a moment and take the smallest possible one: a single weight `w`, one
input `x`, one target `y`, and the loss `(w*x - y)^2`.

    w = 2,  x = 3,  y = 5
    prediction = w * x = 6
    loss = (6 - 5)^2 = 1

The **gradient** `d(loss)/d(w)` answers: *if I nudge `w` up a tiny bit, how much does the loss
change, per unit of nudge?* Here it is `2 * (w*x - y) * x = 2 * 1 * 3 = 6`. Check it by
actually nudging:

    w = 2.01  ->  prediction 6.03  ->  loss 1.0609      the loss rose by ~0.06 = 6 * 0.01

A positive gradient means "raising `w` raises the loss". So to *lower* the loss, move `w`
**against** the gradient:

    w_new = w - learning_rate * gradient = 2 - 0.05 * 6 = 1.7
    prediction = 5.1,  loss = 0.01                      much better

That is all of training, repeated for every number in the model at once. The gradient says
which direction; the optimizer (lesson 11) decides how far.

## What `backward()` does

A real model has millions of weights, and the loss depends on them through a long chain of
operations. `loss.backward()` computes the gradient for **every** parameter in one sweep and
stores it next to the parameter:

    p.grad          same shape as p; entry [i, j] is d(loss)/d(p[i, j])

It does this with the **chain rule**: a gradient through a chain of steps is the product of
each step's local gradient. The toy example is already a two-step chain - `h = w*x`, then
`loss = (h - y)^2`:

    d(loss)/d(h) = 2 * (h - y) = 2        how the loss reacts to h
    d(h)/d(w)    = x           = 3        how h reacts to w
    d(loss)/d(w) = 2 * 3       = 6        multiply them - same answer as before

During the forward pass PyTorch records every operation it performed. `backward()` walks that
record in reverse, from the loss back to the first layer, multiplying local gradients as it
goes. Each layer only needs to know its own local rule; the chain does the rest.

Two practical consequences:

**Memory.** To compute a local gradient, a layer usually needs the input it saw on the way
forward (for `h = w*x`, the gradient for `w` needs `x`). So every one of those intermediate
tensors is kept alive until `backward()` runs. That, not the parameters, usually decides how
big a batch fits. For the real CPU training run (`B = 8`, `T = 256`, `n_embd = 256`,
`V = 8192`):

    one stream-sized activation   8 * 256 * 256    =    524,288 floats  ~  2 MB
    the logits                    8 * 256 * 8192   = 16,777,216 floats  ~ 64 MB

Every block keeps several stream-sized tensors and a few MLP-sized ones (4x wider, ~8 MB), and
those add up across layers. But no single tensor comes close to the logits: at a small
vocabulary-heavy scale like this, the `(B, T, vocab_size)` output is the biggest thing in
memory by a factor of eight.

**Cost.** For each matrix multiply, backward computes *two* things: the gradient for the
layer's input (so the gradient can keep flowing back to earlier layers) and the gradient for
the weight (so this layer can learn). That is two matmuls where the forward had one - so
backward costs about twice the forward. That is the `2N + 4N = 6N` rule from lesson 08.

## Gradients add up - and that is deliberate

`p.grad` is **added to**, never overwritten. If one backward pass gives a weight gradient
`0.3` and a second, without clearing, gives `0.5`, the stored gradient is `0.8`.

This trips everyone up once, and it is also a feature: it is exactly how gradient
accumulation (lesson 12) combines several small batches into one big one. But it means the
loop must clear the gradients after every update:

    model.zero_grad(set_to_none=True)

`set_to_none=True` sets `.grad` to `None` instead of filling it with zeros. It frees the memory
right away, and the next backward just creates a fresh tensor instead of adding to a zero one.

## Measuring "how big" the gradients are

A useful single number for the gradient's overall size is the **global gradient norm**: treat
every gradient in the model as one long list of numbers and take its length (square root of
the sum of squares). With two tiny gradient tensors `[3, 4]` and `[12]`:

    sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13

A sudden jump in this number is often the first warning of a run about to blow up. The exercise
computes it.

## The zero-initialised layers, one more time

Lesson 07 showed that `mlp.c_proj` starts at exactly zero and still learns, because its weight
gradient is `grad_out * activation` and does not involve the weight.

The same rule has a mirror image, and the check tests both. The gradient passed *backwards
through* a layer, to the layer before it, is `W^T * grad_out` - and that one **does** involve
the weight. With `c_proj`'s `W = 0`:

    gradient reaching c_fc = 0 * grad_out = 0

So on step 0, `c_proj` gets a gradient and `c_fc` (the layer feeding it) gets **exactly zero**.
After one update `c_proj` is no longer zero, and from step 1 on `c_fc` learns too. Normally
"this weight has zero gradient" means a bug. Here it is by design, and it fixes itself after a
single step.

Zero `c_fc` instead and nothing ever fixes itself: its output is zero, so `c_proj`'s gradient
(`grad_out * 0`) is zero, and ReLU² has slope zero at zero, so `c_fc`'s is too. The zero has to
go on the layer that writes out, not the one that reads in.

## Fast fail

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

Two ways a run can break:

- **`NaN`** ("not a number") - the result of something like `0/0` or `inf - inf`. Any arithmetic
  involving `NaN` produces `NaN`, so a single one spreads through the whole model within a step
  and never goes away.
- **An exploding loss.** An untrained model sits at ~9. A loss above 100 means the weights have
  diverged; the run will not recover.

In an autonomous research loop (lesson 18), a broken run is worthless, and stopping it in the
first seconds - instead of discovering a `NaN` `val_bpb` ten minutes later - means the next
experiment starts sooner.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch
       model = build_model(); x, y = get_batch(B=2, T=64)

       loss = model(x, y); loss.item()                  # ~9.011, a uniform guess
       [p.grad for p in model.parameters()][:2]          # all None before backward
       loss.backward()
       model.transformer.wte.weight.grad.abs().max()     # non-zero now

       c_proj = model.transformer.h[0].mlp.c_proj
       c_proj.weight.abs().max()                         # 0.0 - the weight is zero
       c_proj.weight.grad.abs().max()                    # but its gradient is NOT
       model.transformer.h[0].mlp.c_fc.weight.grad.abs().max()   # 0.0 - blocked by c_proj

       before = c_proj.weight.grad.norm().item()
       model(x, y).backward()                            # again, without clearing
       c_proj.weight.grad.norm().item() / before         # ~2.0 - gradients accumulate

   And the one-weight example, so the chain rule is something you have run, not read:

       w = torch.tensor(2.0, requires_grad=True)
       loss = (w * 3 - 5) ** 2; loss.backward(); w.grad   # tensor(6.)

2. Fill in `lab/exercises/lesson_10.py`: `grad_report(model, x, y)`.

3. Grade it:

       bash lab/lab.sh check 10

## Hints

- Clear any leftover gradients **first** (`model.zero_grad(set_to_none=True)`), or a second
  call to your function returns the sum of two backward passes and the check will catch it.
- The global gradient norm is the L2 norm of all gradients treated as one long vector:
  `sqrt(sum(g.pow(2).sum() for g in grads))`. Use `.item()` to return a Python float.
- `n_with_grad` counts parameter **tensors** whose `.grad is not None`, not individual
  numbers.
- `model.parameters()` is a generator - it can only be looped over once. If you need it twice,
  make a list first.
- Do not call `optimizer.step()`; this exercise stops at the gradients.

## Solution

    import torch

    def grad_report(model, x, y) -> dict:
        model.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        total_sq = sum(g.pow(2).sum() for g in grads)
        return {
            "loss": loss.item(),
            "grad_norm": total_sq.sqrt().item(),
            "n_with_grad": len(grads),
            "n_params": len(params),
        }

## Summary

The loss is one number for how wrong the predictions are. A gradient says how the loss reacts
to nudging one parameter, and training moves every parameter a little against its gradient.
`backward()` computes all of them at once with the chain rule, keeping forward activations
alive to do it and costing about twice the forward pass. Gradients add up until you clear
them, and a zero-initialised output layer still learns - while briefly blocking the gradient
to the layer before it. Next: what the optimizer does with those gradients, and why this repo
uses two different optimizers at once.
