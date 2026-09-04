# 10 - Loss, gradients, and what backward() actually does

Training is a loop of four steps, and in `train.py` it is literally four lines:

    loss = model(x, y)      # forward:  how wrong are we?
    loss.backward()         # backward: how should every parameter change?
    optimizer.step()        # update:   change them
    model.zero_grad(...)    # reset:    forget the gradients

Everything else in the file - the schedules, the accumulation, the timing - is
bookkeeping around those four lines.

## What `backward()` computes

For every parameter tensor `p`, `p.grad` ends up holding `d(loss)/d(p)`: a tensor the same
shape as `p`, where each entry says how much the loss would rise if that one number rose by
a tiny amount. Increase a parameter whose gradient is positive and the loss goes up; so to
go down, you move **against** the gradient. That is the entire idea; the optimizer's only
job is deciding how big a step to take.

The mechanism is the chain rule, applied backwards through the graph PyTorch recorded
during the forward pass. Two things follow that matter in practice:

- **Memory.** Every intermediate activation needed for the backward pass is kept alive
  until `backward()` runs. That, not the parameters, is what usually decides your batch
  size. `n_layer * B * T * n_embd` floats, several times over.
- **Cost.** The backward pass computes two things per matmul (gradient w.r.t. the input,
  gradient w.r.t. the weight), so it costs roughly twice the forward pass. Hence the 6N
  rule from lesson 08: 2N forward + 4N backward.

## Gradients accumulate - and that is deliberate

`p.grad` is **added to**, never replaced. Call `backward()` twice without clearing and you
get the sum of both. This trips up everyone once and is the mechanism behind gradient
accumulation (lesson 12), so the loop must clear it:

    model.zero_grad(set_to_none=True)

`set_to_none=True` sets `.grad` to `None` instead of a tensor of zeros - it frees memory and
skips a pointless "add zero" in the next backward.

## The zero-initialised layers still learn

Lesson 07 left a question open. `mlp.c_proj.weight` starts at exactly zero, so the layer's
output is zero, so the layer contributes nothing. Does it get a gradient?

Yes - and this is worth being sure about rather than believing. For `y = W @ h`, the
gradient with respect to `W` is `grad_y @ h.T`. It depends on the *input activation* `h`,
not on `W`. `h` is non-zero, so the gradient is non-zero, and `c_proj` starts moving on the
very first step.

Now push on that same equation once more, because it has a second consequence that is easy
to miss. The gradient flowing *backwards past* that layer is `W.T @ grad_y` - and `W` **is**
zero. So on step 0, `c_fc` (the layer feeding `c_proj`) receives **exactly zero gradient**.
The MLP's first layer does not move at all on the first step; it starts learning on step 1,
once `c_proj` has become non-zero. The check for this lesson asserts both halves on the real
model, and it is worth pausing on: "this weight has zero gradient" is normally a bug, and
here it is by design and self-repairing after one step.

That also explains why the zero goes on `c_proj` and not `c_fc`. Zero both and the whole
sub-layer is permanently dead: `h` would be zero, so `c_proj`'s gradient (`grad_y @ h.T`)
would be zero, so nothing would ever move. The asymmetry is the whole trick.

## Fast fail

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

A tiny piece of engineering with a big payoff in an autonomous loop. A diverged run is
worthless, and the sooner it stops the sooner the next experiment starts. `NaN` also has
the property that it poisons everything it touches, so detecting it early - rather than
discovering a `NaN` `val_bpb` twenty minutes later - matters.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch
       model = build_model(); x, y = get_batch(B=2, T=64)

       loss = model(x, y); loss.item()                  # ~9.011
       [p.grad for p in model.parameters()][:2]          # all None before backward
       loss.backward()
       model.transformer.wte.weight.grad.abs().max()     # non-zero now

       c_proj = model.transformer.h[0].mlp.c_proj
       c_proj.weight.abs().max()                         # 0.0 - the weight is zero
       c_proj.weight.grad.abs().max()                    # but its gradient is NOT

       before = c_proj.weight.grad.norm().item()
       model(x, y).backward()                            # again, without clearing
       c_proj.weight.grad.norm().item() / before         # ~2.0 - gradients accumulate

2. Fill in `lab/exercises/lesson_10.py`: `grad_report(model, x, y)`.

3. Grade it:

       bash lab/lab.sh check 10

## Hints

- Clear any leftover gradients **first** (`model.zero_grad(set_to_none=True)`), or a second
  call to your function returns the sum of two backward passes and the check will catch it.
- The global gradient norm is the L2 norm of all gradients treated as one long vector:
  `sqrt(sum(g.pow(2).sum() for g in grads))`. Use `.item()` to return a float.
- `n_with_grad` counts parameter **tensors** whose `.grad is not None`, not individual
  numbers.
- `model.parameters()` is a generator - if you need it twice, make a list.
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

`backward()` fills every parameter's `.grad` with the derivative of the loss, at roughly
twice the cost of the forward pass; gradients accumulate until you clear them; and a weight
initialised to zero still receives a gradient because that gradient depends on the layer's
input, not on the weight. Next: what the optimizer does with those gradients - and why this
repo uses two different optimizers at once.
