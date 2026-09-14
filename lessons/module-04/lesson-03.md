# 12 - Schedules, the time budget, and faking a big batch

The training loop has two pieces left, and neither comes from theory. Each comes from a
constraint:

1. The learning rate should not stay the same for the whole run - so something has to change
   it over time. That is a **schedule**.
2. The batch size you want does not fit in the memory you have - so the big batch has to be
   built out of small ones. That is **gradient accumulation**.

## Progress is measured in seconds, not steps

Almost every training script plans its schedule in steps: "decay the learning rate after step
5,000". This one plans in **seconds**:

    progress = min(total_training_time / TIME_BUDGET, 1.0)

`progress` goes from `0.0` at the start to `1.0` when the time budget is used up. `TIME_BUDGET`
is 300 seconds upstream on a GPU, and 600 in this CPU fork. Every schedule below is a function
of `progress`.

Why does that matter? Take two versions of the model, identical except that one runs each step
in 100 ms and the other, thanks to some speedup, in 80 ms:

    100 ms per step   ->   300 s / 0.10 = 3,000 steps
     80 ms per step   ->   300 s / 0.08 = 3,750 steps

Both runs use the same 300 seconds, and in both the learning rate starts decaying at exactly
the halfway point *in time*. The faster one simply fits 25% more steps into the same schedule.

This is what makes the whole autoresearch idea work. Every experiment costs the same wall-clock
time, so every change is judged by one question: **what `val_bpb` can you reach in this much
time?** A faster kernel and a smarter architecture compete on the same scale - which is how
improvements to real training stacks are actually scored.

**One wrinkle the CPU fork had to fix.** The first few steps are not counted against the timer
(`UNTIMED_STEPS`). Upstream that is 10 steps, because on a GPU the first steps are
`torch.compile` warming up, and it would be unfair to bill that to training. On a CPU with
compile off there is no warm-up to excuse - and at ~8 seconds per step, 10 free steps would be
over a minute of uncounted training. The fork drops it to 2.

## The learning-rate schedule: flat, then down to zero

    WARMUP_RATIO   = 0.0    # no warmup at all
    WARMDOWN_RATIO = 0.5    # the last half of the run lowers the LR
    FINAL_LR_FRAC  = 0.0    # ...all the way to zero

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

The function returns a **multiplier**. Every step, each optimizer group's learning rate is set
to its starting value times this number:

    group["lr"] = group["initial_lr"] * lrm

Read the branches with the repo's settings:

- `progress < 0.0` never happens - there is no warmup phase.
- `progress < 0.5` - the first half of the run - returns `1.0`: full learning rate.
- Otherwise `cooldown = (1 - progress) / 0.5`, which falls from `1` at the halfway point to `0`
  at the end, and the result is `cooldown * 1.0 + (1 - cooldown) * 0.0 = cooldown`.

Evaluated across the run, for Muon's starting learning rate of `0.04`:

    progress     0.0    0.25   0.5    0.6    0.75   0.9    1.0
    multiplier   1.0    1.0    1.0    0.8    0.5    0.2    0.0
    Muon lr      0.04   0.04   0.04   0.032  0.02   0.008  0.0

`FINAL_LR_FRAC` sets the floor. With `0.1` instead of `0.0`, the last line would blend toward
`0.1` and end there instead of at zero.

Two things are worth noticing.

**No warmup.** Most transformer recipes ramp the learning rate up from zero over the first few
hundred steps, because a full learning rate on a randomly initialised model tends to blow up.
This model starts at full speed and does not blow up, because the causes have been designed
out: every layer that writes to the residual stream starts at zero (lesson 07), so the first
updates cannot suddenly swing the output, and QK norm (lesson 05) keeps the attention scores
bounded. With the instability gone, warmup would just waste budget.

**Decay to exactly zero.** Big steps are good for making fast progress; small steps are good for
settling precisely into a good spot instead of bouncing around it. Ending at zero gives the model
that settling phase. The consequence: **a run that is cut short, or whose budget changes, is
unfairly penalised** - it never got its cooldown, so it reports a worse `val_bpb` than the same
model would have reached. Comparisons are only fair at equal budgets, and lesson 19 makes that
a rule.

### Two more schedules ride along

**Muon momentum** rises from `0.85` to `0.95` over the first 300 steps:

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

This one is counted in **steps**, not time, because it is about how many gradients the momentum
buffer has seen, not about the clock. Early on, a lower momentum lets the buffer adapt quickly;
later, higher momentum smooths more. On the CPU, at roughly 8 seconds a step, a 600-second run
is only about 75 steps - so momentum only reaches about `0.875`. It never finishes warming up.
That is one of the quiet ways a laptop run differs from the real thing.

**Weight decay** goes *down* in a straight line as the run progresses:

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)        # 0.2 at the start, 0 at the end

## Gradient accumulation: the batch you cannot fit

A bigger batch gives a less noisy gradient: an average over more text is closer to the true
direction. Upstream wants **524,288 tokens** per optimizer step. That many tokens' worth of
activations (lesson 10) will not fit in memory at once.

So the loop splits the big batch into **micro-batches** that do fit, runs forward and backward
on each, lets the gradients add up (lesson 10), and only then takes one optimizer step:

    TOTAL_BATCH_SIZE = 2**19                                # 524,288 tokens per step
    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN      # 128 * 2048 = 262,144
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd  # 2

    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)
    optimizer.step()

Memory is that of one micro-batch. The result is mathematically the same as one big batch.

### The `/ grad_accum_steps` line is the one people get wrong

Each micro-batch's loss is already an **average** over its own tokens. Suppose two micro-batches
of equal size have losses `2.0` and `4.0`. The loss of the combined batch is their average,
`3.0`, and its gradient is the gradient of `3.0`.

Now look at what the gradients add up to:

    without the division:   grad(2.0) + grad(4.0)          = grad of 6.0    twice too big
    with the division:      grad(2.0 / 2) + grad(4.0 / 2)  = grad of 3.0    correct

Forget the line and every gradient is `grad_accum_steps` times too large. It does not crash
and it does not warn.

How much damage that does depends on the optimizer, and this repo happens to be the forgiving
case. AdamW divides by the gradient's own typical size and Muon rescales the gradient to a
fixed size before orthogonalising (lesson 11), so a constant multiplier on every gradient
mostly cancels out. With plain gradient descent it would multiply the effective learning rate
by `grad_accum_steps`; with gradient clipping it would change which steps get clipped; and any
gradient norm you log or compare would be wrong. The first time you swap the optimizer, the
missing division becomes a real bug. The check for this lesson measures that 4x inflation
directly and makes sure your code avoids it.

### At the course's scale there is no accumulation

With the CPU settings, `2048 / (8 * 256) = 1` - one micro-batch per step, nothing to accumulate.
That is a lesson of its own. The tokens-per-step number is a real hyperparameter, and shrinking
it from 524,288 to 2,048 means each step's gradient is an average over **256 times less text**,
and far noisier. That is another honest cost of training on a laptop.

## Do this

1. In the lab shell, print the schedules:

       bash lab/lab.sh shell

       def lrm(p, warmup=0.0, warmdown=0.5, final=0.0):
           if p < warmup: return p / warmup
           if p < 1 - warmdown: return 1.0
           c = (1 - p) / warmdown
           return c + (1 - c) * final
       [round(lrm(i / 10), 3) for i in range(11)]
       # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
       [round(lrm(i / 10, final=0.1), 3) for i in range(11)]
       # the same shape, ending at 0.1

       [round((1 - min(s / 300, 1)) * 0.85 + min(s / 300, 1) * 0.95, 3) for s in (0, 75, 150, 300, 600)]
       # [0.85, 0.875, 0.9, 0.95, 0.95] - where a 75-step CPU run stops

   And the accumulation bug, measured:

       import torch
       from lib.common import build_model, get_batch
       x, y = get_batch(B=4, T=64)
       def gnorm(m): return sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None).sqrt().item()
       m = build_model(); m(x, y).backward(); one = gnorm(m)
       m = build_model()
       for xb, yb in zip(x.chunk(4), y.chunk(4)): m(xb, yb).backward()     # no / 4
       gnorm(m) / one                                                        # ~4.0

2. Fill in `lab/exercises/lesson_12.py`: `lr_multiplier(...)` and
   `accumulated_grad_norm(model, x, y, micro_batches)`.

3. Grade it:

       bash lab/lab.sh check 12

## Hints

- `lr_multiplier` mirrors the code above, including the `warmup == 0` case: when there is no
  warmup, the multiplier at `progress = 0` is `1.0`, not a division by zero. Guard it with
  `if warmup > 0 and progress < warmup`.
- At exactly `progress = 1 - warmdown` the flat phase has ended, but the decay formula gives
  `cooldown = 1` there, so it returns `1.0` anyway. The curve has no jump.
- For `accumulated_grad_norm`: split the batch with `x.chunk(micro_batches, dim=0)` - it cuts
  along the batch dimension into equal pieces (the check uses a `B` that divides evenly) - and
  divide each loss by `micro_batches` before its `backward()`.
- Clear gradients **once**, at the start - not between micro-batches. Adding up is the whole
  point.
- The norm is the same global one from lesson 10.

## Solution

    import torch

    def lr_multiplier(progress, warmup=0.0, warmdown=0.5, final_frac=0.0):
        if warmup > 0 and progress < warmup:
            return progress / warmup
        if progress < 1.0 - warmdown:
            return 1.0
        cooldown = (1.0 - progress) / warmdown
        return cooldown * 1.0 + (1.0 - cooldown) * final_frac

    def accumulated_grad_norm(model, x, y, micro_batches):
        model.zero_grad(set_to_none=True)
        for xb, yb in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
            loss = model(xb, yb) / micro_batches
            loss.backward()
        total_sq = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        return total_sq.sqrt().item()

## Summary

The schedules run on elapsed *time*, not steps, so a speedup and a modelling change are judged
by the same question: how good a model in this many seconds. The learning rate is flat for the
first half and falls in a straight line to zero; Muon's momentum warms up over 300 steps; weight
decay fades out. Gradient accumulation builds a big batch out of small ones exactly - as long as
each micro-batch's loss is divided by the number of micro-batches. That completes the training
loop. Next module: how to tell whether any of it worked.
