# 12 - Schedules, the time budget, and faking a big batch

Two pieces of the training loop remain, and both exist because of a constraint rather than
a theory: the learning rate must change over the run, and the batch you want does not fit
in the memory you have.

## Progress is measured in seconds, not steps

Almost every training script schedules on step count. This one does not:

    progress = min(total_training_time / TIME_BUDGET, 1.0)

`TIME_BUDGET` is 300 seconds upstream (600 in the CPU fork). The schedules are driven by
**elapsed time**, so a change that makes each step 20% faster automatically buys 20% more
steps in the same run, with the schedule stretching to fit.

That is not a stylistic choice; it is what makes the whole autoresearch premise work. Every
experiment costs the same wall-clock time, so any change - a faster kernel, a smaller model,
a cheaper attention window - is judged on the same axis: **what val_bpb can you reach in
five minutes?** A speedup and a modelling improvement compete directly, which is exactly
how research on real training stacks is scored.

One wrinkle the CPU fork had to fix: the first few steps are excluded from the timer
(`UNTIMED_STEPS`), because upstream that window is `torch.compile` warm-up. With compile off
on a CPU those free steps are minutes of untimed training, so the fork drops the count from
10 to 2.

## The learning-rate schedule: no warmup, half warmdown

    WARMUP_RATIO   = 0.0    # no warmup at all
    WARMDOWN_RATIO = 0.5    # the last half of the run decays the LR
    FINAL_LR_FRAC  = 0.0    # ...all the way to zero

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:   return progress / WARMUP_RATIO
        elif progress < 1 - WARMDOWN_RATIO: return 1.0
        else:
            cooldown = (1 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

Flat at 1.0 for the first half, then a straight line to 0.0. Two things worth noticing:

- **No warmup.** Conventional wisdom says a transformer needs a few hundred warmup steps or
  it diverges. This one does not, because of the zero-initialised output projections
  (lesson 07) and the QK norm (lesson 05) - the instabilities warmup was invented to
  paper over have been designed out. Reintroducing warmup here would waste budget.
- **Decay to exactly zero.** The final steps take vanishingly small steps, which lets the
  model settle into a minimum instead of bouncing around it. Cutting a run short - or
  changing the budget mid-flight - therefore reports a worse `val_bpb` than the same model
  deserves, since it never got its cooldown. Comparisons must use equal budgets, which
  lesson 19 makes a rule.

Two more schedules ride along: Muon's momentum warms from 0.85 to 0.95 over the first 300
steps (step-based, not time-based - it is about the momentum buffer filling up), and weight
decay ramps *down* linearly to zero as the run progresses.

## Gradient accumulation: the batch you cannot fit

    TOTAL_BATCH_SIZE = 2**19     # ~524,288 tokens per optimizer step
    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
    optimizer.step()

Large batches make gradients less noisy, and half a million tokens does not fit in memory
at once. So run several **micro-batches**, let gradients accumulate (lesson 10), and step
once. Mathematically identical to one big batch; memory is that of one micro-batch.

The `/ grad_accum_steps` is the part people get wrong. Each micro-batch's loss is already a
*mean* over its own tokens; summing four means gives four times the mean. Dividing restores
it. Forget that line and your effective learning rate is silently multiplied by
`grad_accum_steps` - a bug that does not crash, does not warn, and just makes training
worse. The check for this lesson catches exactly it.

At the course's CPU scale, `2048 / (8 * 256) = 1`, so there is no accumulation at all -
which is itself a lesson: the tokens-per-step number is a real hyperparameter, and shrinking
it to 2048 from 524,288 makes the gradient far noisier. That is one of the honest costs of
running this on a laptop.

## Do this

1. In the lab shell, plot the schedule crudely:

       bash lab/lab.sh shell

       def lrm(p, warmup=0.0, warmdown=0.5, final=0.0):
           if p < warmup: return p / warmup
           if p < 1 - warmdown: return 1.0
           c = (1 - p) / warmdown
           return c + (1 - c) * final
       [round(lrm(i / 10), 3) for i in range(11)]
       # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

2. Fill in `lab/exercises/lesson_12.py`: `lr_multiplier(...)` and
   `accumulated_grad_norm(model, x, y, micro_batches)`.

3. Grade it:

       bash lab/lab.sh check 12

## Hints

- `lr_multiplier` mirrors the code above, including the `warmup == 0` case: when there is no
  warmup the multiplier at `progress = 0` is `1.0`, not a division by zero. Guard it.
- At exactly `progress = 1 - warmdown` the flat phase has ended; the decay branch there
  returns 1.0 anyway, so the boundary is continuous either way.
- For `accumulated_grad_norm`: split with `x.chunk(micro_batches, dim=0)` (equal-sized
  chunks, since B is divisible), and remember `loss / micro_batches` before each
  `backward()`.
- Clear gradients once at the start, not between micro-batches - accumulating is the whole
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

The learning rate is flat for half the run and then decays linearly to zero, driven by
elapsed *time* rather than step count - which is what lets a speedup and a modelling change
be compared on one axis. Gradient accumulation reproduces a large batch on a small machine
exactly, provided each micro-batch's loss is divided by the number of micro-batches. That
completes the training loop. Next module: how to tell whether any of it worked.
