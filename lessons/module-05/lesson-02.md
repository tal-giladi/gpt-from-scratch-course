# 14 - Reading the training log

While `train.py` runs, it prints one line per optimizer step. Here are lines from the course's
reference 2-minute CPU run (a `depth = 4`, 11.5M-parameter model; the full log is in
`assets/logs/cpu-2min-8threads-a.log`):

    step 00000 (0.0%)  | loss: 9.012250 | lrm: 1.00 | dt: 1584ms | tok/sec: 1,292 | mfu: 28.8% | epoch: 1 | remaining: 120s
    step 00001 (0.0%)  | loss: 8.972597 | lrm: 1.00 | dt: 1303ms | tok/sec: 1,572 | mfu: 35.0% | epoch: 1 | remaining: 120s
    step 00002 (0.0%)  | loss: 8.834017 | lrm: 1.00 | dt: 1667ms | tok/sec: 1,228 | mfu: 27.4% | epoch: 1 | remaining: 120s
    step 00003 (0.0%)  | loss: 8.609586 | lrm: 1.00 | dt: 1105ms | tok/sec: 1,853 | mfu: 41.3% | epoch: 1 | remaining: 119s
    step 00004 (0.9%)  | loss: 8.458106 | lrm: 1.00 | dt: 1333ms | tok/sec: 1,536 | mfu: 34.2% | epoch: 1 | remaining: 118s
    step 00005 (2.0%)  | loss: 8.250811 | lrm: 1.00 | dt: 1147ms | tok/sec: 1,785 | mfu: 39.8% | epoch: 1 | remaining: 116s
    ...
    step 00009 (5.9%)  | loss: 7.760660 | lrm: 1.00 | dt: 1326ms | tok/sec: 1,544 | mfu: 34.4% | epoch: 1 | remaining: 112s
    ...
    step 00045 (50.7%) | loss: 6.692634 | lrm: 0.99 | dt: 1158ms | tok/sec: 1,768 | mfu: 39.4% | epoch: 1 | remaining: 58s
    step 00046 (51.7%) | loss: 6.691249 | lrm: 0.97 | dt: 1120ms | tok/sec: 1,828 | mfu: 40.7% | epoch: 1 | remaining: 57s
    ...
    step 00096 (99.2%) | loss: 6.198533 | lrm: 0.02 | dt: 1314ms | tok/sec: 1,558 | mfu: 34.7% | epoch: 1 | remaining: 0s

Each field answers a different question. Learning to read this line is most of what "debugging
a training run" means in practice.

## Field by field

**`step 00009 (5.9%)`** - the step number, and `progress` from lesson 12 as a percentage of the
time budget. Notice steps 0-3 all say `0.0%`: the percentage is computed *before* the step's time
is added, and steps 0, 1 and 2 are the untimed warm-up steps, which never count.

**`loss`** - the training loss, but smoothed. One batch's loss jumps around too much to read, so
the log shows a running average that keeps 90% of the old value each step:

    smooth = 0.9 * smooth + 0.1 * this_step_loss
    shown  = smooth / (1 - 0.9 ** (step + 1))

The division is the same cold-start correction as Adam's (lesson 11): the average starts at 0,
and without it the first few values would be far too low. That is why step 0 shows `9.012` - the
real loss, `ln(8192)` - instead of `0.9`. What to look for: it should start near 9.01 and keep
falling - here to 6.2 by the last step. A curve that goes flat early usually means the learning
rate is too low; `NaN` or a jump past 100 trips the fast-fail (lesson 10).

**`lrm`** - the learning-rate multiplier from lesson 12. It stays at `1.00` for the first half of
the *time* budget, then falls to 0. In the log above, step 45 is the first past 50% and shows
`0.99`; by step 96, at 99.2%, it is down to `0.02`. If you see it falling in the very first few
steps, your budget is only a few steps long - common the first time you set a short
`AR_TIME_BUDGET` on a slow machine.

**`dt`** - wall-clock milliseconds for this whole optimizer step, including every micro-batch.
This is the number a speed improvement moves. On a laptop it jitters - here anywhere from about
1.0 to 1.7 seconds from one step to the next, as other programs and the operating system take
turns on the CPU. Read it as an average over many steps, never from a single line.

**`tok/sec`** - tokens per second, simply `TOTAL_BATCH_SIZE / dt`. At step 9: `2048 / 1.326 s =
1,544`. The throughput headline.

**`mfu`** - how hard the hardware is working. Its own section below.

**`epoch`** - how many times the data loader has gone through all of the training shards you
downloaded. It starts at 1. If it ever reaches 2 during a run, the model is seeing text it has
already trained on, and you should download more shards.

**`remaining`** - seconds of budget left, `TIME_BUDGET - total_training_time`. It stays at `120s`
for steps 0-2 (untimed), then falls by roughly each step's `dt`: 120 -> 119 -> 118 -> 116.

## MFU: what fraction of the machine you are actually using

**Model FLOPs Utilisation** compares the arithmetic the model *needed* with the arithmetic the
hardware *could* have done in the same time:

    mfu = 100 * flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS

Read it as a fraction:

- top: `flops_per_token * tokens_per_step / seconds_per_step` - FLOPs per second the training
  actually performed.
- bottom: `PEAK_FLOPS` - FLOPs per second this hardware can do at best, on a single big matrix
  multiply.

Step 9 from the log, with the depth-4 model's `33,424,896` FLOPs per token and this fork's measured
CPU peak of 150 billion FLOPs per second:

    done per second:  33,424,896 * 2,048 / 1.326 s  =  51.6 billion FLOPs/s
    possible:                                         150  billion FLOPs/s
    mfu = 100 * 51.6 / 150                           = 34.4%

On a GPU the denominator is `989.5e12` - an H100's peak in bfloat16. A well-tuned GPU training run
usually lands somewhere around 30-50%; far below that, something is wasting the hardware - too
many small operations each paying launch overhead, data loading that cannot keep up, or memory
being shuffled around.

**Why MFU and not just `tok/sec`?** Because `tok/sec` only says *this run is slow*. It cannot say
why. MFU is relative to the hardware, so it separates the two possible fixes:

- low `tok/sec`, **high** MFU - the code is already using the machine well; you need a bigger
  machine or a smaller model.
- low `tok/sec`, **low** MFU - the machine is idling; a better implementation can speed it up.

This fork is a live example of the second case. With PyTorch left on its default 16 threads, the
same run measured ~255 tokens per second, and against the same 150 GFLOP/s peak that is an MFU of
about 6% - the machine was mostly idling. Pinning the thread count to 8 (lesson 16) took it to
~1,650 tokens per second and an MFU in the mid-30s, with no change to the model at all. The run
summary reports `mfu_percent: 36.04` over the whole budget.

## Where the FLOPs go

`flops_per_token` comes from `estimate_flops`, first met in lesson 08:

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = wte + value_embeds + resid_lambdas + x0_lambdas
        attn_flops = sum(12 * h * q * min(window, t) for window in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

Two terms, and they grow with different things.

**`6 * params`, the matrix-multiply term.** 2 FLOPs per parameter forward, 4 backward (lesson 08).
Embedding lookups are left out - they copy a row, they do not multiply. `lm_head` is left **in**:
it is a real multiply against all 8,192 vocabulary rows, and in a small model it is a large part of
this term:

    course toy model (n_embd 128, 2 layers)     lm_head = 73% of the matmul FLOPs
    CPU training run (n_embd 256, 4 layers)     lm_head = 40%
    upstream-sized   (n_embd 512, 8 layers)     lm_head = 14%

**`attn_flops`, the attention term.** `12 * n_head * head_dim * window` per layer. It has nothing
to do with parameter count: it grows with how many earlier positions each token looks at. Halve a
layer's window and its attention FLOPs halve - which is exactly what the `SSSL` pattern buys.

How the split shifts with context length:

    toy model,  context 128        attention =  3% of all FLOPs
    CPU run,    context 256        attention =  6%
    n_embd 512, context 2,048      attention = 26%
    n_embd 512, context 32,768     attention = 85%

At short context the model's cost is almost all matrix multiplies. At long context attention takes
over completely - which is why long-context models care so much about attention kernels (lesson 17).

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       m = build_model()
       m.estimate_flops()                                  # 8,946,048 FLOPs per token
       m.window_sizes                                       # [(64, 0), (128, 0)]
       big = build_model(toy_config(n_layer=6, n_embd=384))
       big.estimate_flops() / m.estimate_flops()            # ~9.5x more per token

       run = build_model(toy_config(n_layer=4, n_embd=256, sequence_len=256))
       f = run.estimate_flops(); f                          # 33,424,896 - the log's model
       100 * f * 2048 / 1.326 / 150e9                       # 34.4 - step 9's mfu, by hand

       long = build_model(toy_config(sequence_len=4096))
       long.estimate_flops() / m.estimate_flops()           # same weights, far more attention

2. Fill in `lab/exercises/lesson_14.py`: `flops_per_token(model)` and
   `mfu_percent(flops_per_token, tokens, seconds, peak_flops)`.

3. Grade it:

       bash lab/lab.sh check 14

## Hints

- Do **not** call `model.estimate_flops()` - that is the thing you are reproducing. Use
  `model.parameters()`, `model.config` and `model.window_sizes`.
- The excluded parameters are exactly: `transformer.wte`, every table in `value_embeds`,
  `resid_lambdas` and `x0_lambdas`. `lm_head` stays in.
- `h = config.n_head`, `q = config.n_embd // config.n_head` (that is `head_dim`),
  `t = config.sequence_len`.
- `model.window_sizes` is a list of `(window, 0)` tuples - loop with `for window, _ in ...`. The
  effective window is `min(window, t)`, and a negative window means "no limit", i.e. `t`.
- `mfu_percent` is a percentage, so multiply by 100. If `seconds` or `peak_flops` is zero, return
  `0.0` rather than dividing by zero.

## Solution

    def flops_per_token(model):
        config = model.config
        nparams = sum(p.numel() for p in model.parameters())
        excluded = (
            model.transformer.wte.weight.numel()
            + sum(ve.weight.numel() for ve in model.value_embeds.values())
            + model.resid_lambdas.numel()
            + model.x0_lambdas.numel()
        )
        h = config.n_head
        q = config.n_embd // config.n_head
        t = config.sequence_len
        attn_flops = 0
        for window, _ in model.window_sizes:
            effective = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective
        return 6 * (nparams - excluded) + attn_flops

    def mfu_percent(flops_per_token, tokens, seconds, peak_flops):
        if seconds <= 0 or peak_flops <= 0:
            return 0.0
        return 100.0 * flops_per_token * tokens / seconds / peak_flops

## Summary

The log line answers three different questions: is it learning (`loss`), is it fast (`dt`,
`tok/sec`), and is it using the hardware well (`mfu`). Only the last one tells you whether a slow
run needs better code or a bigger machine. MFU is FLOPs actually performed per second divided by
the hardware's peak, and the FLOPs estimate behind it splits into a parameter term and an attention
term - with attention going from a rounding error at short context to nearly everything at long
context. Next: what a fixed time budget can actually buy.
