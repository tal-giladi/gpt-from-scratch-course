# 14 - Reading the training log

Here is a line from a real CPU run of this fork:

    step 00003 (0.0%) | loss: 8.510523 | lrm: 1.00 | dt: 31596ms | tok/sec: 259 | mfu: 4.3% | epoch: 1 | remaining: 120s

Six numbers, and each one answers a different question. Learning to read this line is most
of what "debugging a training run" means in practice.

- **`loss`** is an EMA of the training loss, debiased. Not the raw value - one batch is too
  noisy to read. It should fall from `ln(vocab)` and keep falling; if it plateaus early
  the learning rate is probably too low, and if it spikes to `NaN` the fast-fail fires.
- **`lrm`** is the learning-rate multiplier from lesson 12. Flat 1.00 for the first half of
  the budget, then decaying. If it is decaying already at step 3, your time budget is
  smaller than one step - which happens the first time you run this on a CPU.
- **`dt`** is wall-clock per optimizer step, including all micro-batches. This is the number
  a performance change moves.
- **`tok/sec`** is `TOTAL_BATCH_SIZE / dt`. The throughput headline.
- **`mfu`** is Model FLOPs Utilisation - see below.
- **`epoch`** counts passes over the downloaded shards. If it reaches 2 in a five-minute
  run you are training on repeated data and should download more shards.
- **`remaining`** is `TIME_BUDGET - total_training_time`, and it does not move during the
  untimed warm-up steps.

## MFU: what fraction of the machine you are actually using

    mfu = 100 * flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS

Model FLOPs Utilisation is *useful arithmetic performed* divided by *arithmetic the
hardware could have performed*. On an H100 with the bf16 peak of 989.5 TFLOP/s, a
well-tuned training run reaches 40-50%. Below ~20% something is wrong: you are memory-bound,
or launching too many tiny kernels, or waiting on the dataloader.

On the CPU fork it reports about 4%, against a measured fp32 peak of ~30 GFLOP/s. That is
the honest number for eager-mode PyTorch on small matrices: no kernel fusion, no
`torch.compile`, and matrices small enough that per-op overhead dominates the arithmetic.

The reason MFU matters more than `tok/sec` is that it is **hardware-relative**. `tok/sec`
tells you this run is slow; MFU tells you whether the fix is a better implementation or a
bigger machine.

## Where the FLOPs go

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = wte + value_embeds + resid_lambdas + x0_lambdas
        attn_flops = sum(12 * h * q * min(window, t) for window in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

Two terms, and they scale differently:

- **`6 * params`** - the matmul term. 2 FLOPs per parameter forward (a multiply and an add),
  4 more backward. Embeddings are excluded because a lookup does no arithmetic; `lm_head`
  is **not** excluded, because it is a real matmul - and at this course's scale it is about
  half of this term all by itself.
- **`attn_flops`** - the attention term, `12 * n_head * head_dim * effective_window` per
  layer. It scales with sequence length, not parameters, which is why halving a layer's
  window halves its contribution. This is what the `SSSL` pattern buys.

For the course's toy model the split is roughly 90% matmul, 10% attention. Upstream at
`seq_len=2048` it is closer to 70/30 - and at 32k context, attention dominates entirely.
That shift is why long-context models care about attention kernels the way they do.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       m = build_model()
       m.estimate_flops()                                  # FLOPs per token
       m.window_sizes                                       # [(w, 0), ...] per layer
       big = build_model(toy_config(n_layer=6, n_embd=384))
       big.estimate_flops() / m.estimate_flops()            # how much more each token costs

2. Fill in `lab/exercises/lesson_14.py`: `flops_per_token(model)` and
   `mfu_percent(flops_per_token, tokens, seconds, peak_flops)`.

3. Grade it:

       bash lab/lab.sh check 14

## Hints

- Do **not** call `model.estimate_flops()` - that is the thing you are reproducing. Use
  `model.parameters()`, `model.config` and `model.window_sizes`.
- The excluded parameters are exactly: `transformer.wte`, every table in `value_embeds`,
  `resid_lambdas` and `x0_lambdas`. `lm_head` stays in.
- `h = config.n_head`, `q = config.n_embd // config.n_head`, `t = config.sequence_len`.
- `model.window_sizes` is a list of `(window, 0)` tuples. The effective window is
  `min(window, t)` - and a negative window means "no limit", i.e. `t`.
- `mfu_percent` is a percentage, so multiply by 100. Guard `seconds == 0` and
  `peak_flops == 0` by returning `0.0` rather than dividing.

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

The log line tells you three separate things - is it learning (`loss`), is it fast
(`tok/sec`), and is it *efficient* (`mfu`) - and only the third one tells you whether the
problem is your code or your hardware. The FLOPs estimate behind MFU splits cleanly into a
parameter term and a sequence-length term, and which one dominates changes completely with
context length. Next: what a fixed time budget can actually buy.
