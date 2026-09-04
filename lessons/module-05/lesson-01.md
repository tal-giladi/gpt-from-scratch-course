# 13 - val_bpb: the one number that decides everything

Every experiment in this repo is judged by a single scalar printed at the end of a run:

    val_bpb:          2.399582

Everything else in the summary - tokens, steps, MFU, parameters - is context. `val_bpb` is
the score, and the autonomous loop in `program.md` keeps a change if and only if this number
went down. So it is worth understanding exactly what it measures, and exactly which parts of
it are not allowed to move.

## The code, line by line

    @torch.no_grad()
    def evaluate_bpb(model, tokenizer, batch_size):
        token_bytes = get_token_bytes(device=DEVICE)
        val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
        steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
        total_nats, total_bytes = 0.0, 0
        for _ in range(steps):
            x, y, _ = next(val_loader)
            loss_flat = model(x, y, reduction='none').view(-1)
            y_flat = y.view(-1)
            nbytes = token_bytes[y_flat]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
        return total_nats / (math.log(2) * total_bytes)

- **`reduction='none'`** returns the loss for every position separately instead of a mean.
  That is the whole reason the model's `forward` takes a `reduction` argument at all.
- **`token_bytes`** is a lookup table built by `prepare.py`: for each of the 8192 vocabulary
  entries, how many UTF-8 bytes it decodes to. Special tokens decode to nothing, so they
  get 0.
- **`mask = nbytes > 0`** drops those special tokens from the numerator, and they
  contribute 0 to the denominator automatically. You do not get credit for predicting a
  marker you inserted yourself.
- **Sum, then divide.** Not a mean of per-batch bpb values. A batch that happens to contain
  longer tokens covers more bytes and must weigh more.

## The pinned validation shard

    MAX_SHARD = 6542
    VAL_SHARD = MAX_SHARD            # always the last one
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]

One specific shard is the validation set, always, and it is excluded from training no
matter how many shards you download. That is the entire defence against the single most
common way to fool yourself in ML: measuring on data you trained on.

It is worth seeing how cheap that defence is - three lines - and how completely it depends
on nobody "improving" the dataloader in a way that lets the val shard leak into training.
An autonomous agent editing `train.py` is one careless glob away from it, which is why the
eval is fenced off in a file the agent is told not to touch.

## Why "do not change this" is written on it

`prepare.py` marks the section: `# Evaluation (DO NOT CHANGE - this is the fixed metric)`.

A number that the experimenter is free to redefine is not a metric, it is an opinion. Fewer
eval tokens, a different shard, a shorter context, dropping the special-token mask - each
would move `val_bpb` without moving the model's quality. Then two experiments run a week
apart are not comparable, and the whole loop degenerates into optimising the measurement.

The one legitimate exception is a **whole-fork** change, made once, deliberately, and
declared - which is exactly what the CPU port did: `EVAL_TOKENS` drops from 21M to 32,768
and `MAX_SEQ_LEN` from 2048 to 256, so numbers from this fork are internally comparable but
must never be compared to upstream's. Say it out loud, or it becomes a lie by omission.

## What the numbers mean

- `val_bpb ≈ 3.4` - an untrained model at this vocabulary (it is `ln(8192)/ln(2)` spread
  over the ~3.8 bytes an average token covers).
- `val_bpb ≈ 2.4` - what ten CPU-minutes buys. Real, and bad.
- `val_bpb ≈ 1.0` - roughly gzip on English text.
- `val_bpb ≈ 0.6-0.8` - a good small model on the same kind of data.

Each 0.1 of bpb is a real difference in compression. Noise between two identical runs at
this scale is on the order of 0.01-0.02, which is the number lesson 19 turns into a rule
about what counts as an improvement.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from prepare import get_token_bytes, evaluate_bpb, MAX_SEQ_LEN, EVAL_TOKENS
       from lib.common import build_model, tokenizer
       tb = get_token_bytes(); tb.shape, tb.dtype
       (tb == 0).nonzero().flatten()            # the four special tokens
       tb.float().mean()                         # average bytes per token
       MAX_SEQ_LEN, EVAL_TOKENS                  # what the fork pinned them to

       model = build_model()
       evaluate_bpb(model, tokenizer(), 8)       # ~3.4 for an untrained model

2. Fill in `lab/exercises/lesson_13.py`: `bpb_over_batches(model, batches, token_bytes)`.

3. Grade it:

       bash lab/lab.sh check 13

   The check runs your function over the same 16 validation batches `evaluate_bpb` would
   use and requires the same number to six decimal places.

## Hints

- `model(x, y, reduction="none")` returns a loss per position, shaped like `y`. Flatten
  both with `.view(-1)` so the byte lookup lines up.
- `token_bytes[y_flat]` is a gather: one byte-count per target token.
- Multiply the per-token losses by the mask (or index with it) before summing - do not
  forget that masked-out positions must not contribute nats.
- Accumulate Python floats/ints across batches (`.item()`), not tensors, or you will hold
  the graph for the whole eval.
- Run under `torch.no_grad()`.
- Divide **once**, at the end.

## Solution

    import math
    import torch

    @torch.no_grad()
    def bpb_over_batches(model, batches, token_bytes):
        total_nats, total_bytes = 0.0, 0
        for x, y in batches:
            loss_flat = model(x, y, reduction="none").view(-1)
            nbytes = token_bytes[y.view(-1)]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
        if total_bytes == 0:
            return float("inf")
        return total_nats / (math.log(2) * total_bytes)

## Summary

`val_bpb` is a summed, byte-weighted, special-token-masked cross-entropy on one pinned
shard the model never trains on - and it is deliberately frozen, because a metric the
experimenter can edit is not a metric. Next: the other numbers in the log, and what they
tell you about where your time went.
