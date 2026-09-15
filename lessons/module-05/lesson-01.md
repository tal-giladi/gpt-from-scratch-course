# 13 - val_bpb: the one number that decides everything

Every run of `train.py` ends with a short summary. This is the one from the course's reference
2-minute CPU run (`assets/logs/cpu-2min-8threads-a.log`):

    val_bpb:          2.282386
    training_seconds: 120.3
    num_steps:        97
    num_params_M:     11.5

Everything in that summary except the first line is context. **`val_bpb` is the score.** The
autonomous research loop in `program.md` keeps a code change if and only if this number went
down. So it is worth knowing exactly what it measures, and why part of the code that computes
it is fenced off with "do not change".

## What "bits per byte" means

Start with the name, read backwards.

**Per byte.** Text on disk is bytes. The word `cat` is 3 bytes; `café` is 5 (the `é` takes two
in UTF-8).

**Bits.** A bit is the unit of "how surprised were you". If the model gives the correct next
piece of text probability `1/2`, predicting it took 1 bit; probability `1/4` is 2 bits;
probability `1/8192` is 13 bits. In general, `bits = -log2(probability)`.

So **bits per byte** is: on average, how many bits of surprise the model needs for each byte of
real text it has never seen. It is literally a compression rate - a model with `val_bpb = 2.0`
could be used to compress this text to 2 bits per byte, a quarter of its original 8. Lower is
better.

**Val** means it is measured on the validation set: text held back from training, so the model
cannot just have memorised it.

### Why bytes and not tokens?

Because tokens depend on the tokenizer, and bytes do not. Suppose the word `running` (7 bytes)
is scored by two models with different tokenizers:

    tokenizer A:  "running"         1 token,  loss 3.0 nats
    tokenizer B:  "run" + "ning"    2 tokens, loss 1.5 + 1.5 nats

Per **token**, A looks twice as bad (3.0 vs 1.5). Per **byte**, both spent 3.0 nats on 7 bytes:
identical. Dividing by bytes means a change to the vocabulary cannot fake an improvement.

### From the model's loss to bits per byte

The model's loss is cross-entropy in **nats** (natural log) per token. Converting is two steps:
nats to bits is dividing by `ln(2) = 0.693`, and then divide by bytes. On four target tokens:

    target     bytes    loss (nats)
    "The"        3         2.0
    " cat"       4         3.0
    <BOS>        0         1.5         special token - excluded
    " sat"       4         2.5

    total nats  = 2.0 + 3.0 + 2.5  = 7.5         (BOS left out)
    total bytes = 3 + 4 + 4        = 11
    val_bpb     = 7.5 / (0.693 * 11) = 0.984

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

It is the worked example above, in batches.

- **`@torch.no_grad()`** - this is measurement, not training, so do not record anything for a
  backward pass.
- **`steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)`** - how many batches to score. On this
  fork: `32,768 // (8 * 256) = 16` batches.
- **`reduction='none'`** - normally the model returns one averaged loss. This asks for **one loss
  per position**, shape `(B, T)`, so each can be matched to its own byte count. It is the whole
  reason `GPT.forward` has a `reduction` argument.
- **`.view(-1)`** - flatten `(8, 256)` into one list of 2,048 numbers, for both the losses and
  the targets, so position `i` in one lines up with position `i` in the other.
- **`token_bytes[y_flat]`** - `token_bytes` is a table built by `prepare.py` with one entry per
  vocabulary id: how many bytes that token decodes to. Indexing it with 2,048 target ids gives
  2,048 byte counts.
- **`mask = nbytes > 0`** - the four special tokens (ids 8188-8191, one of which is the BOS
  marker put at the start of every document) decode to no text at all, so they have 0 bytes.
  Multiplying the losses by the mask drops their nats from the numerator; their 0 bytes already
  add nothing to the denominator. You get no credit for predicting a marker the data pipeline
  inserted itself.
- **Sum everything, divide once at the end.**

### Why sum-then-divide, not average the batches?

Two batches, very different sizes in bytes:

    batch 1:   10 nats over   5 bytes      2.0 nats per byte
    batch 2:  100 nats over 100 bytes      1.0 nats per byte

    average of the two ratios:   (2.0 + 1.0) / 2   = 1.5      wrong
    sum then divide:             110 / 105         = 1.048    right

Batch 1 covers 5 bytes and batch 2 covers 100. They should not get an equal vote. Summing first
gives every byte the same weight, wherever it happens to fall.

## The pinned validation shard

The dataset is split into files called shards. `prepare.py` fixes one of them as the validation
set, forever:

    MAX_SHARD = 6542
    VAL_SHARD = MAX_SHARD            # always the last one: shard_06542.parquet

    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]

However many shards you download for training, the last one is always fetched, and always
removed from the training list. That is the entire defence against the most common way to fool
yourself in machine learning: **measuring on data you trained on**. A model that has seen the
test text gets a score for memory, not for understanding.

Notice how cheap that defence is - three lines - and how easy it is to break. An autonomous agent
editing the data loading code is one careless file pattern away from letting the validation
shard leak into training, and every score after that is fiction. That is why the evaluation lives
in `prepare.py`, which the agent is told never to touch.

## Why "do not change this" is written on it

`prepare.py` labels the section `# Evaluation (DO NOT CHANGE - this is the fixed metric)`.

A number the experimenter is free to redefine is not a metric; it is an opinion. Each of these
would lower `val_bpb` without making the model one bit better:

- score fewer tokens (the easy ones happen to come first)
- use a different shard
- use a shorter context, or skip the special-token mask

After any of those, a run from today and a run from last week are no longer comparable, and the
research loop quietly turns into optimising the measurement instead of the model.

The one legitimate exception is a **whole-fork** change - made once, on purpose, and announced.
That is exactly what this CPU port did: `EVAL_TOKENS` goes from about 21 million to 32,768 and
`MAX_SEQ_LEN` from 2,048 to 256. Numbers from this fork are comparable with each other, and must
never be compared with upstream's. Saying so out loud is what keeps it honest.

## What the numbers mean

Measured on this fork's own 16 validation batches:

    3.35    an untrained model
    3.29    gzip -9, on the same text
    3.01    xz -9
    2.87    bzip2 -9
    2.28    a 2-minute CPU training run (the course's reference log)
    ~1.0    upstream's 5-minute H100 run, on its own settings (program.md's example: 0.9979)

A few things to take from that:

- **The untrained number is predictable.** A uniform guess over 8,192 tokens costs
  `log2(8192) = 13` bits per token. On this text the average token covers 3.88 bytes, and
  `13 / 3.88 = 3.35`. That is the starting line for every run.
- **A random model is about as good as gzip.** Which says less about the model than about gzip:
  general-purpose compressors are poor at English. A couple of minutes of training already beats
  all three.
- **Upstream's number is not comparable** to any of the others - different eval size, context and
  budget, as above. It is here only to show where serious hardware gets to.

Each 0.1 is a real difference in compression. Two identical 2-minute runs for this course landed at
`2.2824` and `2.2756` - a gap of 0.007 from randomness alone (mostly from fitting 97 steps into the
budget in one run and 102 in the other), and on other machines it is often larger. Lesson 19
turns that into a rule about what counts as an improvement.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import math, torch
       from prepare import get_token_bytes, evaluate_bpb, MAX_SEQ_LEN, EVAL_TOKENS
       from lib.common import build_model, tokenizer
       tb = get_token_bytes(); tb.shape, tb.dtype
       (tb == 0).nonzero().flatten()            # the four special tokens: 8188..8191
       MAX_SEQ_LEN, EVAL_TOKENS                  # 256, 32768 - what the fork pinned them to
       EVAL_TOKENS // (8 * MAX_SEQ_LEN)          # 16 batches

       7.5 / (math.log(2) * 11)                  # the worked example: 0.984

       model = build_model()
       evaluate_bpb(model, tokenizer(), 8)       # ~3.35 for an untrained model
       math.log2(8192) / 3.88                    # ...and where that comes from

2. Fill in `lab/exercises/lesson_13.py`: `bpb_over_batches(model, batches, token_bytes)`.

3. Grade it:

       bash lab/lab.sh check 13

   The check runs your function over the same 16 validation batches `evaluate_bpb` would
   use and requires the same number to six decimal places.

## Hints

- `model(x, y, reduction="none")` returns a loss per position, shaped like `y`. Flatten
  both with `.view(-1)` so the byte lookup lines up position by position.
- `token_bytes[y_flat]` looks up one byte count per target token.
- Multiply the per-token losses by the mask (or index with it) before summing - masked-out
  positions must not contribute nats.
- Accumulate Python numbers across batches with `.item()`, not tensors. Adding tensors keeps
  every batch's intermediate results alive until the end.
- Run under `torch.no_grad()`.
- Divide **once**, at the end. If there were no bytes at all (the check tries a batch of nothing
  but special tokens), return `float("inf")` instead of dividing by zero.

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

`val_bpb` is how many bits the model needs, on average, to encode each byte of text it has never
seen - a compression rate, and a tokenizer-proof one because it divides by bytes, not tokens. It
sums nats over every scored token (special tokens excluded), sums bytes, and divides once, on a
validation shard that is pinned and never trained on. It is frozen on purpose, because a metric
the experimenter can edit is not a metric. Next: the other numbers in the log, and what they tell
you about where your time went.
