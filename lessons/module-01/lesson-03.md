# 03 - What the model is actually asked to do

Everything a GPT knows, it learned from one task: **given all the tokens so far, put a
probability on every possible next token.** No labels, no human feedback, no task
descriptions. Just that, a few trillion times.

That task is why the row capacity in lesson 02 was `T + 1` and not `T`. One row of `T + 1`
ids becomes two tensors of length `T`:

    row      = [ a  b  c  d  e ]        # capacity T+1 = 5
    inputs   = [ a  b  c  d ]           # row[:-1]
    targets  = [ b  c  d  e ]           # row[1:]

Position `i` of `inputs` must predict position `i` of `targets`, which is the token that
actually came next. One row gives you `T` training examples, not one - and crucially, all
`T` of them are computed in a **single forward pass**, because the causal mask (lesson 05)
makes position `i`'s prediction depend only on positions `0..i`. That is the trick that
makes pretraining affordable at all.

In `prepare.py` this is two lines:

    cpu_inputs.copy_(row_buffer[:, :-1])
    cpu_targets.copy_(row_buffer[:, 1:])

## The loss: cross-entropy in one sentence

The model's output for one position is a vector of `vocab_size` real numbers, the
**logits**. Softmax turns them into a probability distribution. The loss is the negative
log of the probability the model assigned to the token that actually came next:

    loss = -log p(correct token)

Averaged over every position in the batch. Minimising it means "make the right token more
likely", and because the probabilities must sum to 1, that automatically means "make the
wrong ones less likely".

The units are **nats** (natural log). Two numbers are worth memorising:

- A model that has learned *nothing* - uniform over 8192 tokens - has loss
  `ln(8192) = 9.0109`. When you start a training run, the first step prints `9.011`. If it
  does not, something is wrong before training even began.
- Perfect prediction is loss 0. Real models land between; each 0.69 nats (= ln 2) is one
  bit.

## Why the metric is bits per byte

Loss-per-token is not comparable across tokenizers (lesson 01). So the course's real
metric converts to **bits per byte**:

    bpb = total_nats / (ln(2) * total_bytes)

Three things happen in that formula:

1. **Sum, do not average.** `evaluate_bpb` sums the per-token loss in nats over the whole
   evaluation set and separately sums the UTF-8 byte length of every target token.
2. **Divide by bytes, not tokens.** Now a tokenizer that packs more bytes into each token
   gets no free credit: it has to predict a fatter token to earn the same score.
3. **Divide by `ln 2`** to convert nats to bits. "Bits per byte" is literally a compression
   rate: a model scoring 1.0 bpb could, in principle, compress this text 8:1.

Special tokens (BOS) have a byte length of 0 and are excluded from both sums - you should
not get credit for predicting a marker you inserted yourself.

This is why `val_bpb` is the number the autonomous research loop optimises, and the one
number in `train.py` that lesson 18 will tell you an agent is *not* allowed to touch.

## Do this

1. In the lab shell, see the shift and the initial loss for yourself:

       bash lab/lab.sh shell

       import math, torch
       from lib.common import get_batch, build_model
       x, y = get_batch(B=2, T=128)
       (y[:, :-1] == x[:, 1:]).all()        # True: y is x shifted left by one
       model = build_model()                # random init, nothing learned
       loss = model(x, y); loss.item()      # ~9.011
       math.log(8192)                       # 9.0109... - the same number

       from prepare import get_token_bytes
       tb = get_token_bytes(); tb.shape     # one byte-length per vocab entry
       tb[y[0, :10]]                        # bytes of the first ten targets
       (tb == 0).sum()                      # the 4 special tokens

2. Fill in `lab/exercises/lesson_03.py`: three small functions - `split_row`,
   `uniform_loss`, and `bits_per_byte`.

3. Grade it:

       bash lab/lab.sh check 03

## Hints

- `split_row(row)` is pure slicing: `row[:-1], row[1:]`. It works for a list or a tensor;
  the check uses both.
- `uniform_loss(vocab_size)` is the loss of a model that puts equal probability `1/V` on
  every token: `-log(1/V) = log(V)`. Use `math.log`, natural log.
- `bits_per_byte(total_nats, total_bytes)` is the formula above. Return `float("inf")` if
  `total_bytes` is 0 rather than dividing by zero - a metric with no data is not 0.0, and
  saying so is the difference between a bug you see and one you do not.

## Solution

    import math

    def split_row(row):
        return row[:-1], row[1:]

    def uniform_loss(vocab_size: int) -> float:
        return math.log(vocab_size)

    def bits_per_byte(total_nats: float, total_bytes: int) -> float:
        if total_bytes == 0:
            return float("inf")
        return total_nats / (math.log(2) * total_bytes)

## Summary

The whole training objective is next-token prediction: shift the row by one, ask for a
distribution at every position, score it with cross-entropy. An untrained model scores
`ln(vocab)`; a useful one is measured in bits per byte, which is a compression rate and is
comparable across tokenizers. That closes the data path - next module opens the model
itself, starting with what a token id turns into.
