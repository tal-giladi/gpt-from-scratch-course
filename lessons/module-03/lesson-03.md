# 09 - From logits to text

After the last block, every position holds one vector of `n_embd` numbers. That vector is not
a word. This lesson covers the last few lines of the model, which turn it into **one score
for every word-piece in the vocabulary**, and then the separate step - not part of the model
at all - that turns those scores into actual text.

## The last four lines of the forward pass

    x = norm(x)
    softcap = 15
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

Take them one at a time.

### 1. A final `norm`

The residual stream has had `2 * n_layer` updates added to it (lesson 07), so its overall size
is whatever it happens to be - it might have RMS 1 or RMS 40. One `norm` rescales each
position's vector to RMS 1 before it is read out, so `lm_head` always sees inputs of the same
size, however deep the model is.

### 2. `lm_head`: one score per vocabulary entry

`lm_head` is a linear layer from `n_embd` to `vocab_size`:

    x        (B, T, 128)
    logits   (B, T, 8192)

At every position, it compares the vector against 8,192 learned rows - one per token in the
vocabulary - and outputs one number per row. A high number means "this token fits well here".
These numbers are called **logits**. With `T = 32` positions you get `32 * 8192` scores in one
multiply, which is why this is the single biggest matmul in a small model.

Note that the model produces a prediction at *every* position at once, not just the last. That
is what lets training learn from all `T` positions of a row in one pass (lesson 03).
Generation only uses the last one.

### 3. `.float()`: do the risky part in full precision

On a GPU the model runs in bfloat16, a 16-bit number format that is fast but coarse. The next
steps - softmax and cross-entropy over 8,192 options - involve exponentials, and exponentials
of coarse numbers lose real accuracy (lesson 16 shows where). So the logits are converted to
32-bit float first. On CPU the model is already float32 and this line does nothing; it stays
because the same file runs on both.

### 4. Softcap: a soft ceiling on confidence

    logits = 15 * tanh(logits / 15)

`tanh` is an S-shaped curve that is almost a straight line near zero and flattens out at `-1`
and `+1`. Dividing by 15, applying `tanh`, and multiplying by 15 gives a curve that leaves
small logits alone and bends large ones so they can never pass `15`:

    raw logit      1        5        10       30       100
    softcapped     0.999    4.82     8.74     14.46    15.00

So a logit of `1` is untouched, `10` is pulled down a bit, and `100` is squashed to the
ceiling. Why do this?

- **It stops runaway confidence.** Without a cap, the model can make one logit enormous to be
  "extra sure", which makes the loss and gradients spiky. With it, the most extreme logit
  gap is `15 - (-15) = 30`, so the largest possible probability ratio between two tokens is
  `e^30` - about 10 trillion to one. That is plenty.
- **It damps the gradient exactly where the model is already sure.** On the flat part of the
  curve, pushing a logit higher barely changes the output, so the model has little reason to
  keep pushing.

This is a stability trick popularised by Google's Gemma models. It costs nothing measurable.

## Logits are not probabilities

A logit is a raw score. It can be negative, it does not sum to anything, and on its own a
single logit means nothing. **Softmax** turns a row of them into probabilities:

    p_i = exp(logit_i) / sum over all j of exp(logit_j)

Exponentiate each score (making everything positive), then divide by the total (making it sum
to 1). On a tiny 3-token vocabulary:

    logits           [2.0,    1.0,    0.0  ]
    exp              [7.389,  2.718,  1.000]      sum = 11.107
    probabilities    [0.665,  0.245,  0.090]      sum = 1

Two properties matter in practice.

**Adding the same number to every logit changes nothing.** Add 10 to all three and every
`exp` is multiplied by the same `e^10`, which cancels in the division:

    [12.0, 11.0, 10.0]   ->   [0.665, 0.245, 0.090]      identical

This is why a raw logit value is meaningless on its own - only the *differences* between
logits matter. (It is also why real implementations subtract the largest logit before
exponentiating: same answer, no overflow.)

**Multiplying every logit by a number does change things.** Double them:

    [4.0, 2.0, 0.0]   ->   [0.867, 0.117, 0.016]       sharper

The gaps got bigger, so the top token took more of the probability. That is exactly the
lever temperature pulls.

## Sampling: turning a distribution into a token

Generating text is a loop:

1. Run the model on the tokens so far.
2. Take the logits at the **last** position.
3. Turn them into probabilities.
4. Pick one token.
5. Append it and go back to step 1.

Steps 3 and 4 have three common knobs, and none of them is part of the model - they are
choices you make at generation time.

### Temperature

    probabilities = softmax(logits / temperature)

Dividing by a temperature below 1 is the same as multiplying by a number above 1 - it
sharpens. Above 1, it flattens. On the same three logits:

    temperature 0.5    [4.0, 2.0, 0.0]    ->   [0.867, 0.117, 0.016]    confident
    temperature 1.0    [2.0, 1.0, 0.0]    ->   [0.665, 0.245, 0.090]    the model's own view
    temperature 2.0    [1.0, 0.5, 0.0]    ->   [0.506, 0.307, 0.186]    closer to random

Temperature never changes *which* token is most likely - the order of the logits is
preserved - only how strongly the top one wins. As temperature approaches 0, the top token's
probability approaches 1: that limit is greedy decoding.

### Top-k

Keep the `k` highest logits, set every other logit to `-inf`, then softmax. `exp(-inf) = 0`,
so the dropped tokens get probability exactly 0, and the survivors automatically share all of
the probability. With `k = 2`:

    logits       [2.0,   1.0,   0.0 ]
    keep top 2   [2.0,   1.0,   -inf]
    softmax      [0.731, 0.269, 0.000]

Why bother? With 8,192 tokens, the bottom 8,000 might each have a tiny probability - but
together they can hold a few percent. Sample often enough and you *will* pick one, and that is
where sudden nonsense words come from. Top-k cuts the tail off. `k = 50` is a common default,
and it is what `sample.py` uses.

### Greedy

Always take the single most likely token. It is deterministic - the same prompt always gives
the same text - and famously repetitive: once a phrase repeats, its most likely continuation
is often the same phrase again, and greedy decoding has no randomness to escape the loop.

## Why the fork saves a checkpoint

`val_bpb` (lesson 13) tells you the model got *better*. It does not tell you *what* it learned.
That is why this fork saves `model.pt` at the end of a run and ships `sample.py`, which does
exactly the loop above with temperature `0.9` and top-k `50`.

After a ten-minute CPU run - on the order of a hundred thousand tokens of training, depending
on your machine - expect real words, plausible word endings and punctuation in roughly the
right places, and no sentences that mean anything. Seeing that is worth more than being told
it.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch, tokenizer
       model = build_model(); tok = tokenizer()
       x, _ = get_batch(B=1, T=32)
       logits = model(x); logits.shape          # (1, 32, 8192) - one row per position
       logits.abs().max()                        # tiny: lm_head starts near zero (lesson 07)
       probs = torch.softmax(logits[0, -1], -1)
       probs.sum(), probs.max()                  # 1.0, and max ~ 1/8192: near-uniform
       probs.topk(5)                             # the five "most likely" next tokens

       l = torch.tensor([2.0, 1.0, 0.0])         # reproduce the tables above
       torch.softmax(l, -1), torch.softmax(l + 10, -1), torch.softmax(l / 0.5, -1)
       15 * torch.tanh(torch.tensor([1., 5., 10., 30., 100.]) / 15)   # the softcap curve

2. If your ten-minute training run has finished, look at the real thing instead - from the
   autoresearch repo, not the lab:

       docker compose run --rm autoresearch python sample.py --tokens 120

   Try `--temperature 0.3` and `--temperature 1.5` on the same prompt and compare.

3. Fill in `lab/exercises/lesson_09.py`: `next_token_probs(model, idx, temperature, top_k)`
   and `greedy_generate(model, ids, n_new)`.

4. Grade it:

       bash lab/lab.sh check 09

## Hints

- `model(idx)` with no targets returns logits for **every** position, shape
  `(B, T, vocab_size)`. You want the last one: `logits[:, -1, :]`, shape `(B, vocab_size)`.
- Order of operations: take the last position, divide by temperature, apply top-k, *then*
  softmax. Dividing the probabilities after softmax is a different (and wrong) function - the
  result would not even sum to 1.
- Top-k: `values, _ = torch.topk(logits, k, dim=-1)` gives the `k` largest per row, sorted, so
  `values[:, -1:]` is the k-th largest (kept 2-D so it broadcasts). Then
  `logits.masked_fill(logits < values[:, -1:], float("-inf"))`. Softmax does the renormalising
  for you.
- `top_k=None` (or `k >= vocab_size`) means no filtering at all.
- `greedy_generate` takes a plain `list[int]` and returns the full list including the new
  tokens. The model wants a batch dimension, so wrap it: `torch.tensor([ids])` is shape
  `(1, len(ids))`. Do it under `torch.no_grad()` - you are not training, so there is no reason
  to record the operations for a backward pass.
- The model's rotary table and training only cover `config.sequence_len` positions, so never
  feed it more than that: slice with `ids[-sequence_len:]`.

## Solution

    import torch

    def next_token_probs(model, idx, temperature=1.0, top_k=None):
        with torch.no_grad():
            logits = model(idx)[:, -1, :].float()
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and top_k < logits.size(-1):
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def greedy_generate(model, ids, n_new):
        ids = list(ids)
        seq_len = model.config.sequence_len
        for _ in range(n_new):
            idx = torch.tensor([ids[-seq_len:]], dtype=torch.long)
            probs = next_token_probs(model, idx)
            ids.append(int(probs[0].argmax()))
        return ids

## Summary

The model ends by normalising the stream, scoring every vocabulary entry at every position,
upcasting to float32, and softcapping the scores into `(-15, 15)`. Softmax turns scores into
probabilities; only the differences between scores matter, and scaling them changes how
confident the result is. Turning probabilities into text - temperature, top-k, greedy - is a
separate choice made at generation time, and none of it is part of the model. That is the
architecture complete. Next module: how the thing is actually trained.
