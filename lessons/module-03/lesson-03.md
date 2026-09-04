# 09 - From logits to text

The forward pass ends in four lines that are easy to skim and worth reading slowly:

    x = norm(x)
    softcap = 15
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

- **A final `norm`.** The stream has been accumulating for `n_layer` blocks and its scale
  is arbitrary; normalise once before reading it out.
- **`lm_head`** projects `(B, T, n_embd)` to `(B, T, vocab_size)`. One score per token in
  the vocabulary, at every position. This is the single biggest matmul in a small model.
- **`.float()`** forces float32 even when the rest of the model runs in bf16. Softmax and
  cross-entropy over 8192 categories in low precision lose real accuracy; this is a
  standard and non-negotiable upcast.
- **Softcap.** `15 * tanh(logits / 15)` squashes logits smoothly into `(-15, 15)`. Small
  logits pass through nearly unchanged (`tanh(z) ≈ z` near 0); extreme ones saturate. It
  prevents the model from becoming pathologically over-confident, which is a training
  stability trick borrowed from Gemma. The cost: the model *cannot* express a probability
  ratio bigger than `e^30`, which is fine, and its gradients are gently damped where it is
  most sure, which is the point.

## Logits are not probabilities

A logit is an unnormalised score. `softmax` turns a row of them into a distribution:

    p_i = exp(logit_i) / sum_j exp(logit_j)

Two facts that matter in practice. Softmax is **shift-invariant** - adding a constant to
every logit changes nothing - which is why implementations subtract the max before
exponentiating, and why you should never interpret a raw logit value on its own. And it is
**not** scale-invariant: multiplying all logits by 2 makes the distribution sharper. That
second fact is exactly what temperature exploits.

## Sampling: three knobs

Once you have a distribution over 8192 tokens for the next position, generating text is a
loop: pick one, append it, run the model again.

**Temperature** divides the logits before softmax:

    logits / temperature

`T = 1.0` is the model's honest distribution. `T < 1` sharpens it (at the limit, `T -> 0`
is picking the argmax - greedy decoding). `T > 1` flattens it toward uniform. Note that
because it is applied to the *logits*, temperature is doing exactly the scaling that
softmax is not invariant to.

**Top-k** keeps only the `k` highest-scoring tokens and renormalises over them. Its job is
to cut off the long tail: with 8192 tokens, the bottom 8,000 might hold 2% of the
probability mass collectively, and sampling from that tail is where incoherent text comes
from. `k = 50` is a common default.

**Greedy** is just "always take the argmax". Deterministic, and famously repetitive - a
greedy model finds a loop and stays in it, because the highest-probability continuation of
a repeated phrase is more of the same phrase.

This is what `sample.py` in the fork does, and it is why the fork saves a checkpoint at all:
`val_bpb` tells you the model got better, but reading its output tells you *what* it
learned. After a ten-minute CPU run, expect real words, plausible-looking word endings, and
no sentences that mean anything. That is what 180,000 tokens of training buys, and seeing
it is worth more than being told it.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch, tokenizer
       model = build_model(); tok = tokenizer()
       x, _ = get_batch(B=1, T=32)
       logits = model(x); logits.shape          # (1, 32, 8192) - one row per position
       logits.abs().max()                        # < 15, thanks to the softcap
       probs = torch.softmax(logits[0, -1], -1)
       probs.sum(), probs.max()                  # 1.0, and near-uniform: it is untrained
       probs.topk(5)                             # the five "most likely" next tokens

2. If your ten-minute training run has finished, look at the real thing instead - from the
   autoresearch repo, not the lab:

       docker compose run --rm autoresearch python sample.py --tokens 120

3. Fill in `lab/exercises/lesson_09.py`: `next_token_probs(model, idx, temperature, top_k)`
   and `greedy_generate(model, ids, n_new)`.

4. Grade it:

       bash lab/lab.sh check 09

## Hints

- `model(idx)` with no targets returns logits for **every** position. You want the last
  one: `logits[:, -1, :]`.
- Order of operations: take the last position, divide by temperature, apply top-k, *then*
  softmax. Dividing after softmax is a different (and wrong) function.
- Top-k: `values, _ = torch.topk(logits, k, dim=-1)`, then
  `logits.masked_fill(logits < values[:, -1:], float("-inf"))`. Using `-inf` before the
  softmax makes those probabilities exactly 0, with no renormalising needed afterwards -
  softmax does it for you.
- `top_k=None` (or `k >= vocab_size`) means no filtering at all.
- `greedy_generate` takes a plain `list[int]` and returns the full list including the new
  tokens. Wrap it in a tensor with a batch dimension for the model:
  `torch.tensor([ids])`. Do it under `torch.no_grad()` - you are not training.
- Never feed the model more than `config.sequence_len` tokens: slice with
  `ids[-sequence_len:]`.

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

The model's output is one score per vocabulary entry per position, softcapped for
stability and upcast to float32 for accuracy. Turning those scores into text is a separate,
parameter-free decision - temperature, top-k, or greedy - and it is worth remembering that
none of it is part of the model. That is the architecture complete. Next module: how the
thing is actually trained.
