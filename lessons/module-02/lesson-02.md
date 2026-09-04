# 05 - Attention: queries, keys, values, and the mask

The MLP (lesson 07) processes each position on its own. Attention is the only place in the
whole model where information moves **between** positions - it is how the token at
position 40 finds out what happened at position 3. Everything else is per-position
arithmetic.

The mechanism is a soft dictionary lookup, and the three names are the dictionary:

- **query** (`q`): what this position is looking for.
- **key** (`k`): what each position offers as an index.
- **value** (`v`): what each position hands over if it is selected.

All three are linear projections of the same input:

    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

Note the `view`: one matrix of width `n_embd` is *reshaped* into `n_head` separate heads of
width `head_dim`. The heads are not separate parameters; they are slices of the same
matrix, and they attend independently. Multiple heads let one layer do several different
lookups at once ("the subject of this sentence" and "the last open bracket").

## The four lines of maths

For one head:

    scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)   # (T, T): every query vs every key
    scores = scores.masked_fill(~allowed, -inf)           # causal + window mask
    weights = softmax(scores, dim=-1)                     # each row sums to 1
    out     = weights @ v                                 # weighted average of values

The `/ sqrt(head_dim)` is not decoration. Dot products of `d`-dimensional vectors grow like
`sqrt(d)`; without the scaling, softmax saturates, gradients vanish, and training stalls.

## The causal mask is the whole reason this trains fast

A row of `T` tokens gives `T` training examples (lesson 03) only if position `i` cannot see
position `i+1`. That is enforced here, by setting the scores of future positions to `-inf`
before the softmax, which makes their weights exactly zero.

So the mask is not a safety feature — it is what makes one forward pass compute `T`
independent predictions instead of one.

## Sliding windows: not every layer sees everything

`autoresearch` goes one step further. `window_pattern = "SSSL"` means layer 0, 1, 2 use a
**short** window (half the context), layer 3 uses the **long** one (full context), and the
pattern repeats - with the last layer forced to long. A short-window layer at position `i`
may only attend back to `i - window`.

Why: attention costs `O(T^2)` per layer. If most layers only need local context - and
empirically they do - you buy back a large slice of that cost and spend it on depth
instead. The mask does the work:

    delta = i[:, None] - i[None, :]        # how far back each key is
    mask = (delta >= 0) & (delta <= window)   # causal AND inside the window

`delta >= 0` is causality; `delta <= window` is the window. One boolean matrix expresses
both, and that matrix is exactly what this fork's CPU port had to build by hand, because
on a GPU the FlashAttention-3 kernel takes `causal=True, window_size=(w, 0)` and never
materialises it (lesson 17).

## Two upstream twists

- **QK norm.** Before attention, `q, k = norm(q), norm(k)` - RMS-normalising the queries
  and keys. It bounds the dot products and is one of the cheap training-stability tricks
  that separates a 2019 transformer from a 2025 one.
- **Value embeddings.** Some layers add a *second* embedding table's output into `v`,
  gated per head. That is the `ve` argument threaded through every block. It is a
  "ResFormer"-style trick: give attention direct access to raw token identity, not only to
  the processed stream.

## Do this

1. In the lab shell, watch one attention head work:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs()
       B, T, H, D = 1, 8, 2, 4
       torch.manual_seed(0)
       q, k, v = (torch.randn(B, T, H, D) for _ in range(3))
       out = defs.attention(q, k, v, (T, 0))       # full causal window
       out.shape                                   # (1, 8, 2, 4) - same shape as q

       m = defs._causal_window_mask(8, 3, q.device).int()
       m                                           # look at the band: causal + window 3

2. Fill in `lab/exercises/lesson_05.py`: `manual_attention(q, k, v, window)` implements the
   four lines above with explicit `softmax`, no `scaled_dot_product_attention`.

3. Grade it:

       bash lab/lab.sh check 05

   The check compares your output against `train.py`'s own `attention()` on random tensors
   at several window sizes. Agreeing to 1e-5 means you have written, by hand, what the
   library call does - and incidentally proves that this fork's SDPA fallback is
   numerically the same thing the GPU kernel computes.

## Hints

- Shapes: `q, k, v` arrive as `(B, T, n_head, head_dim)`. Softmax over keys needs
  `(B, n_head, T, head_dim)` - `transpose(1, 2)` on the way in, and again on the way out so
  your return value has the same layout you were given.
- The mask is `(T, T)` and broadcasts across batch and heads without any reshaping.
- Use `float("-inf")` for masked positions, then `torch.softmax(scores, dim=-1)`. Do not
  try to be clever with a large negative constant; `-inf` is exact.
- `window >= T` means "no window at all" - the mask is then just causal. Your `delta <=
  window` condition handles that case automatically, so you do not need a branch.
- Keep everything in float32; the tolerance is tight enough that a stray `.bfloat16()` will
  fail the check.

## Solution

    import math
    import torch

    def manual_attention(q, k, v, window):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        T, head_dim = q.size(2), q.size(3)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        i = torch.arange(T, device=q.device)
        delta = i[:, None] - i[None, :]
        allowed = (delta >= 0) & (delta <= window)
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights @ v).transpose(1, 2)

## Summary

Attention is a masked, scaled, softmax-weighted average of value vectors, and it is the
only cross-position operation in the model. The causal mask is what lets one forward pass
train on every position at once; the sliding window is a compute-versus-context trade made
per layer. Next: how the model knows *where* each token is, given that nothing above
mentioned position at all.
