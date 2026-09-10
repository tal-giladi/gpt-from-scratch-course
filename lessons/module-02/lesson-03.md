# 06 - Where am I? Rotary embeddings, and why RMS norm

Read lesson 05 again and look for the word "position". It is not there. `q @ k.T` is a sum
over the feature dimension; permute the tokens and every dot product is unchanged. Left
alone, attention treats a sentence as a **bag of tokens** - "dog bites man" and "man bites
dog" are the same input.

So position has to be injected deliberately. There are three generations of answers:

1. **Learned absolute embeddings** (GPT-2): a second lookup table indexed by position,
   added to the token embedding. Simple; cannot extrapolate past the trained length; and it
   tells a token where it *is*, not how far away another token is.
2. **Sinusoidal** (the original transformer paper): the same idea with fixed sin/cos
   instead of learned values.
3. **Rotary** (RoPE, what this model uses): do not *add* anything. **Rotate** `q` and `k` by
   an angle proportional to their position, before the dot product.

## Why rotating is the better idea

Take the feature vector of a head and read it as `head_dim / 2` two-dimensional pairs.
Rotate every pair by an angle `theta_j * position`, with a different frequency `theta_j`
per pair. The dot product of two rotated vectors depends only on the **difference** of
their angles - which is to say, only on the **relative distance** between the two tokens.

The model never learns "I am at position 137". It learns "this key is 4 tokens behind me",
which is the thing that actually generalises: the same relation is available at position
137 and at position 6,000.

In code, `_precompute_rotary_embeddings` builds the angles once:

    inv_freq = 1.0 / (base ** (channel_range / head_dim))   # base = 10000
    freqs = torch.outer(t, inv_freq)                        # t = 0..seq_len-1
    cos, sin = freqs.cos(), freqs.sin()

Low-index pairs get high frequencies (they turn fast, and encode fine local distance);
high-index pairs get low frequencies (they turn slowly, and encode long-range position).
That geometric spread of frequencies is the whole design, and `base = 10000` is the knob
that sets it. It is precomputed for `sequence_len * 10` positions and sliced per forward
pass, so it costs nothing at training time.

And the rotation itself:

    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

Two details worth catching. The pairing is **first half with second half**
(`x[..., :d]` with `x[..., d:]`), not adjacent elements - a convention, and one you must
match exactly or nothing lines up. And this is applied to `q` and `k` only, never to `v`:
position belongs to *where you look*, not to *what you carry back*.

## RMS norm, and where it is applied

`norm(x)` is `F.rms_norm(x, (x.size(-1),))`:

    x / sqrt(mean(x^2) + eps)

Divide each position's vector by its own root-mean-square. Note what is missing versus the
LayerNorm you may have seen: **no mean subtraction, no learned gain, no bias**. Mean
subtraction turns out to buy nothing here; the gain is redundant next to the very next
linear layer, which can absorb any rescaling. Fewer parameters, fewer kernels, same result.

Where it goes matters as much as what it does. This model is **pre-norm**:

    x = x + attn(norm(x))
    x = x + mlp(norm(x))

The normalisation is on the way *into* each sub-layer; what gets added back to the residual
stream is un-normalised. The alternative (post-norm, normalising the stream itself) makes
deep models much harder to train, because it keeps rescaling the very thing gradients flow
straight through.

There are three separate uses of the same function in this model, and it is worth naming
them: on the embedding (lesson 04), inside every block (twice), and on `q` and `k` before
attention (lesson 05).

## Do this

1. In the lab shell, see the frequencies and the invariance:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs(); model = build_model()
       model.cos.shape                      # (1, seq*10, 1, head_dim/2)
       model.cos[0, :4, 0, :4]              # angles at the first four positions
       model.cos[0, :4, 0, -1]              # the slowest pair barely moves

       H = model.config.n_head              # 2 at the course's toy scale
       D = model.config.n_embd // H         # head_dim = 64, so cos/sin are (..., 32)
       x = torch.randn(1, 6, H, D)          # x's head_dim must match the model's
       cos, sin = model.cos[:, :6], model.sin[:, :6]
       r = defs.apply_rotary_emb(x, cos, sin)
       x.norm(dim=-1), r.norm(dim=-1)       # rotation preserves length

2. Fill in `lab/exercises/lesson_06.py`: `apply_rotary(x, cos, sin)` and `rms_norm(x)`.

3. Grade it:

       bash lab/lab.sh check 06

## Hints

- `d = x.shape[-1] // 2`, then `x1 = x[..., :d]`, `x2 = x[..., d:]`. Concatenate along the
  last dimension (`dim=-1`, which for a 4-D tensor is the same `3` the repo writes).
- `cos` and `sin` arrive shaped `(1, T, 1, d)` and broadcast against `(B, T, H, d)` on
  their own - no reshaping needed. Their last dimension is `head_dim // 2`, so `x`'s
  `head_dim` has to be the model's: build test tensors from `model.config`, not from a
  number you picked, or you get "size of tensor a (4) must match tensor b (32)".
- Get the signs right: `y1 = x1*cos + x2*sin`, `y2 = -x1*sin + x2*cos`. Flipping them
  rotates the other way; the check compares against the repo and will catch it.
- `rms_norm`: `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` with a small `eps`
  (1e-6 is fine). Mean over the **last** dimension only, and `keepdim=True` so it
  broadcasts back.
- Do not call `F.rms_norm` - write the arithmetic.

## Solution

    import torch

    def apply_rotary(x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1)

    def rms_norm(x, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

## Summary

Attention has no idea what order the tokens are in, so position is injected by rotating
queries and keys - which makes the resulting scores depend on relative distance, not
absolute index. RMS norm, with no learned parameters, is applied on the way into every
sub-layer, never to the residual stream itself. That is the attention half of a block
complete; next module builds the other half and then assembles the whole model.
