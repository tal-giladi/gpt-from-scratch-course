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

## First: what `head_dim` counts

Everything below is per **head**, so the number has to be nailed down before the maths makes
sense.

One attention head does not see the whole embedding vector. The model's width `n_embd` is
divided evenly among `n_head` heads, and `head_dim` is the slice each one gets:

    head_dim = n_embd // n_head

Concretely, a model of width 32 with 4 heads gives `head_dim = 8`: head 0 works on features
0-7, head 1 on 8-15, head 2 on 16-23, head 3 on 24-31. At the course's toy scale it is
`128 // 2 = 64`.

The splitting is not a separate operation - it is the `view` from lesson 05:

    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)

`c_q` is one `n_embd -> n_embd` matrix. The `view` reinterprets its output as `n_head`
vectors of `head_dim` instead of one vector of `n_embd`, without moving any data. Q, K and V
are all projected from the same input `x` and then split this way, each head attends on its
own slice, and the per-head outputs are concatenated back into one `n_embd` vector.

That concatenation is why `c_proj` exists. Head 2's output sits in features 16-23 and head
3's in 24-31, and nothing so far has let them interact - up to this point the heads are
completely independent computations laid side by side. `c_proj` is `n_embd -> n_embd`, so
dimensionally it does nothing; its real job is to **mix across heads**, letting the block
combine "what the syntax head found" with "what the long-range head found" before the result
is added back to the residual stream.

For the rest of this lesson, take `head_dim = 8` as the worked example.

## Why rotating is the better idea

Take the feature vector of a head and read it as `head_dim / 2` two-dimensional pairs.
Rotate every pair by an angle `theta_j * position`, with a different frequency `theta_j`
per pair. The dot product of two rotated vectors depends only on the **difference** of
their angles - which is to say, only on the **relative distance** between the two tokens.

The model never learns "I am at position 137". It learns "this key is 4 tokens behind me",
which is the thing that actually generalises: the same relation is available at position
137 and at position 6,000.

### One frequency per pair

Careful with the word **pair**: it means *two features of the same token*, never two tokens.
A `head_dim = 8` vector belonging to one token is read as 4 pairs, and each of those 4 pairs
gets its own rotation speed.

    channel_range = torch.arange(0, head_dim, 2)           # [0, 2, 4, 6] for head_dim = 8
    inv_freq = 1.0 / (base ** (channel_range / head_dim))  # base = 10000

Step by step, with `head_dim = 8`:

    channel_range          = [0,   2,    4,    6   ]
    channel_range/head_dim = [0,   0.25, 0.5,  0.75]
    10000 ** that          = [1,   10,   100,  1000]
    inv_freq = 1/that      = [1,   0.1,  0.01, 0.001]

Four numbers, one per pair. The `arange` steps by 2 because there are only `head_dim/2`
pairs to give frequencies to - an equivalent way to write it is "pair index `j = 0..3`,
divided by the number of pairs, 4", which produces the same exponents.

Pair 0 turns at 1 radian per position: it completes a full circle every ~6 tokens, so it
resolves fine local distance and says nothing useful about position 500. Pair 3 turns at
0.001 radians per position: over a whole 2048-token context it barely completes a third of a
turn, so it encodes coarse, long-range position. Between them the frequencies are spread
geometrically, and `base = 10000` is the single knob that sets how wide that spread is.

### The angle is just position times frequency

There is no more to it than one multiplication:

    angle = position * inv_freq[j]

Take pair 1, whose frequency is `0.1`:

    position:  0     1     2     3     ...
    angle:     0.0   0.1   0.2   0.3   ...

And pair 0, frequency `1.0`, over the same positions: `0, 1, 2, 3`. Same tokens, same pair
index - different speed. Position 0 always gets angle 0 for every pair, which is why
position 0 comes out of the rotation unchanged (the check tests exactly this).

### `torch.outer` builds every (position, pair) angle at once

    t = torch.arange(seq_len)          # [0, 1, 2, ...]
    freqs = torch.outer(t, inv_freq)   # (seq_len, head_dim/2)

`torch.outer` multiplies every element of the first vector by every element of the second.
So `freqs` is a table: **rows are token positions, columns are pairs**, and every cell holds
that position's angle for that pair.

    freqs[pos, j] = pos * inv_freq[j]

            pair0   pair1   pair2   pair3
            (1.0)   (0.1)   (0.01)  (0.001)
    pos 0 [  0.0     0.0     0.0     0.0    ]
    pos 1 [  1.0     0.1     0.01    0.001  ]
    pos 2 [  2.0     0.2     0.02    0.002  ]
    pos 3 [  3.0     0.3     0.03    0.003  ]

Then `cos, sin = freqs.cos(), freqs.sin()` applies elementwise, giving two tables of the
same shape, and `cos[None, :, None, :]` reshapes each to `(1, T, 1, head_dim/2)` so it
broadcasts against `(B, T, n_head, head_dim/2)` - the same angles for every batch row and
every head, which is exactly right: position depends on neither.

This is computed once for `sequence_len * 10` positions and sliced per forward pass, so it
costs nothing at training time.

### The rotation, and which features pair up

    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

`x` is `(B, T, H, head_dim)`. With `head_dim = 8`, `d = 4`, so:

    x1 = x[..., :4]   ->  (B, T, H, 4)    features 0, 1, 2, 3
    x2 = x[..., 4:]   ->  (B, T, H, 4)    features 4, 5, 6, 7

The pairing is **first half with second half**, so the four rotating pairs are

    (x0, x4)   (x1, x5)   (x2, x6)   (x3, x7)

and *not* `(x0, x1), (x2, x3), ...`. Both conventions exist in the wild and they are not
interchangeable - the cos/sin tables are laid out to match this one, so pairing adjacent
elements instead produces a plausible-looking tensor that is simply wrong.

`y1` and `y2` are each `(B, T, H, 4)`: `y1` holds the rotated first component of all four
pairs, `y2` the rotated second component. `cos` and `sin` are `(1, T, 1, 4)` and broadcast
across batch and heads on their own. Concatenating on the last axis puts the tensor back to
`(B, T, H, 8)` - the same shape and feature order it arrived in, which is required, because
the next thing that happens to it is a `q @ k.T` that assumes that order.

    torch.cat([y1, y2], 3)     # 4 + 4 = 8, back to head_dim

(`3` is the last axis of a 4-D tensor; `dim=-1` is the same thing, and is what the exercise
uses.)

Two more things worth catching. A rotation preserves length, so `x.norm(dim=-1)` and
`r.norm(dim=-1)` come out equal - it moves a vector, it never grows or shrinks it. And this
is applied to `q` and `k` only, never to `v`: position belongs to *where you look*, not to
*what you carry back*.

## RMS norm, and where it is applied

`norm(x)` is `F.rms_norm(x, (x.size(-1),))`:

    x / sqrt(mean(x^2) + eps)

Divide each position's vector by its own root-mean-square.

### The arithmetic, on four numbers

Take `x = [2, 4, 4, 6]`:

    square:   [4, 16, 16, 36]
    mean:     (4 + 16 + 16 + 36) / 4 = 18
    sqrt:     sqrt(18) = 4.2426
    divide:   [2, 4, 4, 6] / 4.2426 = [0.471, 0.943, 0.943, 1.414]

Check it: the squares of the result are `[0.222, 0.889, 0.889, 2.0]`, and their mean is
`1.0`. That is the entire guarantee - **root-mean-square 1**, nothing more.

In code the mean is over the last dimension only, with `keepdim=True` so the `(..., 1)`
result broadcasts back across the row:

    x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

Use `torch.sqrt`, not `math.sqrt`. The argument here is a tensor holding one value *per
row*, not a Python float; `math.sqrt` cannot take it and will raise. (`x *
torch.rsqrt(...)` is the same arithmetic in one fused op, and is what the repo effectively
does.)

### What is missing versus LayerNorm

LayerNorm centres first:

    layer_norm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
    rms_norm(x)   =          x            / sqrt(mean(x^2) + eps)

Two deliberate omissions:

- **No mean subtraction.** For `[2, 4, 4, 6]` the mean is `4`; LayerNorm would subtract it
  and normalise `[-2, 0, 0, 2]`, RMS norm does not. Empirically the centring buys nothing
  here, and dropping it removes a reduction pass.
- **No learned `gamma` or `beta`.** This is not an oversight. Every use of `norm()` is
  immediately followed by a linear layer, and a per-feature gain is exactly what a linear
  layer's weights already express - scaling column `j` of the next weight matrix has the
  same effect as a `gamma[j]` would, so the parameter is redundant. The bias goes for the
  same reason (and every `nn.Linear` in this model is `bias=False` anyway). Fewer
  parameters, fewer kernels, same expressive power.

### Pre-norm: where it goes matters as much as what it does

This model normalises on the way **into** each sub-layer, not on the way out:

    x = x + attn(norm(x))          # pre-norm  (this model)
    x = x + mlp(norm(x))

    x = norm(x + attn(x))          # post-norm (the 2017 original)
    x = norm(x + mlp(x))

The difference is what happens to `x` itself. In pre-norm the residual stream is never
touched: `x` is copied, normalised, fed through the sub-layer, and only the *result* is
added back. That leaves an unbroken identity path from the embedding all the way to the
final layer - a gradient can travel the whole depth of the network without passing through
a single normalisation.

In post-norm the stream itself is rescaled at every one of the `2 * n_layer` steps, so that
straight path does not exist and deep stacks become much harder to train. (Post-norm
transformers generally need a learning-rate warmup to survive the first few hundred steps;
pre-norm ones mostly do not.)

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
- `rms_norm`: `x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)` with a small `eps`
  (1e-6 is fine). Mean over the **last** dimension only, and `keepdim=True` so it
  broadcasts back. It must be `torch.sqrt`, not `math.sqrt` - the argument is a tensor with
  one value per row, and `math.sqrt` only takes a scalar. `x * torch.rsqrt(...)` is the same
  arithmetic in one op if you prefer it.
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
