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
instead.

### A window does not make Q, K or V smaller

This is the first thing to get straight, because the name suggests otherwise. A
short-window layer still projects **every** token:

    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)   # all T tokens
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)  # all T tokens
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)  # all T tokens

`q`, `k` and `v` have exactly the same shapes in a `window = 8` layer as in a full-context
one, and the score matrix is still `(B, n_head, T, T)`. Nothing is sliced away, and no
token is dropped from the sequence.

What the window restricts is only **which (query, key) pairs are allowed to interact**. It
is a rule about pairs, not about tokens. Every token is still a query, and every token is
still a key; the mask just says that query `i` may not read key `j` when `j` is more than
`window` positions behind it.

So where does the saving come from? Not from the code you are about to write — writing the
full `T x T` score matrix and then `-inf`-ing most of it computes *everything* and throws
most of it away. That version is correct and it is what the check compares against, but it
saves nothing.

The saving comes from a kernel that never computes the forbidden pairs in the first place.
FlashAttention-3 takes `causal=True, window_size=(w, 0)` as arguments and works in tiles:
for each block of queries it walks only the blocks of keys the window permits, and skips
the rest entirely — no scores, no `-inf`, no softmax terms. The mask is never materialised
as a tensor at all. That turns `O(T^2)` into roughly `O(T * w)`, which is the whole point
(lesson 17 does this arithmetic properly, and shows how this fork's CPU fallback has to
build the mask by hand because it has no such kernel).

Two ways to say the same thing: masking is the *definition* of windowed attention; skipping
is the *optimisation*. They must agree numerically, and they do — that is exactly what
`bash lab/lab.sh check 05` proves.

### Why `SSSL` and not all-long

`SSSL` is a bet, not a theorem. The bet is that **most of what a language model does is
local** — agreement, morphology, the current clause, the last few tokens of code — and that
only some of it needs to reach across the whole context. So spend the cheap local layers
freely and pay for the expensive global ones only every fourth layer.

The objection people raise first is: surely the token I need is sometimes far away? Yes —
and a short window does not prevent that, because **information propagates through depth**.
A short window limits how far one *layer* can reach, not how far *information* can travel.

With `window = w`, position `i` reads positions `i-w .. i`. But each of those positions has
already, in the previous layer, read `w` positions behind *itself*. After two short layers
the receptive field is `2w`; after three, `3w`. Information hops. It is the same reason a
stack of 3x3 convolutions sees a large image: a small kernel applied repeatedly is not a
small receptive field.

What multi-hop propagation is bad at is moving information *unchanged* over a long distance
— each hop mixes it with everything else at that position, so a specific token far back gets
diluted. That is what the periodic `L` layer is for: one layer that can reach any position
in a single hop, with no dilution, every fourth layer, plus the last layer forced to long so
the final prediction always has global access.

So the pattern is a trade with a name for each half: the `S` layers buy depth, and the `L`
layers guarantee that the depth is not stuck talking to its neighbours.

### The mask, one line at a time

The mask is built from positions alone — it does not look at the data:

    i = torch.arange(T)
    delta = i[:, None] - i[None, :]
    mask = (delta >= 0) & (delta <= window)

Take `T = 4` and `window = 2`, so `i = [0, 1, 2, 3]`.

`i[:, None]` is a **column** — the query index, varying down the rows:

    [[0],
     [1],
     [2],
     [3]]

`i[None, :]` is a **row** — the key index, varying across the columns:

    [[0, 1, 2, 3]]

Subtracting broadcasts them into a `(4, 4)` matrix where `delta[i, j] = i - j`, "how many
positions back key `j` is from query `i`":

    delta =
    [[ 0, -1, -2, -3],
     [ 1,  0, -1, -2],
     [ 2,  1,  0, -1],
     [ 3,  2,  1,  0]]

The diagonal is `0` (a position looking at itself), below it is positive (the past), above
it is negative (the future).

`delta >= 0` — **causality**. Negative means key `j` is in the future of query `i`:

    [[ True, False, False, False],
     [ True,  True, False, False],
     [ True,  True,  True, False],
     [ True,  True,  True,  True]]

`delta <= window` — **locality**. With `window = 2`, anything more than 2 steps back is out.
Note this condition says nothing about the future; it only trims the past:

    [[ True,  True,  True,  True],
     [ True,  True,  True,  True],
     [ True,  True,  True,  True],
     [False,  True,  True,  True]]

`&` — **both at once**, which is the band you actually want:

    mask = (delta >= 0) & (delta <= window) =
    [[ True, False, False, False],
     [ True,  True, False, False],
     [ True,  True,  True, False],
     [False,  True,  True,  True]]

Read the last row: query 3 may attend to keys 1, 2 and 3, but not to key 0 — it is 3 steps
back and the window is 2. Read the first row: query 0 may only attend to itself. Everything
lives in a diagonal band `window + 1` wide, and `window >= T` widens the band until it is
the plain causal triangle again — which is why no special case is needed for the `L` layers.

**A warning about the name.** In this matrix `True` means **allowed**, and `False` means
blocked. That is the opposite of what "mask" usually suggests, and it is the opposite
polarity from what `masked_fill` wants — `masked_fill` fills where the argument is `True`.
So the call has to invert it:

    scores = scores.masked_fill(~mask, float("-inf"))

You can avoid the `~` by building the complement instead, and then the variable name is
honest:

    masked = (delta < 0) | (delta > window)     # True = forbidden
    scores = scores.masked_fill(masked, float("-inf"))

Both are correct and they are exact negations of each other (De Morgan: `~(a & b)` is
`~a | ~b`). Pick one, name it for what `True` means — `allowed` or `masked` — and check the
polarity at the `masked_fill` call. Getting it backwards masks the *legal* positions, which
does not crash; it gives you rows of all-`-inf`, `NaN`s out of the softmax, and a loss that
is `nan` from step 1.

## Two upstream twists

- **QK norm.** Before attention, `q, k = norm(q), norm(k)` - RMS-normalising the queries
  and keys. It bounds the dot products and is one of the cheap training-stability tricks
  that separates a 2019 transformer from a 2025 one.
- **Value embeddings.** Some layers add a *second* embedding table's output into `v`,
  gated per head. That is the `ve` argument threaded through every block. It is a
  "ResFormer"-style trick: give attention direct access to raw token identity, not only to
  the processed stream.

## Writing it out: one line at a time

The four lines above are short, but every one of them is a statement about shapes. Before
you implement them, work through what each one is actually doing to the tensor.

### 1. Transpose in: `(B, T, n_head, head_dim)` -> `(B, n_head, T, head_dim)`

`q`, `k` and `v` arrive laid out as `(B, T, n_head, head_dim)` — that is the layout the
`view` after the projection produced, and it is the layout the caller expects back.

Attention, though, is a per-head operation: each head does its own `T x T` comparison and
knows nothing about the others. So you want the head axis to sit *outside* the two axes the
maths works on, which means moving it next to the batch axis:

    q = q.transpose(1, 2)      # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)

`transpose` swaps two axes. It does not touch a single number — the same values are still
there, they are just addressed in a different order. `B` and `n_head` are now both "outer"
dimensions that every later operation carries along untouched, and the interesting part of
the tensor is the trailing `(T, head_dim)`: `T` vectors of length `head_dim`, one per
position. Do this to all three of `q`, `k` and `v`.

### 2. `q @ k^T`: comparing every query with every key

You want a score for every (query, key) pair: `T x T` numbers per head. A score is the dot
product of a query vector with a key vector, both of length `head_dim`.

Matrix multiply contracts the last axis of the left operand with the second-to-last axis of
the right one. Your `q` is `(B, n_head, T, head_dim)` and so is `k` — multiplying them
directly would try to contract `head_dim` against `T`. So transpose `k`'s **last two**
dimensions first:

    k.transpose(-2, -1)        # (B, n_head, T, head_dim) -> (B, n_head, head_dim, T)
    scores = q @ k.transpose(-2, -1)     # (B, n_head, T, T)

Use `-2, -1` rather than `2, 3` — they mean the same thing here, but the negative form keeps
working if the leading dimensions ever change.

Read the result as: `scores[b, h, i, j]` is **how relevant key `j` is to query `i`** — how
much position `i` wants to hear from position `j`. Row `i` is one query's opinion of all `T`
keys.

### 3. Scaling: which `d` is `sqrt(d)`?

The divisor is `sqrt(head_dim)` — the length of the vectors actually being dotted together,
not the model's embedding width. Each head only ever sees a `head_dim`-sized slice, so
`head_dim` is the `d` whose square root the dot products grow with.

They happen to be the same number when `n_head == 1`, which is exactly the case that lets a
wrong `sqrt(n_embd)` slip through your own testing and then fail on a real config. Take the
dimension from the tensor you are dotting:

    head_dim = q.size(-1)

### 4. Mask before softmax

The mask decides which scores are allowed to exist at all, and it must be applied **before**
the softmax — masking afterwards would leave the illegal positions' probability already
baked into the denominator.

Key `j` is visible from query `i` exactly when `0 <= i - j <= window`. Everything else is
one of two failures:

- `i - j < 0` — key `j` is in the **future**. Causality forbids it.
- `i - j > window` — key `j` is real but **too far back** for this layer's window.

Building `delta` and turning that rule into a boolean matrix is worked through element by
element in "The mask, one line at a time" above. You can express it either way round;
just be clear which one you built:

    i = torch.arange(T, device=q.device)
    delta = i[:, None] - i[None, :]
    masked = (delta < 0) | (delta > window)     # True = forbidden

Forbidden scores become `-inf`, not a small number. `exp(-inf)` is exactly `0`, so those
keys receive exactly zero probability and contribute exactly nothing to the output.

    scores = scores.masked_fill(masked, float("-inf"))

### 5. Broadcasting: a `(T, T)` mask against `(B, n_head, T, T)` scores

`delta` — and the boolean matrix built from it — is `(T, T)`. The scores are
`(B, n_head, T, T)`. They are different shapes, and that is fine: the mask is the same for
every sequence in the batch and for every head, so it should be *reused* across those axes
rather than copied.

`masked_fill` broadcasts for you. It lines the two shapes up from the right — `(T, T)`
against the trailing `(T, T)` of the scores — and stretches the mask across the leading `B`
and `n_head` axes for free.

What does **not** work is boolean indexing:

    scores[masked] = float("-inf")     # wrong

Indexing a `(B, n_head, T, T)` tensor with a `(T, T)` boolean mask does not mean "apply this
pattern to each (b, h) slice". Boolean indexing aligns with the *leading* dimensions, so a
`(T, T)` mask is interpreted against `(B, n_head)`, and you get a shape error or a silently
wrong selection. Use `masked_fill` and let broadcasting do the work.

### 6. `dim=-1`: softmax over the keys

    attn = torch.softmax(scores, dim=-1)

`dim=-1` is the key axis — the last `T` in `(B, n_head, T, T)`. So for one fixed batch item,
one fixed head, and one fixed query position `i`, softmax takes that query's whole row of
key scores and turns it into a probability distribution that sums to 1.

It is a distribution *over keys*, computed independently for every query. Every row of every
`T x T` block sums to 1; the columns mean nothing. Softmaxing over `dim=-2` instead would
normalise each key across queries, which is a different (and wrong) question.

### 7. `attn @ v`: the weighted sum

    out = attn @ v      # (B,n_head,T,T) @ (B,n_head,T,head_dim) -> (B,n_head,T,head_dim)

Look at what the matrix multiply contracts: the key axis of `attn` against the position axis
of `v`. That is exactly the weighted sum you want.

Concretely, for query `i`: take its row of `T` probabilities, multiply each one by the
corresponding position's value vector, and add them all up. Masked positions carry weight
`0` and drop out. The result is a single vector of length `head_dim` per query — attention's
answer for that position, assembled out of the value vectors it was allowed to see.

### 8. Transpose back

`out` is `(B, n_head, T, head_dim)`, but the caller — and the rest of the model — expects
`(B, T, n_head, head_dim)`, the same layout `q` arrived in. Undo the transpose from step 1:

    return out.transpose(1, 2)

A function that takes and returns the same layout composes; one that quietly hands back a
different one produces a bug that only shows up several layers later.

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
