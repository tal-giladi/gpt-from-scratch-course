# 04 - Embeddings and the residual stream

A token id is an integer, and an integer carries no meaning - `4711` is not "bigger" than
`312` in any sense the model should care about. The first thing the model does is throw
the integer away and replace it with a **vector**: a row of `n_embd` learned numbers.

That is all `nn.Embedding` is. It is a matrix of shape `(vocab_size, n_embd)` and a lookup:

    x = self.transformer.wte(idx)     # (B, T) ints  ->  (B, T, n_embd) floats

No arithmetic, just indexing. Every training step, gradients flow back into the rows that
were used, so tokens that behave alike drift toward similar vectors. Nobody designs the
embedding table; it is learned like everything else.

## The residual stream

After the lookup, that vector enters a loop, and this is the single most important
structural idea in the whole architecture:

    x = self.transformer.wte(idx)
    x = norm(x)
    x0 = x
    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = ...
        x = block(x, ve, cos_sin, self.window_sizes[i])
    x = norm(x)
    logits = self.lm_head(x)

Look at what a block does (lesson 07 opens it up): `x = x + attn(norm(x))`, then
`x = x + mlp(norm(x))`. Every layer **reads** the current vector and **adds** to it.
Nothing is overwritten. The tensor `x` keeps the same shape `(B, T, n_embd)` from the
embedding all the way to the output.

That running vector is called the **residual stream**. It is best thought of as a shared
whiteboard: each layer looks at what is written, writes its own contribution, and passes
it on. The final layer's contents are read off by `lm_head` to produce logits.

Two consequences worth carrying for the rest of the course:

- **Depth is not a pipeline.** Layer 7 does not receive "the output of layer 6"; it
  receives the embedding plus everything all six previous layers added. Skipping a layer's
  contribution is legal and models routinely learn to nearly ignore some.
- **Anything the model knows is a direction in this space.** If a model represents "this
  text is a question", that has to be a vector added into the stream, because there is
  nowhere else to put it. That is the premise a whole subfield (and the sibling
  abliteration course) is built on.

## The two extra terms upstream added

`autoresearch` is not vanilla GPT-2 here. Two learned scalars per layer sit in front of
every block:

    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

- `resid_lambdas[i]` (init `1.0`) lets the model scale the stream down or up before a
  layer reads it.
- `x0_lambdas[i]` (init `0.1`) re-injects **the original embedding** at every depth. The
  stream drifts a long way from the token identity by layer 8; this gives every layer a
  cheap way to look at the raw token again.

Both are just `nn.Parameter` vectors of length `n_layer`, trained like anything else. Two
tiny tensors, and they are exactly the kind of thing the autonomous research loop is
allowed to change.

## `norm(x)` - RMS norm, no parameters

`norm` is a one-liner: `F.rms_norm(x, (x.size(-1),))`. It rescales each position's vector
to unit root-mean-square, with **no learned gain and no bias** - unusual, and deliberate.
Lesson 06 covers why it is applied where it is; for now note that the value written back
into the stream is always the *unnormalised* one. Normalisation happens on the way *in* to
a layer, never on the stream itself.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch, train_defs
       defs = train_defs(); norm = defs.norm
       model = build_model()
       x, y = get_batch(B=2, T=128)

       emb = model.transformer.wte(x); emb.shape          # (2, 128, 128)
       model.transformer.wte.weight.shape                  # (8192, 128) - the table
       model.transformer.wte.weight.dtype                  # float32 on CPU
       stream = norm(emb)
       stream.norm(dim=-1)[0, :5]                          # per-token magnitudes
       model.resid_lambdas, model.x0_lambdas               # the two scalar vectors
       len(model.transformer.h)                            # 2 blocks at course scale

2. Fill in `lab/exercises/lesson_04.py`: `residual_stream(model, idx)` re-implements the
   loop above and returns the stream **after every stage**: element 0 is the normalised
   embedding, element `i+1` is the stream after block `i`. `n_layer + 1` tensors in total.

3. Grade it:

       bash lab/lab.sh check 04

   The check does something worth noticing: it takes *your* last stream, applies the
   model's final norm, `lm_head` and softcap, and requires the result to equal
   `model(idx)` exactly. If your loop is right, you have reproduced the model's forward
   pass; if it is subtly wrong, no amount of plausible-looking shapes will save you.

## Hints

- `norm` comes from `train_defs()`: `defs = train_defs(); norm = defs.norm`.
- The order inside the loop matters and is easy to get backwards: **first** the
  `resid_lambdas / x0_lambdas` mix, **then** the block. `x0` is the normalised embedding,
  captured once before the loop, and never updated.
- A block is called as `block(x, ve, cos_sin, window_size)`:
  - `ve` is `model.value_embeds[str(i)](idx)` when `str(i)` is a key of
    `model.value_embeds`, otherwise `None`;
  - `cos_sin` is `(model.cos[:, :T], model.sin[:, :T])` where `T = idx.size(1)`;
  - `window_size` is `model.window_sizes[i]`.
- Append a copy of the stream to your list at each stage - appending the same tensor object
  repeatedly is fine here since nothing is modified in place, but `x` is rebound each
  iteration, so append inside the loop, not after it.
- Run under `torch.no_grad()` if you like; the check does not care either way.

## Solution

    import torch
    from lib.common import train_defs

    def residual_stream(model, idx):
        norm = train_defs().norm
        T = idx.size(1)
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        x = norm(model.transformer.wte(idx))
        x0 = x
        streams = [x]
        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
            x = block(x, ve, cos_sin, model.window_sizes[i])
            streams.append(x)
        return streams

## Summary

Token ids become vectors by table lookup, and those vectors enter a running sum that every
layer reads from and adds to - the residual stream, unchanged in shape from input to
output. Upstream adds two learned per-layer scalars: one scaling the stream, one
re-injecting the original embedding. Next: what the first of the two writers to that
stream - attention - actually computes.
