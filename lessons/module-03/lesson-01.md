# 07 - The MLP, the block, and the zero that makes training work

A transformer block has two halves, and they do two different jobs.

- **Attention** is the half that *talks to other tokens*. It is how position 40 collects
  what it needs from positions 3, 17 and 39.
- The **MLP** is the half that *thinks about what it has*. It takes one position's vector,
  runs a small calculation on it, and writes the result back. It never looks at any other
  position.

Attention on its own does not do much computing: it is a weighted average of vectors that
are already in the stream. The real computation - the part that can turn "the previous word
was *not*" plus "this word is *good*" into a new feature meaning *negative* - happens in the
MLP.

## What "MLP" means

**MLP** stands for **Multi-Layer Perceptron**, which is just the old name for "a few linear
layers with a nonlinearity between them". In transformer papers you will see the exact same
component called the **feed-forward network**, or **FFN**. MLP and FFN are two names for one
thing. This repo calls it `MLP`:

    class MLP(nn.Module):
        def __init__(self, config):
            self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        def forward(self, x):
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
            return x

Three steps: widen, bend, narrow.

## `n_embd -> 4*n_embd -> n_embd`: a bigger desk, not more tokens

At the course's toy scale `n_embd = 128`, so the shapes are:

    x              (B, T, 128)
    c_fc(x)        (B, T, 512)     4 x wider
    relu(..)^2     (B, T, 512)     same width, values bent
    c_proj(..)     (B, T, 128)     back to the stream's width

Look at what does **not** change: `B` and `T`. There are still exactly `T` positions, and
position 5 is still only computed from position 5. The widening happens along the *feature*
axis, inside each token's own vector.

Think of it as a bigger desk to work on. The 512 numbers are scratch space: `c_fc` writes
out 512 different "questions" about the vector ("is this a number?", "is this inside a
quote?"), the nonlinearity decides which ones fired, and `c_proj` combines the answers into
128 numbers again. No new information enters - the MLP only has this position's 128 numbers
to start from - but it has much more room to *compute* with them. It is not more tokens, and
it is not information from other positions.

The `4x` is a convention that has survived since 2017, and it is where most of the model's
parameters live: two matrices of `4 * n_embd^2` each, versus attention's four matrices of
`n_embd^2`. That makes the MLPs two thirds of the model's matrix parameters.

## Attention mixes, the MLP computes

So the division of labour is:

| | Attention | MLP (FFN) |
|---|---|---|
| looks at other positions? | **yes** - that is its whole job | **no** - each position alone |
| does a nonlinear calculation? | barely - a weighted average | **yes** - that is its whole job |

Which raises an obvious question: if the MLP never looks at other tokens, how can it ever
compute anything about the sentence?

Because **attention runs first and has already put the information there.** After the
attention half of a block, position 40's vector is no longer "just token 40" - it has had a
weighted mix of positions 3, 17 and 39 added into it. When the MLP then processes position
40 alone, it is working on a vector that *already contains* that context. The MLP does not
need to reach other positions; attention has delivered them to its door.

Stack blocks, and the pattern repeats: gather, compute, gather, compute.

## `relu(x).square()` - ReLU squared

The nonlinearity is two plain steps applied to each of the 512 numbers separately.

1. **ReLU**: `max(0, x)`. Negative numbers become 0, positive numbers pass through.
2. **Square** what is left.

On a few values:

    x          -3      -0.5      0      0.5     3
    relu(x)     0       0        0      0.5     3
    squared     0       0        0      0.25    9

So a negative input is switched off completely, and among the positive ones the contrast is
sharpened: `3` becomes `9`, but `0.5` becomes `0.25`. Strong activations get much stronger
relative to weak ones.

Why this and not GELU or SiLU (the smoother functions you may have seen)? Mostly because it
is **cheap**. ReLU² is a comparison with zero followed by one multiplication, and its
derivative is just `2 * relu(x)`. GELU needs the error function `erf`, and SiLU needs `exp`
for its sigmoid - both more expensive to compute, on every one of the `4 * n_embd` numbers,
at every position, in every layer, forward and backward. ReLU² is also smoother than plain
ReLU where it matters: its slope rises continuously from 0, instead of jumping at zero.

Recent small-model work found it trains at least as well as the fancier options. There is no
deep theory here - it is an empirical choice, of exactly the kind the autonomous research loop
is invited to revisit. Swapping this one line is a legitimate experiment.

## The block, in plain language

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

Read the first line from the inside out:

1. `norm(x)` - make a **normalised copy** of `x` (lesson 06).
2. `self.attn(...)` - run attention on that copy, producing an update.
3. `x + ...` - **add** the update to the original, un-normalised `x`.

The second line does the same thing with the MLP, starting from the `x` the first line just
produced.

This is **pre-norm**: the normalisation is applied to the *input sent into* attention and the
MLP, while the original `x` travels past untouched on what is called the **residual path** -
the `x +` part. Nothing on that path is ever normalised, multiplied or replaced; it only ever
has things added to it.

### Two writers, one stream

That running `x` is the **residual stream**: one vector per position that the whole model
reads from and writes to. Inside a block there are two writers, and they both write to the
same stream, one after the other.

With a tiny 3-number stream at one position:

    x at block entry                     [1.0,  2.0,  0.5]
    attention writes                   + [0.2,  0.0, -0.1]
    x after the first line               [1.2,  2.0,  0.4]
    MLP reads norm of that, writes     + [0.0, -0.5,  0.3]
    x after the second line              [1.2,  1.5,  0.7]

Neither writer replaces the stream; each adds a small correction to it. And because the MLP
reads the stream *after* attention wrote to it, the MLP sees attention's contribution - which
is exactly the "attention delivers, MLP computes" order from above.

### No learned scaling on the branches

Some architectures write `x = x + alpha * f(x)` with a learned scalar `alpha` per branch, so
the model can turn a branch down with one number. This block has no `alpha`: the update is
added at full strength.

So if the model wants attention to contribute only a little at some layer, there is no dial
to turn. The branch itself has to learn to *output smaller numbers* - in practice, its
`c_proj` weights stay small. (The stream does get learned scalars *between* blocks,
`resid_lambdas` and `x0_lambdas` from lesson 04, but those scale the incoming stream, not
either branch's output.)

## Zero init: the block starts as the identity function

First, the names. Each branch ends in a layer called `c_proj`; "proj" is short for
**projection**, and it is simply the conventional name for "the linear layer that maps back
to the stream's width". There is nothing special about it beyond its position. In the MLP:

- `c_fc` is the **first** linear layer, `n_embd -> 4*n_embd` (it widens).
- `c_proj` is the **last** linear layer, `4*n_embd -> n_embd` (it narrows back, and its output
  is what gets added to the stream).

Attention has its own `c_proj` in the same role: the last layer, whose output is added to the
stream.

Now read `init_weights` and one pattern jumps out:

    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)      # random
    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)      # random
    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)      # random
    torch.nn.init.zeros_(block.attn.c_proj.weight)            # ZERO
    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)      # random
    torch.nn.init.zeros_(block.mlp.c_proj.weight)             # ZERO

**Every layer that writes back into the residual stream starts at exactly zero.** So at step
0, `attn(...)` returns zeros, `mlp(...)` returns zeros, and

    x = x + 0 = x

Every block is the identity function: whatever goes in comes out unchanged.

### 2 layers or 50 layers: the same function at the start

Push one position's stream `[1.0, 2.0, 0.5]` through a 2-layer model and a 50-layer model at
initialisation:

    2 layers:    [1.0, 2.0, 0.5] -> block -> +0 -> block -> +0            -> [1.0, 2.0, 0.5]
    50 layers:   [1.0, 2.0, 0.5] -> block -> +0 -> ... 50 times ... -> +0 -> [1.0, 2.0, 0.5]

Both compute `x -> x`. The deep model starts out *behaving like* a shallow one, and each layer
has to earn a contribution by growing its `c_proj` away from zero.

(For the full model there is one extra ingredient: `x0_lambdas` starts at `0.1`, so each layer
also mixes in `0.1 * x0`, and the stream grows to `x0 * (1 + 0.1 * n_layer)`. But the stream
then passes through a final `norm`, which removes any overall scale - so the 2-layer and
50-layer models still produce identical outputs at init.)

### Why a zero `c_proj` is safe: it still learns

The worry is obvious: if a weight is zero, doesn't it stay zero? No - because a linear layer's
weight gradient does not depend on the weight. For `out = W * a`:

    grad_W = grad_out * a

where `a` is the activation coming *into* the layer. The weight's own value does not appear.
With numbers: `grad_out = 2`, `a = 5`, so

    grad_W = 2 * 5 = 10        even though W = 0

The MLP's `c_proj` receives the ReLU² activations from the random `c_fc`, which are nonzero,
so its gradient is nonzero and it moves on the very first step. Once it is nonzero, gradient
flows back through it into `c_fc` too. (At step 0 exactly, `c_fc` gets no gradient - backprop
to it has to pass *through* `c_proj`, which is still zero - but that lasts one step.)

### Why zeroing `c_fc` instead would be fatal

Now do it the other way round, and zero the **first** layer:

    c_fc(x)          = 0            for every input
    relu(0).square() = 0            so the activation a = 0
    grad of c_proj   = grad_out * 0 = 0
    grad of c_fc     = ... * (slope of relu² at 0) = ... * 0 = 0

`c_proj` gets no gradient because its input is zero. `c_fc` gets no gradient because ReLU²
has slope `2 * relu(0) = 0` at zero. Neither weight moves, so on the next step the same thing
happens again, forever. The branch is dead. That is why the zero goes on the layer that
**writes out**, never on the layer that **reads in**.

### The gradient highway

This is not a trick to save a step of training; it is why very deep residual networks train
at all.

Look at one residual line, `y = x + f(x)`. When the gradient flows backward through it, it
splits in two: one part goes through `f` (the attention or MLP), and one part goes straight
through the `x` term, which passes it along **unchanged** (the derivative of `x` with respect
to `x` is 1).

Chain 50 of those and the gradient from the loss has a direct route - a highway - all the way
back to layer 0, without passing through a single weight matrix. Without the `x +`, it would
have to be multiplied through 50 random matrices in a row, and a long product of random
numbers either blows up or shrinks to nothing. With zero init the side roads (`f`) contribute
nothing at the start, so at step 0 the highway is the *only* route, and every layer receives
a clean, full-strength signal from the start.

### The same idea, twice more

**`lm_head` starts at `std=0.001`.** `lm_head` turns the final vector into one score (a
*logit*) per vocabulary token. With weights around `0.001`, every logit is close to `0`:

    logits  ~ [0.0003, -0.0001, 0.0002, ...]      V entries, all nearly equal
    softmax ~ [1/V,    1/V,     1/V,    ...]      nearly uniform

A uniform guess over `V` tokens assigns probability `1/V` to the right answer, so the initial
cross-entropy is `-ln(1/V) = ln(V)` - about `9.01` for this tokenizer's `V = 8192`, exactly
the untrained loss lesson 03 predicted and measured. The model starts from "I have no idea",
not from confident nonsense it first has to unlearn.

**The value-embedding gate starts neutral.** Some layers add an extra value embedding into
`v` (lesson 05), scaled by a gate:

    gate = 2 * sigmoid(ve_gate(x))

`ve_gate` is a linear layer (one output per head, so each head gets its own gate) whose
weights are initialised to zero, so its output is `0` for any input. Then:

    sigmoid(0) = 0.5
    gate       = 2 * 0.5 = 1.0

A gate of `1` multiplies the value embedding by one - it leaves it exactly as it is. So the
gate starts **neutral**, neither boosting nor suppressing, and training can move it either way:
toward `0` (switch the extra embedding off) or toward `2` (double it). The `2 *` is there
precisely so that the zero-init starting point lands in the middle of that range.

## Do this

1. In the lab shell, watch the identity property directly:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs(); norm = defs.norm
       model = build_model()
       block = model.transformer.h[0]

       block.mlp.c_proj.weight.abs().max()      # 0.0
       block.attn.c_proj.weight.abs().max()     # 0.0

       x = torch.randn(1, 8, model.config.n_embd)
       cos_sin = (model.cos[:, :8], model.sin[:, :8])
       out = block(x, None, cos_sin, model.window_sizes[0])
       (out - x).abs().max()                     # ~0: the block is the identity at init

       block.mlp.c_proj.weight.data.normal_(0, 0.02)   # break it
       (block(x, None, cos_sin, model.window_sizes[0]) - x).abs().max()   # now it moves

2. Fill in `lab/exercises/lesson_07.py`: `mlp_forward(mlp, x)` and
   `block_forward(block, x, ve, cos_sin, window_size)`, both written from the equations,
   using the module's own weights.

3. Grade it:

       bash lab/lab.sh check 07

## Hints

- `mlp.c_fc` and `mlp.c_proj` are callable: `mlp.c_fc(x)` applies the linear layer. You do
  not need to touch `.weight` at all.
- ReLU squared is `torch.relu(h).square()` - or `F.relu(h) ** 2`. Not `relu(h**2)`, which
  is a different (and much worse) function: it would be symmetric and never negative-clipped.
- `block_forward` needs `norm` from `train_defs()`, and calls `block.attn(...)` with all
  four arguments in order, and `block.mlp(...)` with one.
- Both additions are `x = x + f(norm(x))`, using the **updated** `x` for the second one -
  not the original.

## Solution

    import torch
    from lib.common import train_defs

    def mlp_forward(mlp, x):
        h = mlp.c_fc(x)
        h = torch.relu(h).square()
        return mlp.c_proj(h)

    def block_forward(block, x, ve, cos_sin, window_size):
        norm = train_defs().norm
        x = x + block.attn(norm(x), ve, cos_sin, window_size)
        x = x + block.mlp(norm(x))
        return x

## Summary

A block is attention plus a per-position ReLU² MLP, each reading a normalised copy of the
stream and adding its result back. Every matrix that writes to the stream is initialised to
zero, so the model begins life as the identity function and each layer has to earn its
place - which is what makes depth trainable. Next: putting the blocks together into a whole
model, and counting exactly where the parameters went.
