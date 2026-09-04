# 07 - The MLP, the block, and the zero that makes training work

Attention moves information between positions. It does not do much *thinking* - it is a
weighted average of vectors that are already in the stream. The thinking happens in the
other half of a block: a two-layer MLP applied to every position independently.

    class MLP(nn.Module):
        def __init__(self, config):
            self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        def forward(self, x):
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
            return x

Up to `4 * n_embd`, a nonlinearity, back down to `n_embd`. That `4x` expansion is a
convention that has survived since 2017 and it is where most of a transformer's parameters
live: two matrices of `4 * n_embd^2` each, versus attention's four of `n_embd^2`. Two
thirds of the model's matrix parameters are in the MLPs.

## `relu(x).square()` - ReLU squared

Not GELU, not SiLU. **ReLU²**: negatives clamped to zero, then everything squared. It is
cheap (no `erf`, no `exp`), it is smooth where it matters, and recent small-model work
found it trains at least as well as the fancier gates. There is no deep theory here - it is
an empirical choice, of exactly the kind the autonomous research loop is invited to
revisit. Swapping this one line is a legitimate experiment.

## The block

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

Two writers, one stream, pre-norm on the way into each (lesson 06). Note there is no
learned scaling on either addition - a block's contribution is added at full strength, and
if the model wants to attenuate it, it must learn to output smaller values.

## Zero init: the block starts as the identity function

Read `init_weights` carefully and one pattern jumps out:

    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)      # random
    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)      # random
    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)      # random
    torch.nn.init.zeros_(block.attn.c_proj.weight)            # ZERO
    torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)      # random
    torch.nn.init.zeros_(block.mlp.c_proj.weight)             # ZERO

**Every matrix that writes back into the residual stream starts at exactly zero.** So at
step 0, `attn(...)` returns zeros, `mlp(...)` returns zeros, and

    x = x + 0 = x

Every block is the identity. A 50-layer model and a 2-layer model produce the *same*
output at initialisation, and that output is the embedding, unchanged. The network starts
as a shallow model and each layer has to earn its contribution by growing its `c_proj` away
from zero.

This is not a trick to save a step of training; it is why very deep residual networks train
at all. At init, gradients reach every layer without being multiplied by a long chain of
random matrices, so nothing explodes and nothing vanishes. And note the gradient of a
zeroed `c_proj` is *not* zero - it is `grad_output * activation`, and the activation is
nonzero - so the layer starts learning immediately. Initialising `c_fc` to zero instead
would genuinely be dead.

The same idea appears twice more: `lm_head` is initialised to `std=0.001` (near-zero logits
means near-uniform predictions means loss `ln(V)`, exactly what lesson 03 predicted), and
the value-embedding gates are zeroed so that `2*sigmoid(0) = 1.0` - a neutral gate.

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
