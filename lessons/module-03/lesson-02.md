# 08 - Assembling a GPT, and where the parameters went

You now have every part. This lesson puts them in one box and then does something that
sounds boring and turns out to be revealing: it **counts the numbers the model learns**.

A *parameter* is one learned number - one entry of one weight matrix. When people say a model
"has 3.5 million parameters" they mean exactly that: add up the sizes of every weight tensor.
Knowing where those numbers sit tells you which knob makes the model bigger, what it costs to
train, and - in lesson 11 - which optimizer each group gets.

## The whole model, in one constructor

Here is everything `GPT.__init__` creates:

    self.transformer = nn.ModuleDict({
        "wte": nn.Embedding(config.vocab_size, config.n_embd),
        "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
    })
    self.lm_head       = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
    self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))
    self.value_embeds  = nn.ModuleDict({str(i): nn.Embedding(vocab_size, kv_dim)
                                        for i in range(n_layer) if has_ve(i, n_layer)})
    cos, sin = self._precompute_rotary_embeddings(...)   # buffers, not parameters

In plain words, from input to output:

| Part | What it is for | Lesson |
|---|---|---|
| `wte` | token id -> starting vector ("word token embedding") | 04 |
| `h` | the stack of `n_layer` blocks, each attention + MLP | 05-07 |
| `lm_head` | final vector -> one score per vocabulary token | 09 |
| `resid_lambdas`, `x0_lambdas` | one pair of learned dials per layer | 04 |
| `value_embeds` | extra per-layer token tables mixed into `v` | 05 |
| `cos`, `sin` | the rotary angle tables | 06 |

Five kinds of learned thing, plus one table that is not learned at all.

## Counting, on the course's actual model

The course's toy config is `n_embd = 128`, `vocab_size = 8192`, `n_layer = 2`, `n_head = 2`.
Write `n = 128`, `V = 8192`, `L = 2`.

A linear layer from width `a` to width `b` is a `(b, a)` matrix: `a * b` numbers. An embedding
table for `V` tokens of width `n` is a `(V, n)` matrix: `V * n` numbers. That is all the
arithmetic you need.

**Inside one block:**

    attention   c_q      (128, 128)    128 * 128 =  16,384
                c_k      (128, 128)                 16,384
                c_v      (128, 128)                 16,384
                c_proj   (128, 128)                 16,384
    MLP         c_fc     (512, 128)    512 * 128 =  65,536
                c_proj   (128, 512)                 65,536
                                                   -------
                                                   196,608   per block

Attention is `4 * n^2 = 65,536`; the MLP is `8 * n^2 = 131,072`. **The MLP is two thirds of
every block**, because its two matrices are each four times the size of an attention matrix.

Two blocks give `393,216`. Layer 1 also has a `ve_gate` of shape `(2, 32)` - 64 numbers - so
`transformer_matrices = 393,280`.

**Outside the blocks:**

    wte              (8192, 128)    8192 * 128 = 1,048,576
    lm_head          (8192, 128)                 1,048,576
    value_embeds[1]  (8192, 128)                 1,048,576
    resid_lambdas    (2,)                                2
    x0_lambdas       (2,)                                2

**Total: 3,539,012.** And look at the split:

    three (V, n) tables    3,145,728    89%
    all the blocks           393,280    11%
    scalars                        4

This model is mostly **lookup tables with a small transformer bolted on**. That is normal for
a tiny model, and it flips as width grows. Block matrices scale with `n^2` (double the width,
four times the parameters); embedding tables scale only with `n` (double the width, twice the
parameters). At `n_embd = 384, n_layer = 6` the blocks are already 40% of the model, and at
real GPT sizes they are the overwhelming majority.

## The general table

With `n = n_embd`, `V = vocab_size`, `L = n_layer`, `head_dim = n / n_head`, and
`kv_dim = n_kv_head * head_dim`:

| What | Shape | Count |
|---|---|---|
| `wte` | `(V, n)` | `V*n` |
| `lm_head` | `(V, n)` | `V*n` |
| `c_q` | `(n_head*head_dim, n)` | `n^2` |
| `c_k`, `c_v` | `(kv_dim, n)` | `n*kv_dim` each |
| `c_proj` (attn) | `(n, n)` | `n^2` |
| `ve_gate` (only on VE layers) | `(n_kv_head, 32)` | `32*n_kv_head` |
| `c_fc` | `(4n, n)` | `4n^2` |
| `c_proj` (mlp) | `(n, 4n)` | `4n^2` |
| `value_embeds[i]` (only on VE layers) | `(V, kv_dim)` | `V*kv_dim` |
| `resid_lambdas`, `x0_lambdas` | `(L,)` each | `2L` |

`kv_dim` only differs from `n` when `n_kv_head < n_head` (grouped-query attention, where
several query heads share one key/value head to save memory). Every configuration in this
course uses `n_kv_head == n_head`, so `kv_dim == n` throughout.

## Three things that surprise people

**`lm_head` is not the same matrix as `wte`.** Both are `(V, n)`, and GPT-2 used one matrix for
both jobs ("tied" embeddings): look a token up by row on the way in, score every token by the
same rows on the way out. This model keeps two separate matrices ("untied"). That costs an
extra `V * n` = 1M parameters here, and it buys two things: at this scale it trains better,
and the two can learn at very different speeds. `setup_optimizer` gives the input embedding a
learning rate of `0.6` and the output head `0.004` - 150 times smaller - before both are scaled
by `1/sqrt(n_embd/768)`.

**Value embeddings are expensive.** Each one is a whole `(V, kv_dim)` table - as big as `wte`
itself. That is why they are on every *other* layer, not all of them.

**`cos` and `sin` are not parameters.** They are registered with
`register_buffer(..., persistent=False)`. A *buffer* is a tensor that belongs to the model -
it moves to the GPU when the model does - but is never trained. `persistent=False`
additionally keeps it out of the saved checkpoint: there is no point storing a table that can
be recomputed exactly from the config. The rule of thumb: anything learned is a parameter;
anything *derived* from the config is a buffer.

## Which layers get value embeddings

    def has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

Read it as: "a layer gets one if it has the **same odd-or-even-ness as the last layer**". The
last layer always matches itself, so it always gets one, and then every second layer counting
backwards from it:

    n_layer = 2    last = 1 (odd)     ->  layer 1
    n_layer = 3    last = 2 (even)    ->  layers 0, 2
    n_layer = 4    last = 3 (odd)     ->  layers 1, 3
    n_layer = 8    last = 7 (odd)     ->  layers 1, 3, 5, 7

So the count is `ceil(n_layer / 2)`, but the hints suggest calling `has_ve` rather than
reasoning about parity - it is one less thing to get wrong.

## `estimate_flops`, and the 6N rule

A **FLOP** is one floating-point operation - one multiply or one add. Training cost is
measured in FLOPs *per token*, because that number times the tokens you train on is the
total compute bill.

    return 6 * (nparams - nparams_exclude) + attn_flops

**Where the 6 comes from.** Take one linear layer with weight `W` of shape `(b, a)`. For every
token it:

- **forward**: computes `W x` - about `a * b` multiplies and `a * b` adds. **2 FLOPs per
  parameter.**
- **backward**: computes two things - the gradient for the *input* (`W^T grad_out`, so the
  previous layer can keep backpropagating) and the gradient for the *weight* (`grad_out x^T`,
  so this layer can learn). Each is another matmul of the same size. **4 FLOPs per
  parameter.**

`2 + 4 = 6` FLOPs per parameter per token, for forward plus backward.

**What is excluded, and why.** `wte`, the value embeddings and the scalars are subtracted. An
embedding lookup is not a matrix multiply - it just copies out one row - so it does no
arithmetic that scales with its parameter count: a table of 8192 rows costs the same to read
from as a table of 8 rows. Note that `lm_head` is *not* excluded, even though it is the same
shape as `wte`: it really is a matmul, scoring the vector against all `V` rows.

**Why attention gets its own term.** The `q @ k.T` score computation has no weights of its own
- it multiplies two activations together - so it does not show up in any parameter count. Its
cost grows with how many keys each query looks at: `12 * n_head * head_dim * window` per layer.
That is where the sliding windows from lesson 05 appear, as `min(window, sequence_len)`: a
short-window layer is literally cheaper here.

With the course's numbers:

    6 * (3,539,012 - 1,048,576 - 1,048,576 - 4)   = 8,651,136
    layer 0 (window 64):   12 * 2 * 64 * 64       =    98,304
    layer 1 (window 128):  12 * 2 * 64 * 128      =   196,608
                                                   ---------
                                                   8,946,048 FLOPs per token

Lesson 14 divides by this number to compute MFU, the "how hard is the hardware working"
figure in the training log.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       model = build_model()
       model.config
       model.num_scaling_params()          # the breakdown you are about to predict
       sum(p.numel() for p in model.parameters())
       list(model.value_embeds.keys())     # which layers got one: ['1']
       model.estimate_flops()              # 8,946,048 FLOPs per token, fwd+bwd

       for name, p in model.named_parameters():
           print(name, tuple(p.shape))     # every tensor in the table above

       big = build_model(toy_config(n_layer=6, n_embd=384))
       big.num_scaling_params()            # watch the ratio shift from embeddings to matrices

2. Fill in `lab/exercises/lesson_08.py`: `predict_param_counts(config)` computes the same
   five numbers `num_scaling_params()` reports - **from the config alone**, without
   building a model.

3. Grade it:

       bash lab/lab.sh check 08

## Hints

- The five keys are exactly: `wte`, `value_embeds`, `lm_head`, `transformer_matrices`,
  `scalars`, plus `total` which is their sum.
- `transformer_matrices` is everything under `transformer.h` - that includes `ve_gate` on
  the layers that have one. Forgetting it is the single most likely way to be off by a
  small, confusing amount (64, at the course's scale).
- `has_ve` is importable: `train_defs().has_ve(i, n_layer)`. Count the VE layers with it
  rather than reasoning about parity yourself.
- `head_dim = config.n_embd // config.n_head`, and `kv_dim = config.n_kv_head * head_dim`.
- `ve_gate` is `nn.Linear(32, n_kv_head, bias=False)`, so `32 * n_kv_head` parameters. The
  32 is `ve_gate_channels`, a hardcoded constant in `CausalSelfAttention`.
- `scalars` is `2 * n_layer`.
- Check your function against the worked numbers above before running the grader:
  `predict_param_counts(model.config)["total"]` should be `3,539,012`.

## Solution

    from lib.common import train_defs

    def predict_param_counts(config):
        has_ve = train_defs().has_ve
        n, V, L = config.n_embd, config.vocab_size, config.n_layer
        head_dim = n // config.n_head
        kv_dim = config.n_kv_head * head_dim
        ve_layers = sum(1 for i in range(L) if has_ve(i, L))

        per_layer = (
            n * n                    # c_q
            + n * kv_dim             # c_k
            + n * kv_dim             # c_v
            + n * n                  # attn c_proj
            + 4 * n * n              # mlp c_fc
            + 4 * n * n              # mlp c_proj
        )
        counts = {
            "wte": V * n,
            "value_embeds": ve_layers * V * kv_dim,
            "lm_head": V * n,
            "transformer_matrices": L * per_layer + ve_layers * 32 * config.n_kv_head,
            "scalars": 2 * L,
        }
        counts["total"] = sum(counts.values())
        return counts

## Summary

A GPT is an embedding table, a stack of identical blocks, two tiny vectors of scalars, some
extra value-embedding tables, and an output matrix. At the course's scale the three
`(vocab, n_embd)` tables are 89% of the parameters; as width grows, the `n^2` block matrices
take over. The same counting gives the training cost: about 6 FLOPs per non-embedding
parameter per token, plus a separate attention term that depends on the window. Next:
running the whole thing forward and turning its output into text.
