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

## One width, three different jobs

`n_embd` (write it `n`) is the width of every vector travelling through the model - 128 numbers
at the course's scale. The width never changes. What those 128 numbers *mean* does:

1. **At the input**, `wte` hands each token id its own vector. Row 2 of `wte` is "what the model
   knows about token 2, on its own". Every occurrence of token 2 starts from the same 128
   numbers.
2. **Inside the blocks**, attention and the MLP keep adding to that vector (lesson 07). By the
   end, position 40's vector no longer describes token 40. It describes *the context up to
   position 40* - everything the model gathered that is useful for guessing what comes next.
3. **At the output**, `lm_head` turns that context vector into `vocab_size` scores, one per
   possible next token.

Same width all the way, but a token vector at the start and a context vector at the end.

## `ModuleDict` and `ModuleList`: containers PyTorch can see

Why `nn.ModuleDict({...})` and `nn.ModuleList([...])` instead of a plain `{...}` and `[...]`?

Because PyTorch finds a model's parameters by looking at the attributes of each `nn.Module`. It
recognises other modules and `nn.Parameter`s, and it looks *inside* `ModuleDict` and `ModuleList`.
It does **not** look inside ordinary Python dicts and lists. Measured on a toy module holding one
`nn.Linear(4, 4)` in a list and one in a dict:

    plain list and dict           model.parameters() finds 0 tensors
    ModuleList and ModuleDict     model.parameters() finds 4 tensors (2 weights, 2 biases)

With a plain list, the blocks would still run - but the optimizer would never see their weights,
`model.to(device)` would not move them, and the checkpoint would not save them. Nothing crashes;
the model just never learns. Here, `transformer.h` is a `ModuleList`, so all 26 parameter
tensors of the course model are registered.

## `wte` and `lm_head`: same shape, opposite jobs

Both are `(V, n)` matrices, with `V = vocab_size`. They get that shape for different reasons:

- `nn.Embedding(V, n)` is a table with one row of width `n` for each of the `V` tokens.
- `nn.Linear(n, V)` maps width `n` to width `V`. PyTorch stores a linear layer's weight as
  `(out_features, in_features)`, so that is also `(V, n)`: one row per output, one column per
  input.

A tiny example with `V = 4` tokens and `n = 3`:

    wte                                  lm_head
    token 0: [ 0.5, -0.1,  0.2]          row 0: [ 1.0,  0.0,  0.0]
    token 1: [ 0.0,  0.9, -0.3]          row 1: [ 0.0,  1.0,  0.0]
    token 2: [-0.4,  0.1,  0.8]          row 2: [ 0.0,  0.0,  1.0]
    token 3: [ 0.3,  0.3,  0.3]          row 3: [ 0.5,  0.5,  0.5]

**`wte` is a lookup.** The input token id `2` does no arithmetic at all - it just picks row 2:

    wte[2] = [-0.4, 0.1, 0.8]

**`lm_head` is a multiply.** Suppose the final context vector is `h = [0.2, -0.1, 0.9]`. Then
each logit is the dot product of `h` with one row:

    logit 0 = 1.0*0.2 + 0.0*(-0.1) + 0.0*0.9   =  0.2
    logit 1 = 0.0*0.2 + 1.0*(-0.1) + 0.0*0.9   = -0.1
    logit 2 = 0.0*0.2 + 0.0*(-0.1) + 1.0*0.9   =  0.9     highest: token 2 is the best guess
    logit 3 = 0.5*0.2 + 0.5*(-0.1) + 0.5*0.9   =  0.5

So one maps a token to a vector (read a row) and the other maps a vector to a score for every
token (dot product with every row). Opposite directions, same shape.

### Tied vs untied

Look at `lm_head` again: logit `i` is `h · row_i`, a measure of how closely `h` points in the
direction of row `i`. And `wte`'s row `i` is a vector describing token `i`. So there is a natural
idea: **use the same matrix for both** ("tied" embeddings, as GPT-2 did). Then

    logit_i = h · wte[i]

"Score token `i` by how similar the context vector is to token `i`'s own embedding." With the
example `wte` and the same `h`:

    logit 0 = 0.5*0.2  + (-0.1)*(-0.1) + 0.2*0.9    =  0.29
    logit 1 = 0.0*0.2  +   0.9*(-0.1)  + (-0.3)*0.9 = -0.36
    logit 2 = -0.4*0.2 +   0.1*(-0.1)  +   0.8*0.9  =  0.63     still token 2
    logit 3 = 0.3*0.2  +   0.3*(-0.1)  +   0.3*0.9  =  0.30

That saves `V * n` parameters. This model does **not** tie them - it keeps two separate `(V, n)`
matrices ("untied"). That costs an extra 1M parameters here, and buys two things: at this scale it
trains better (the "what a token is" and "what should come next" jobs get their own matrices), and
the two can learn at very different speeds. `setup_optimizer` gives `wte` a learning rate of `0.6`
and `lm_head` `0.004` - 150 times smaller (lesson 11).

## Parameters and buffers

Two different kinds of tensor live on the model.

**`nn.Parameter`** - a tensor marked "learned". `resid_lambdas` is created directly as one, and
every `nn.Linear` and `nn.Embedding` creates its weight as one. `model.parameters()` returns
them, the optimizer updates them, the checkpoint saves them.

**Buffers** - tensors that belong to the model but are *not* learned. `cos` and `sin` are
registered with:

    self.register_buffer("cos", cos, persistent=False)

- A buffer **moves with the model**: `model.to("cuda")` sends it to the GPU along with the weights,
  so it is always on the same device as the tensors it is used with.
- A buffer is **not optimised**: it does not appear in `model.parameters()`, so the optimizer never
  touches it.
- `persistent=False` additionally **leaves it out of the checkpoint**: `model.state_dict()` has no
  `cos` or `sin` key. There is no point saving a table that can be recomputed exactly from the
  config.

The rule of thumb: learned -> parameter; derived deterministically from the config -> buffer.

## Inside one block: why attention has four big matrices

At the course's scale: `n = 128`, `n_head = 2`, so `head_dim = 128 / 2 = 64`. Follow one token's
vector `x`, 128 numbers, through attention:

    c_q     weight (128, 128)     q = W_q @ x     (128, 128) @ (128,) -> (128,)
    c_k     weight (128, 128)     k = W_k @ x     -> (128,)
    c_v     weight (128, 128)     v = W_v @ x     -> (128,)

Each 128-vector is then split into 2 heads of 64 (the `view` from lesson 05). Each head runs its own
attention and produces a 64-number output. Then:

    head outputs    (64,) and (64,)
    concatenate  -> (128,)                         the two heads, side by side
    c_proj  weight (128, 128)     out = W_proj @ y   (128, 128) @ (128,) -> (128,)

Why a fourth matrix? After concatenation, features 0-63 came only from head 0 and features 64-127
only from head 1 - nothing has combined what the two heads found. `c_proj` mixes them into one
update for the residual stream (and, being the layer that writes to the stream, it is the one that
starts at zero, lesson 07). So attention needs `q`, `k`, `v` **and** an output projection: four
matrices of `n * n` each, `4n^2 = 65,536` numbers.

### Grouped-query attention: when K and V are narrower than Q

Nothing says keys and values need as many heads as queries. `n_kv_head` sets how many key/value
heads there are, and each can be **shared by several query heads**. Try `n = 128`, `n_head = 4`,
`n_kv_head = 2`, so `head_dim = 128 / 4 = 32`:

    c_q     (128, 128)    q -> 4 heads x 32 = 128 numbers
    c_k     ( 64, 128)    k -> 2 heads x 32 =  64 numbers
    c_v     ( 64, 128)    v -> 2 heads x 32 =  64 numbers
    c_proj  (128, 128)

`kv_dim = n_kv_head * head_dim = 64` is the **total** width of all key (or value) heads together.
Query heads 0 and 1 both use key/value head 0; query heads 2 and 3 both use key/value head 1. (The
CPU attention path does this with `repeat_interleave`, copying each K/V head twice.)

Q can be wider overall than K because the matching happens **per head**. Each score is one
32-number query dotted with one 32-number key, so what must agree is `head_dim` - never the total
widths. The saving: `c_k` and `c_v` are half the size, and so is the per-token key/value cache a
model keeps during generation. On that model the block matrices come to 360,512 instead of
393,280.

Every configuration in this course uses `n_kv_head == n_head`, so `kv_dim == n` throughout - but
the formulas below keep `kv_dim` so they stay right in general.

### The MLP: `n -> 4n -> n`

    c_fc     nn.Linear(n, 4n)    weight (4n, n)   = (512, 128)    4n * n = 4n^2 = 65,536
    c_proj   nn.Linear(4n, n)    weight (n, 4n)   = (128, 512)    n * 4n = 4n^2 = 65,536

Up to four times the width, then back down (lesson 07). Two matrices of `4n^2` each: `8n^2 =
131,072`.

### Why the MLP is two thirds of every block

With equal Q/K/V heads, one block holds:

    attention   c_q + c_k + c_v + c_proj  =  n^2 + n^2 + n^2 + n^2  =  4n^2
    MLP         c_fc + c_proj             =  4n^2 + 4n^2           =  8n^2
    total                                                           = 12n^2

    MLP share = 8n^2 / 12n^2 = 2/3

Each MLP matrix is as big as all four attention matrices put together. At `n = 128`: 131,072 of
196,608.

### Value embeddings and their gate

Some layers get a **value embedding**: an extra lookup table whose row for the current token is
mixed straight into `v` (lesson 05). Its shape is `(V, kv_dim)`: one row per token, and each row is
as wide as `v` itself - all the key/value heads together - because it is added directly to `v`
before `v` is split into heads:

    ve = value_embeds[str(i)](idx)           # (B, T, kv_dim)
    ve = ve.view(B, T, n_kv_head, head_dim)  # split into K/V heads, like v

A whole `(V, kv_dim)` table is as big as `wte`, which is why only some layers get one (next
section).

On those layers, `ve_gate` decides how much of it to mix in, separately for each key/value head:

    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))    # (B, T, n_kv_head)
    v = v + gate.unsqueeze(-1) * ve

`ve_gate` is `nn.Linear(32, n_kv_head)`, so its weight is `(n_kv_head, 32)` - here `(2, 32)`, just
64 numbers. It reads only the first 32 features of the block's input and produces **one number per
key/value head**: a small learned volume knob (starting at exactly 1, lesson 07), not another big
projection. Compare `c_v`'s 16,384.

## Which layers get value embeddings

    def has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

`% 2` gives `0` for even numbers and `1` for odd. `n_layer - 1` is the index of the last layer. So
the rule says: **a layer gets one if it is odd-or-even the same way as the last layer**. The last
layer always matches itself, so it always gets one, and then every second layer counting back from
it:

    n_layer = 2    last layer = 1 (odd)     ->  layer 1                 (layer 0 is even: no)
    n_layer = 3    last layer = 2 (even)    ->  layers 0, 2
    n_layer = 4    last layer = 3 (odd)     ->  layers 1, 3
    n_layer = 8    last layer = 7 (odd)     ->  layers 1, 3, 5, 7       (0, 2, 4, 6: no)

Alternating keeps the cost down to half the layers; anchoring on the last layer guarantees the
final layer - the one whose output is read out - always has direct access to token identity.

## Counting, on the course's actual model

The course's toy config is `n = 128`, `V = 8192`, `L = n_layer = 2`, `n_head = n_kv_head = 2`.

A linear layer from width `a` to width `b` is a `(b, a)` matrix: `a * b` numbers. An embedding
table for `V` tokens of width `n` is a `(V, n)` matrix: `V * n` numbers. That is all the arithmetic
you need.

**Inside one block:**

    attention   c_q      (128, 128)    128 * 128 =  16,384
                c_k      (128, 128)                 16,384
                c_v      (128, 128)                 16,384
                c_proj   (128, 128)                 16,384
    MLP         c_fc     (512, 128)    512 * 128 =  65,536
                c_proj   (128, 512)                 65,536
                                                   -------
                                                   196,608   per block

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

### Why embeddings dominate a small model - and stop dominating a big one

The tables grow like `V * n`: double the width, **twice** the parameters. The block matrices grow
like `n^2`: double the width, **four times** the parameters. At small `n` the vocabulary term wins;
as `n` grows the `n^2` term catches up and overtakes it.

    n_embd = 128, 2 layers     blocks = 11% of all parameters
    n_embd = 384, 6 layers     blocks = 40%

And at real GPT sizes (`n` in the thousands) the blocks are the overwhelming majority.

## The general table

With `n = n_embd`, `V = vocab_size`, `L = n_layer`, `head_dim = n / n_head`, and
`kv_dim = n_kv_head * head_dim`:

| What | Shape | Count | How many |
|---|---|---|---|
| `wte` | `(V, n)` | `V*n` | 1 |
| `lm_head` | `(V, n)` | `V*n` | 1 |
| `c_q` | `(n_head*head_dim, n)` | `n^2` | per layer |
| `c_k`, `c_v` | `(kv_dim, n)` | `n*kv_dim` each | per layer |
| `c_proj` (attn) | `(n, n)` | `n^2` | per layer |
| `c_fc` | `(4n, n)` | `4n^2` | per layer |
| `c_proj` (mlp) | `(n, 4n)` | `4n^2` | per layer |
| `value_embeds[i]` | `(V, kv_dim)` | `V*kv_dim` | per VE layer |
| `ve_gate` | `(n_kv_head, 32)` | `32*n_kv_head` | per VE layer |
| `resid_lambdas`, `x0_lambdas` | `(L,)` each | `2L` | 1 |

### A worked example, symbolically

Keep four quantities separate - mixing them up is the usual mistake:

1. **the number of VE layers** - how many layers `has_ve` picks;
2. **one VE table** - `V * kv_dim` parameters;
3. **one `ve_gate`** - `32 * n_kv_head` parameters, also once per VE layer;
4. **one layer's attention + MLP matrices** - `n^2 + 2*n*kv_dim + n^2 + 8n^2`, on *every* layer.

Take `L = 3`, with `n`, `V` and `n_head = n_kv_head` (so `kv_dim = n`) left as symbols.

    VE layers         has_ve(i, 3) for i = 0, 1, 2  ->  layers 0 and 2  ->  2 VE layers
    wte               V*n
    lm_head           V*n
    value_embeds      2 * (V*n)                                   (2 VE layers x one table)
    per-layer         n^2 + 2*n*n + n^2 + 8n^2 = 12n^2
    matrices          3 * 12n^2  +  2 * (32*n_kv_head)            (3 layers, plus 2 gates)
    scalars           2 * 3 = 6

Now plug in `n = 128`, `V = 8192`, `n_kv_head = 2`:

    wte + lm_head     2 * 8192 * 128                =   2,097,152
    value_embeds      2 * 8192 * 128                =   2,097,152
    matrices          3 * 12 * 128^2 + 2 * 64       =     589,952
    scalars                                                     6
    total                                               4,784,262

**Your turn** (do it on paper first, then check): `n_layer = 4`, `n_embd = 64`, `head_dim = 16`
(so `n_head = n_kv_head = 4`), `vocab_size = 1000`. Work out the number of VE layers, then all five
counts and the total. Check with:

    build_model(toy_config(n_layer=4, n_embd=64, head_dim=16, vocab_size=1000)).num_scaling_params()

## `estimate_flops`, and the 6N rule

A **FLOP** is one floating-point operation - one multiply or one add. Training cost is measured in
FLOPs *per token*, because that number times the tokens you train on is the total compute bill.

    return 6 * (nparams - nparams_exclude) + attn_flops

**Where the 6 comes from.** Take one linear layer with weight `W` of shape `(b, a)`. For every
token it:

- **forward**: computes `W x`. Each of the `b` outputs is a sum of `a` products, so about `a * b`
  multiplies and `a * b` adds. **About 2 FLOPs per parameter.**
- **backward**: computes two things - the gradient for the *input* (`W^T grad_out`, so the previous
  layer can keep backpropagating) and the gradient for the *weight* (`grad_out x^T`, so this layer
  can learn). Each is another matmul of the same size. **About 4 FLOPs per parameter.**

`2 + 4 = 6` FLOPs per parameter per token.

**Why it is only an approximation.** It counts matrix multiplies and nothing else. It ignores the
norms, residual adds, activations, softmax and rotary (all cheap per number, and - as lesson 16
shows - often slow for other reasons); it treats a sum of `a` products as exactly `2a` operations;
and it assumes backward is exactly twice forward. At transformer scale the multiplies dominate so
completely that `6N` lands close to the truth, which is why everyone uses it.

**What is excluded, and why.** `wte`, the value embeddings and the scalars are subtracted. An
embedding lookup is not a matrix multiply - it copies out one row - so it does no arithmetic that
scales with its parameter count: a table of 8,192 rows costs the same to read from as a table of 8.
`lm_head` is *not* excluded, even though it is the same shape as `wte`: as in the example above, it
really multiplies against all `V` rows.

**Why attention gets its own term.** The `q @ k.T` score computation multiplies two *activations*
together - it has no weights of its own, so it does not show up in any parameter count. And its
cost depends on something parameters know nothing about: how many earlier positions each token
looks at. So it is counted separately, as `12 * n_head * head_dim * min(window, sequence_len)` per
layer. That is where the sliding windows from lesson 05 appear: a short-window layer is literally
cheaper here.

With the course's numbers:

    6 * (3,539,012 - 1,048,576 - 1,048,576 - 4)   = 8,651,136
    layer 0 (window 64):   12 * 2 * 64 * 64       =    98,304
    layer 1 (window 128):  12 * 2 * 64 * 128      =   196,608
                                                   ---------
                                                   8,946,048 FLOPs per token

Lesson 14 divides by this number to compute MFU, the "how hard is the hardware working" figure in
the training log.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, toy_config
       model = build_model()
       model.config
       model.num_scaling_params()          # the breakdown you are about to predict
       sum(p.numel() for p in model.parameters())
       list(model.value_embeds.keys())     # which layers got one: ['1']
       model.estimate_flops()              # 8,946,048 FLOPs per token, fwd+bwd

       for name, p in model.named_parameters():
           print(name, tuple(p.shape))     # every tensor in the table above

       [n for n, _ in model.named_buffers()]            # ['cos', 'sin']
       "cos" in model.state_dict()                      # False: persistent=False

       wte, W = model.transformer.wte.weight, model.lm_head.weight
       wte.shape, W.shape                                # both (8192, 128)
       h = torch.randn(128)
       torch.allclose(W @ h, torch.stack([W[i] @ h for i in range(8192)]))   # logit i = W[i] . h

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
  `predict_param_counts(model.config)["total"]` should be `3,539,012`, and
  `predict_param_counts(toy_config(n_layer=3))["total"]` should be `4,784,262`.

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

`n_embd` is one width for three jobs: a token's own vector at the input, a context vector at the
output, and `lm_head` scoring every token against it. PyTorch only sees modules held in
`ModuleDict`/`ModuleList`; learned tensors are parameters, recomputable ones are buffers. A block
is four `n^2` attention matrices (fewer if K/V heads are shared) and two `4n^2` MLP matrices, so the
MLP is two thirds of it; value-embedding layers add a `(V, kv_dim)` table and a tiny per-head gate.
At the course's scale the `(V, n)` tables are 89% of the parameters, and as width grows the `n^2`
matrices take over. The same counting gives the training cost: about 6 FLOPs per non-embedding
parameter per token, plus a separate attention term that depends on the window. Next: running the
whole thing forward and turning its output into text.
