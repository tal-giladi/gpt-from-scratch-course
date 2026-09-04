# 08 - Assembling a GPT, and where the parameters went

You now have every part. Here is the whole model, as `GPT.__init__` builds it:

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

Five kinds of thing, and the difference between them decides how they are *optimised*
(lesson 11), so it is worth being able to count each one.

## The shapes, one at a time

With `n = n_embd`, `V = vocab_size`, `L = n_layer`, and `head_dim = n / n_head`:

| What | Shape | Count |
|---|---|---|
| `wte` | `(V, n)` | `V*n` |
| `lm_head` | `(V, n)` | `V*n` |
| `c_q` | `(n_head*head_dim, n)` | `n^2` |
| `c_k`, `c_v` | `(n_kv_head*head_dim, n)` | `n*kv_dim` each |
| `c_proj` (attn) | `(n, n)` | `n^2` |
| `ve_gate` (only on VE layers) | `(n_kv_head, 32)` | `32*n_kv_head` |
| `c_fc` | `(4n, n)` | `4n^2` |
| `c_proj` (mlp) | `(n, 4n)` | `4n^2` |
| `value_embeds[i]` (only on VE layers) | `(V, kv_dim)` | `V*kv_dim` |
| `resid_lambdas`, `x0_lambdas` | `(L,)` each | `2L` |

`n_kv_head == n_head` in every configuration this course uses, so `kv_dim == n`, and each
block's matrices come to `4n^2` for attention and `8n^2` for the MLP - **the MLP is two
thirds of every block.**

Three things that surprise people the first time:

- **`lm_head` is not tied to `wte`.** GPT-2 shared one matrix for both; this model has two
  separate `(V, n)` matrices. Untied costs `V*n` extra parameters and trains better at this
  scale - and it lets the two be optimised with different learning rates, which
  `setup_optimizer` does (0.6 for the embedding, 0.004 for the unembedding).
- **The embeddings dominate a small model.** At the course's scale (`n=128`, `V=8192`,
  `L=2`), the embedding tables are ~2.1M parameters and every matrix in every block adds up
  to ~0.4M. The model is mostly a lookup table with a small transformer bolted on. This
  flips as `n` grows: matrices scale with `n^2`, embeddings only with `n`.
- **`cos`/`sin` are buffers, not parameters.** `register_buffer(..., persistent=False)`
  means they move with the model to a device, are not trained, and are not saved in the
  checkpoint - they are recomputed. Anything derived deterministically from the config
  belongs here, not in `state_dict`.

## Which layers get value embeddings

    def has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

Alternating layers, arranged so that the **last** layer always has one. With `n_layer=8`:
layers 1, 3, 5, 7. With `n_layer=2`: layers 1 (and not 0). Each one costs a whole extra
`(V, kv_dim)` embedding table, which is why they are on every *other* layer and not all of
them.

## `estimate_flops`, and the 6N rule

    return 6 * (nparams - nparams_exclude) + attn_flops

The famous rule of thumb: a forward-plus-backward pass costs about **6 FLOPs per parameter
per token** - 2 for the forward multiply-accumulate, 4 for the backward pass, which
computes two gradients (one for the input, one for the weight). Embeddings are excluded
because a lookup is not a matmul: it does no arithmetic proportional to `n_embd`.

The attention term is separate because it does not scale with parameters at all - it scales
with sequence length, which is why the sliding windows in lesson 05 show up here as
`min(window, t)`. Lesson 14 uses this number to compute MFU.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       model = build_model()
       model.config
       model.num_scaling_params()          # the breakdown you are about to predict
       sum(p.numel() for p in model.parameters())
       list(model.value_embeds.keys())     # which layers got one
       model.estimate_flops()              # FLOPs per token, fwd+bwd

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
  small, confusing amount.
- `has_ve` is importable: `train_defs().has_ve(i, n_layer)`. Count the VE layers with it
  rather than reasoning about parity yourself.
- `head_dim = config.n_embd // config.n_head`, and `kv_dim = config.n_kv_head * head_dim`.
- `ve_gate` is `nn.Linear(32, n_kv_head, bias=False)`, so `32 * n_kv_head` parameters. The
  32 is `ve_gate_channels`, a hardcoded constant in `CausalSelfAttention`.
- `scalars` is `2 * n_layer`.

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

A GPT is an embedding table, a stack of identical blocks, two tiny scalar vectors, and an
output matrix - and at small scale the two `(vocab, n_embd)` tables outweigh everything
else. Counting them by hand is the fastest way to know which knob changes the model's size,
and the same arithmetic drives the FLOPs estimate the training log reports. Next: running
the whole thing forward and turning its output into text.
