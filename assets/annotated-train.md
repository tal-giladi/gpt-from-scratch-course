# `train.py`, annotated

The lessons teach the ideas in the order they build on each other. This document does the
other thing: walks `train.py` **top to bottom, in file order**, so you can sit with the
actual file open and know what every block is for and how it connects to the rest.

Read the map first. Then the walkthrough. Then the file itself, and it should feel like
re-reading something you already know.

---

## The whole thing on one page

Six stages. Data comes in as text, weights come out as a checkpoint, and one number comes
out as the score.

    prepare.py                        train.py
    ----------                        --------
    text shards (parquet)
        |
        v  BPE, 8192 tokens
    token ids
        |
        v  best-fit packing, BOS per row
    (B, T+1) rows  ->  x = row[:-1], y = row[1:]
        |
        |                             GPT.forward
        +---------------------------> wte lookup ....................... (B,T,n_embd)
                                          |
                                          v  norm, keep a copy as x0
                                      +-- residual stream ------------------+
                                      |   per block, n_layer times:         |
                                      |     x = resid_l*x + x0_l*x0         |
                                      |     x = x + attn(norm(x), ve, rope) |  <- moves info
                                      |     x = x + mlp(norm(x))            |  <- thinks
                                      +-------------------------------------+
                                          |
                                          v  norm, lm_head, softcap
                                      logits (B,T,vocab)
                                          |
                                          v  cross-entropy against y
                                      loss  ->  backward()  ->  MuonAdamW.step()
                                          |
                                          v  when the time budget runs out
    evaluate_bpb  <---------------------- val_bpb, checkpoint

Three invariants worth holding on to while you read:

1. **The residual stream never changes shape.** `(B, T, n_embd)` from the embedding to the
   final norm. Every layer adds to it; nothing replaces it.
2. **Attention is the only operation that moves information between positions.** Everything
   else is per-position arithmetic.
3. **The loss is computed at every position at once**, which is only sound because the
   causal mask stops position `i` from seeing position `i+1`.

---

## The walkthrough

### Lines 1-30: imports and the kernel

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

Set before torch is imported, because the allocator reads it once at import. It reduces GPU
memory fragmentation; it does nothing on CPU.

    USE_FLASH = (not IS_CPU) and _env_flag("AR_FLASH", True)
    ...
    fa3 = get_kernel(repo).flash_attn_interface

Upstream this is unconditional: FlashAttention-3 is downloaded from the `kernels` hub as a
compiled CUDA kernel. The fork made it optional and falls back to `attention()` below
(lesson 17). `get_device_capability() == (9, 0)` is Hopper - a different build of the same
kernel is used on other GPUs.

    from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

The only import from the fixed half of the repo. `prepare.py` owns the data, the tokenizer,
the sequence length, the time budget and the metric - all the things an experiment is not
allowed to change.

### Lines 33-45: `GPTConfig`

Everything about the model's shape, and nothing about its training. Note `window_pattern`,
a string like `"SSSL"` - four characters that decide the attention cost of every layer.

### Lines 47-58: two helpers

    def norm(x):
        return F.rms_norm(x, (x.size(-1),))

No parameters. Called on the embedding, twice per block, and on `q`/`k`. (Lesson 06.)

    def has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

Which layers get a value embedding: alternating, arranged so the last layer always does.

    def apply_rotary_emb(x, cos, sin):

The rotation that injects position. Pairs the **first half** of the head dimension with the
**second half**, not neighbouring elements.

### Lines 61-96: `CausalSelfAttention`

    self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
    self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
    self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
    self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

Four matrices, no biases anywhere in this model. `n_kv_head < n_head` would be grouped-query
attention (fewer key/value heads than query heads, a memory saving at inference); every
config here sets them equal.

    self.ve_gate = nn.Linear(32, n_kv_head, bias=False) if has_ve(...) else None

A tiny gate that reads the **first 32 channels** of the stream and produces one number per
head. `2 * sigmoid(...)` puts it in `(0, 2)` with 1.0 as neutral, and it is zero-initialised
so it starts exactly neutral.

    v = v + gate.unsqueeze(-1) * ve

The value residual: mix a per-layer embedding of the raw token into `v`. This gives
attention direct access to token identity, not just to the processed stream.

    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)

Position first, then QK norm. `v` gets neither - it carries content, not position.

    y = attention(q, k, v, window_size)

One call, two possible implementations (FA3 or masked SDPA), identical outputs to ~1e-6.

### Lines 99-121: `MLP` and `Block`

    x = F.relu(x).square()

ReLU squared. Cheap, no exponentials, empirically as good as the gated alternatives at this
scale. A prime candidate for an experiment.

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))

The entire block. Pre-norm on the way in, raw addition on the way out.

### Lines 124-147: `GPT.__init__`

    self.rotary_seq_len = config.sequence_len * 10
    self.register_buffer("cos", cos, persistent=False)

Rotary tables for ten times the training context, precomputed once. `persistent=False` keeps
them out of `state_dict` - they are derived from the config, so saving them would be storing
a cache.

### Lines 149-182: `init_weights`

The single most load-bearing 30 lines in the file.

    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

Embeddings at unit scale (they are immediately normalised anyway); the unembedding almost
zero, so the initial logits are near-zero, the initial distribution is near-uniform, and the
initial loss is `ln(vocab)`.

    s = 3**0.5 * n_embd**-0.5
    torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
    torch.nn.init.zeros_(block.attn.c_proj.weight)
    torch.nn.init.zeros_(block.mlp.c_proj.weight)

`s` makes the uniform distribution have variance `1/n_embd`, so a matmul preserves scale.
And **every matrix that writes into the residual stream starts at zero**, so at step 0 every
block is the identity and the model is exactly its embedding table. (Lessons 07 and 10.)

    self.resid_lambdas.fill_(1.0)
    self.x0_lambdas.fill_(0.1)

Pass the stream through unchanged; re-inject 10% of the original embedding at every depth.

### Lines 209-234: `estimate_flops` and `num_scaling_params`

`6 * (params - embeddings) + attention` - two FLOPs per parameter forward, four backward,
embeddings excluded because a lookup is not arithmetic. (Lesson 14.)

### Lines 236-266: `setup_optimizer`

    dmodel_lr_scale = (model_dim / 768) ** -0.5

Learning rates scale as `1/sqrt(width)`, calibrated at 768. This is what makes width a
one-line experiment instead of a re-tuning project.

    for shape in sorted({p.shape for p in matrix_params}):

Muon groups are formed **by tensor shape**, because `_step_muon` stacks a group into one
tensor and orthogonalises the whole stack in a single batched call.

### Lines 268-300: `GPT.forward`

    x = self.transformer.wte(idx)
    x = norm(x)
    x0 = x
    for i, block in enumerate(self.transformer.h):
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i])
    x = norm(x)
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

Twelve lines. That is the whole model. (Lessons 04 and 09.)

    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1, reduction=reduction)

`reduction` is a parameter so that `evaluate_bpb` can ask for per-position losses instead of
a mean.

### Lines 305-354: the fused optimizer kernels

Two `@torch.compile(fullgraph=True)` functions - the only compiled code besides the model.
`adamw_step_fused` is textbook AdamW with decoupled decay; `muon_step_fused` is Nesterov
momentum, then five polar-express Newton-Schulz iterations to orthogonalise the update, then
NorMuon variance reduction, then cautious weight decay. (Lesson 11.)

    self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

0-D tensors instead of Python floats, so changing a learning rate does not trigger a
`torch.compile` recompilation.

### Lines 429-455: the hyperparameters

Everything an experiment is meant to touch, in one block, with no CLI flags. `program.md`
tells the agent to edit these values directly - the file *is* the config.

### Lines 457-512: setup

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

Build the model on the **meta device** - shapes only, no memory - then allocate uninitialised
storage on the real device and fill it. This skips PyTorch's default initialisation entirely,
which for a large model is seconds of pointless random number generation.

### Lines 514-530: schedules

`get_lr_multiplier`, `get_muon_momentum`, `get_weight_decay` - all pure functions of
progress or step. (Lesson 12.)

### Lines 537-600: the training loop

    while True:
        sync(); t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)
        ... set lr/momentum/weight-decay from progress ...
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL"); exit(1)
        if step > UNTIMED_STEPS: total_training_time += dt
        if step > UNTIMED_STEPS and total_training_time >= TIME_BUDGET: break

Note the ordering: the next batch is fetched *inside* the accumulation loop, right after
`backward()`, so the dataloader's work overlaps with the GPU's. And the loop exits on
**elapsed time**, not step count.

    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()

Python's garbage collector caused ~500 ms stalls. `freeze()` moves everything currently
alive into a permanent generation, then collection is turned off and run manually every 5000
steps. A profiling result, embedded as three lines of code.

### Lines 604-630: the summary

    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

The only call to the metric, at the very end, on the pinned validation shard. Then the
summary block that `program.md` greps, and (in this fork only) the checkpoint.

---

## What to change first, if you want to learn by breaking it

| Change | What you should see | Lesson |
|---|---|---|
| `torch.nn.init.zeros_(block.mlp.c_proj.weight)` -> `uniform_` | trains worse or diverges; the identity-at-init property is gone | 07 |
| remove `q, k = norm(q), norm(k)` | less stable, especially at higher LR | 05 |
| `softcap` 15 -> 1000 | effectively no cap; watch for confident, spiky logits | 09 |
| `x0_lambdas.fill_(0.0)` | no embedding re-injection; usually slightly worse | 04 |
| `window_pattern = "L"` | more FLOPs per token, fewer steps in the budget | 14 |
| `apply_rotary_emb` returning `x` unchanged | position information gone; loss plateaus badly | 06 |

Each one is a single line, a three-minute run, and a number you can defend.
