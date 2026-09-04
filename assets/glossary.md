# Glossary

Terms as this course uses them, with the lesson that introduces each.

**AdamW** (11) - the optimizer used for embeddings, the unembedding and the scalars. Keeps a
smoothed gradient and a smoothed squared gradient per coordinate, and divides one by the
square root of the other. The "W" is decoupled weight decay.

**Attention** (05) - the only operation in the model that moves information between token
positions. A softmax-weighted average of value vectors, where the weights come from
query-key dot products.

**Bits per byte (bpb)** (03, 13) - the course's quality metric.
`total_nats / (ln(2) * total_bytes)`. Comparable across tokenizers, unlike loss per token.
Literally a compression rate.

**BOS** (02) - beginning-of-sequence marker, `<|reserved_0|>`. Prepended to every document,
so every packed row starts with one.

**BPE** (01) - byte-pair encoding. Builds a vocabulary by repeatedly merging the most
frequent adjacent symbol pair, starting from raw bytes.

**Causal mask** (05) - forbids position `i` from attending to positions after it. What makes
one forward pass produce `T` independent training examples.

**Chinchilla ratio** (15) - roughly 20 training tokens per parameter for a compute-optimal
model. A fitted rule of thumb, not a law.

**Cross-entropy** (03) - `-log p(correct token)`, in nats. `ln(vocab)` for an untrained
model.

**FlashAttention** (17) - a CUDA kernel that computes exact attention in tiles, keeping the
`T x T` score matrix out of memory. Fast because of memory traffic, not FLOP count.

**Gradient accumulation** (12) - summing gradients over several micro-batches before one
optimizer step, to reproduce a large batch on small hardware. Exact, provided each
micro-batch's loss is divided by the number of micro-batches.

**Logits** (09) - the model's unnormalised scores, one per vocabulary entry per position.

**Micro-batch** (12) - one forward+backward pass, `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` tokens.

**MFU** (14) - Model FLOPs Utilisation: useful arithmetic divided by what the hardware could
have performed. Hardware-relative, unlike tokens per second.

**Muon** (11) - the optimizer used for the 2-D matrices inside the blocks. Replaces the
momentum-smoothed gradient with its nearest semi-orthogonal matrix, computed by Newton-Schulz
iterations rather than an SVD.

**Nats** (03) - units of the natural logarithm. Divide by `ln 2` for bits.

**Newton-Schulz iteration** (11) - an iterative scheme using only matrix multiplies to drive
a matrix's singular values toward 1. Muon's orthogonalisation; here the "polar express"
variant with five coefficient triples.

**Noise floor** (19) - the spread between two runs of *identical* code. The smallest
improvement you are entitled to believe.

**Packing** (02) - filling each row with whole documents chosen best-fit, cropping the
shortest when nothing fits. 100% utilisation, no padding.

**Pre-norm** (06) - normalising the input to each sub-layer rather than the residual stream
itself. `x = x + f(norm(x))`.

**ReLU²** (07) - `relu(x).square()`, this model's MLP activation.

**Residual stream** (04) - the running `(B, T, n_embd)` tensor every layer reads from and
adds to. Never overwritten, never changes shape.

**RMS norm** (06) - `x / sqrt(mean(x^2))`. No mean subtraction, no learned gain, no bias.

**RoPE / rotary embedding** (06) - injects position by rotating `q` and `k` by an angle
proportional to their index, so scores depend only on relative distance.

**Sliding window** (05) - restricting a layer's attention to the last `w` positions.
`window_pattern = "SSSL"` gives three short layers then one long, repeating.

**Softcap** (09) - `15 * tanh(logits / 15)`. Bounds logits smoothly, damping over-confidence.

**Time budget** (12, 18) - the fixed wall-clock training time every experiment gets. What
makes a speed change and a modelling change comparable on one axis.

**Untimed steps** (12) - the first few steps, excluded from the time budget so compilation
warm-up is not billed to training. 10 upstream, 2 in the CPU fork.

**Value embedding (VE)** (05, 08) - a second embedding table whose output is mixed into `v`
on alternating layers, gated per head.

**val_bpb** (13) - bits per byte on the pinned validation shard. The number every experiment
is judged by, and the one thing an experiment may not change.

**Zero init** (07) - initialising every matrix that writes into the residual stream to
exactly zero, making each block the identity at step 0.
