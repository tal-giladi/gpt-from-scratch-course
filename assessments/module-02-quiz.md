# Module 02 quiz - The residual stream and attention

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** A colleague says "layer 5 transforms the output of layer 4." Correct them in two
sentences, and say what `x0_lambdas` adds to the picture.

**2.** You delete the line `q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)`
and retrain. The model still trains and the loss still falls. Explain what it can and cannot
learn now, and predict roughly where the loss will plateau relative to the original.

**3.** `window_pattern = "SSSL"` with `n_layer = 8` and `sequence_len = 2048`. What window
does each layer get? Which layer is the exception to the pattern, and why is it forced?

**4.** In `manual_attention` you mask with `float("-inf")` before the softmax. A colleague
suggests `-1e9` instead, "to avoid NaNs." When would `-1e9` give a materially different
answer, and when would `-inf` actually produce a NaN?

**5.** Attention is applied to `q`, `k` and `v`, but rotary embeddings are applied only to
`q` and `k`. Give the one-sentence reason, and say what would go wrong if `v` were rotated
too.

**6.** RMS norm here has no learned gain (`F.rms_norm(x, (x.size(-1),))` with no weight).
Show with an explicit 2-channel example why a gain in front of a bias-free linear layer
adds no expressive power. Then say whether the same argument covers `norm(q)` and `norm(k)`,
which do **not** feed a linear layer.

---

## Answers

**1.** Layer 5 reads the accumulated stream - the embedding plus every earlier layer's
contribution - and *adds* one more term to it; nothing is replaced. `x0_lambdas` makes this
explicit: at every depth, a fraction (0.1 at init) of the *original* normalised embedding is
mixed back in, so even a deep layer has cheap access to the raw token identity rather than
only to whatever the stack has accumulated.

**2.** Without rotary, attention scores are a pure function of *content*: `q_i · k_j` no
longer depends on `i - j` at all. The model becomes order-blind inside each attention layer
- it can still learn which tokens tend to co-occur, but not "the token immediately before
me" or "the matching open bracket". Loss will fall (unigram and bag-of-context statistics are
a lot of the signal) and then plateau substantially higher; at this course's scale expect a
visibly worse `val_bpb`, not a broken run. That failure mode - trains fine, just worse - is
exactly why silent position bugs survive so long.

**3.** `long = 2048`, `short = 1024`. Pattern repeated over 8 layers gives
S,S,S,L,S,S,S,L → layers 0,1,2,4,5,6 short (1024) and layers 3,7 long (2048). The exception
is the **last** layer: `window_sizes[-1] = (long_window, 0)` unconditionally. Here it already
was long, but with `n_layer = 6` it would not have been. It is forced because the final layer
is the one whose output is read out for the prediction; restricting its context caps what any
prediction can depend on, no matter what earlier layers gathered.

**4.** They differ when a row has *many* masked entries relative to unmasked ones: `-1e9`
still contributes `exp(-1e9 - max)` which underflows to 0 in float32, so in practice they
agree - until the scores themselves are large, or you are in float16 where `-1e9` overflows
to `-inf` anyway. The real difference is that `-inf` is exact by construction and needs no
reasoning about magnitudes. `-inf` produces a NaN only if an entire row is masked (softmax of
all `-inf` is `0/0`); in a causal mask that cannot happen because position `i` always sees
itself.

**5.** Position belongs to *where you look*, not to *what you carry back*: rotating `q` and
`k` makes the attention **score** depend on relative distance, which is the whole objective.
Rotating `v` would rotate the retrieved content itself, so the same information would come
back differently depending on where it was found - the model would have to learn to undo a
position-dependent rotation before it could use anything it retrieved.

**6.** A learned gain is a per-channel multiplier applied after the normalisation:

    y = rms_norm(x) * g

which is the same as multiplying by a diagonal matrix `G = diag(g)`, so `y = G x` writing
`x` for the already-normalised vector. The next layer is a bias-free linear map:

    z = W y = W (G x) = (W G) x

`W G` is just `W` with its **columns** rescaled: column `i` of `W` multiplied by `g[i]`. It
is another weight matrix of exactly the same shape, and `W` is free to be learned, so
whatever `G` could have contributed, `W` can express on its own.

*The 2-channel example.* Normalised input `x = [2, 3]`, learned gain `g = [10, 5]`, and a
next layer that computes `z = 2*x1 + 3*x2`, i.e. `W = [2, 3]`.

    with the gain:      G x = [20, 15]        z = 2*20  + 3*15  = 40 + 45 = 85
    without the gain:   W G = [20, 15]        z = 20*2  + 15*3  = 40 + 45 = 85

Identical, and not by coincidence — scaling input channel `i` by `g[i]` and scaling column
`i` of `W` by `g[i]` are the same multiplication, done in the other order.

The intuition: the gain lets the norm scale each input channel independently, but the linear
layer *already* holds an independent weight for every input channel. It can learn the same
scaling itself, for free, as part of weights it was going to learn anyway. So wherever
`norm(x)` feeds `c_q`, `c_k`, `c_v` or `c_fc`, a learned gain is extra parameters and an
extra kernel buying nothing.

*The subtlety.* This argument is about a gain that is immediately followed by a **free
linear map**. It does **not** generalise to "a gain before any dot product is redundant",
and in particular it does not cover this model's QK norm, where `norm(q)` and `norm(k)` feed
each other:

    score = (G q) · (G k) = sum_i g[i]^2 * q[i] * k[i]

There is no free weight matrix downstream to absorb `G` into - the next operation is a
bilinear product of two *normalised* vectors, then a scalar `1/sqrt(head_dim)`. Nor can it
be pushed upstream into `c_q`/`c_k`, because `rms_norm` sits in between and is non-linear:
rescaling a channel before the norm also changes the root-mean-square that divides it, so
the two do not commute. A gain there would be a genuine extra degree of freedom - a learned
diagonal metric weighting which channels matter in the attention score. The repo omits it
for simplicity and stability, not because it would be redundant.
