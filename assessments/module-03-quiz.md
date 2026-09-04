# Module 03 quiz - The rest of the model

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** At `n_embd = 128`, `vocab = 8192`, `n_layer = 2`, the model has ~3.5M parameters and
~89% of them are embedding tables. At `n_embd = 1024`, `vocab = 8192`, `n_layer = 24`, that
share collapses. Explain the scaling that causes it, in one sentence per term.

**2.** A colleague "simplifies" `init_weights` by removing the two `zeros_` calls and
initialising `c_proj` uniformly like every other matrix. Describe what changes at step 0,
and why the change is more dangerous in a 32-layer model than in a 2-layer one.

**3.** `lm_head` is a separate `(vocab, n_embd)` matrix rather than reusing `wte`. Give one
argument for tying them and one for not, and say which one this repo's learning-rate table
supports.

**4.** Why is `logits.float()` there, given that on CPU the model already runs in float32?

**5.** `softcap = 15`. A colleague raises it to 1000 "since it barely does anything anyway."
What is the largest probability ratio the model can express before and after? Is the change
safe?

**6.** Greedy decoding on a well-trained model produces `"the the the the"`. The model is
not broken. Explain, and name the two decoding knobs that fix it and what each trades away.

---

## Answers

**1.** Embedding tables are `vocab x n_embd` - **linear** in width. Block matrices are
`n_embd^2` per matrix and there are 12 of them per layer (`4n^2` attention + `8n^2` MLP),
so they are **quadratic** in width and linear in depth. Widen by 8x and deepen by 12x and
the matrices grow ~768x while the embeddings grow 8x, so the share flips.

**2.** At step 0 every block would add a random contribution to the residual stream instead
of zero, so the model is no longer the identity, and the output is a random function of the
input rather than the embedding. In a 2-layer model that is survivable noise. In a 32-layer
model the contributions compound: the stream's magnitude and the backward gradient are both
multiplied through 32 random maps, which is the classic exploding/vanishing regime. The zero
init is what makes depth safe *at initialisation* - which is exactly when a deep model is
most fragile.

**3.** For tying: it halves the parameters spent on `vocab x n_embd` and expresses the
intuition that "the vector that means token X" and "the vector that recognises token X"
should be related. Against: they are used in genuinely different ways (a sparse lookup vs a
dense projection over the whole vocabulary), and untied models train better at small scale.
This repo's table settles it - `wte` gets `lr = 0.6` and `lm_head` gets `0.004`, 150x apart.
Tied weights could not have two learning rates.

**4.** Because it is not there for CPU. It is there so the same code is correct under
`autocast(bfloat16)` on a GPU, where the logits would otherwise be bf16 and the softmax and
cross-entropy over 8192 categories would lose real precision - bf16 has ~3 decimal digits of
mantissa. On CPU in fp32 it is a no-op, and leaving it in is what makes the file run
correctly on both.

**5.** `15 * tanh(z/15)` bounds logits to `(-15, 15)`, so the largest expressible ratio
between two probabilities is `e^30 ≈ 1e13`. At 1000 it is effectively unbounded (`e^2000`).
The change is not obviously safe: the cap also damps gradients where the model is most
confident, which is a training-stability mechanism, not just a numerical guard. It is a
legitimate experiment, and it is exactly the kind that should be run and measured rather than
reasoned about.

**6.** Greedy always takes the argmax, and after "the the the" the highest-probability
continuation of a repeated phrase is more of the same phrase - it is a fixed point the
decoder cannot leave, even though the model's distribution has plenty of mass elsewhere.
**Temperature** > 0 reintroduces sampling (trading determinism and some coherence for
variety); **top-k** keeps the tail from being sampled at all (trading away rare-but-correct
continuations to avoid incoherent ones). Neither is a change to the model.
