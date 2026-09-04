# Module 02 - The residual stream and attention

**Lessons 04-06.** The structural idea the whole architecture rests on, and the only
operation in the model that moves information between positions.

## Why this order

The residual stream comes before attention because attention is defined by what it writes
*into* the stream. Position comes after attention because you cannot see why rotation is
needed until you have watched a dot product ignore token order.

## Learning objectives

After this module a learner can:

1. Explain what the residual stream is and why "every layer adds, nothing overwrites" is
   the structural claim that makes depth trainable and interpretability possible.
2. Re-implement `GPT.forward`'s block loop, including the two per-layer scalars, and prove
   it reproduces the model's own logits.
3. Write causal windowed attention from the four equations, with an explicit softmax, and
   match the repo's implementation to 1e-5.
4. Explain why the causal mask is what makes one forward pass train on `T` positions.
5. Explain rotary embeddings as rotation, and demonstrate that scores depend only on
   relative distance.
6. Say where RMS norm is applied and why pre-norm, and why it has no learned parameters.

## Dependencies

Module 01 - the exercises consume real batches, and lesson 04's check compares against the
loss path from lesson 03.

## Misconceptions this module is written to break

- *"Each layer transforms the previous layer's output."* It does not. Layer `i` reads the
  accumulated sum of everything before it and adds one more term.
- *"Attention is where the model does its thinking."* Attention moves information; the MLP
  (Module 03) does most of the arithmetic and holds most of the parameters.
- *"The mask is a safety feature so the model cannot cheat."* It is what makes efficient
  training possible at all - without it you would need one forward pass per position.
- *"Positional embeddings are added to the input."* Not here. Rotary rotates `q` and `k`
  inside every attention layer, and `v` never sees position at all.
- *"Normalisation normalises the activations."* It normalises the *copy* fed into each
  sub-layer. The residual stream itself is never normalised in place.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 04 | `residual_stream` | your forward loop reproduces `model(idx)` exactly through the final norm, `lm_head` and softcap |
| 05 | `manual_attention` | your four-line attention matches the repo's kernel at four different windows, and respects causality under a direct edit-the-future test |
| 06 | `apply_rotary`, `rms_norm` | your rotation matches the repo's, preserves norms, and makes dot products depend only on relative position |

## Time

About three hours. Lesson 05 is the densest in the course.
