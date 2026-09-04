# Module 04 - Training

**Lessons 10-12.** From "the model produces a loss" to "the weights get better": gradients,
the two optimizers, and the loop's bookkeeping.

## Learning objectives

After this module a learner can:

1. Describe what `backward()` fills in, what it costs relative to the forward pass, and why
   activations - not parameters - usually set the batch size.
2. Explain gradient accumulation as a consequence of `.grad` being added to, and demonstrate
   both the feature and the bug it causes.
3. Show, on the real model, that a zero-initialised `c_proj` receives a gradient while the
   `c_fc` feeding it does not - and explain both.
4. Write an AdamW step matching the repo's fused kernel over several steps, and explain what
   each of the four lines is for, including decoupled decay.
5. Explain what Muon does differently, why orthogonalising an update helps, and why
   Newton-Schulz rather than an SVD.
6. Explain why the learning rates differ 150x between the embedding and the unembedding, and
   what `1/sqrt(width)` scaling buys.
7. Implement the learning-rate schedule, and explain why progress is measured in seconds.
8. Reproduce a large batch exactly with micro-batches, and identify the missing-division bug
   by its signature.

## Dependencies

Modules 01-03. Lesson 10's check reuses the initialisation facts from lesson 07.

## Misconceptions this module is written to break

- *"Gradients are computed fresh each backward pass."* They accumulate. This is deliberate
  and it is the mechanism behind micro-batching.
- *"Adam is just gradient descent with momentum."* The second moment - dividing by the
  square root of the smoothed squared gradient - is what makes it scale-invariant per
  coordinate, and that is the property that matters.
- *"The optimizer is a solved component you never touch."* This repo runs two at once, with
  five parameter groups and per-group learning rates spanning three orders of magnitude.
- *"Warmup is mandatory for transformers."* Not here, and the lesson says exactly which
  design choices replaced it.
- *"Gradient accumulation is an approximation of a big batch."* It is exact, provided the
  scaling is right.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 10 | `grad_report` | you can run a backward pass, measure a global gradient norm, and clear state correctly - and you have seen the zero-gradient surprise |
| 11 | `adamw_update`, `nesterov_momentum` | your AdamW tracks the repo's kernel over three steps on parameter, first and second moments; your momentum matches the closed form |
| 12 | `lr_multiplier`, `accumulated_grad_norm` | your schedule matches at seven points and three shapes; your accumulation matches a single big batch, and the check demonstrates the 4x bug you avoided |

## Time

About three hours. Lesson 11 rewards reading `muon_step_fused` line by line even though the
exercise only covers its momentum.
