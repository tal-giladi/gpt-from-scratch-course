# Module 03 - The rest of the model

**Lessons 07-09.** The other half of a block, the assembly of the whole model, and the exit
from tensors back to text.

## Learning objectives

After this module a learner can:

1. Implement the ReLU² MLP and a full block from the equations, matching the real modules.
2. Explain why every matrix that writes into the residual stream is initialised to zero, and
   what that makes the model equivalent to at step 0.
3. Predict a model's parameter breakdown from its config alone, including value embeddings
   and the gate matrices, and match `num_scaling_params()` exactly.
4. Explain why `lm_head` is untied from `wte` here, and why the embedding tables dominate a
   small model.
5. Describe the four operations at the end of the forward pass - final norm, `lm_head`,
   float32 upcast, softcap - and what each is for.
6. Turn logits into text with temperature, top-k or greedy decoding, and explain why none of
   it is part of the model.

## Dependencies

Modules 01-02. Lesson 09's sampling builds directly on lesson 03's softmax and lesson 04's
forward loop.

## Misconceptions this module is written to break

- *"Attention is most of a transformer."* Two thirds of every block's parameters are in the
  MLP.
- *"Zero-initialised weights are dead weights."* `c_proj`'s gradient depends on its input,
  not on itself. (Module 04 completes this: `c_fc`'s gradient *is* zero on step 0, and
  recovers on step 1.)
- *"Deeper models are harder to train because of vanishing gradients."* At init this model
  is exactly the identity at every depth, so gradients reach every layer unattenuated. The
  problem was solved by initialisation, not by architecture.
- *"Softmax outputs are the model's confidence."* They are, after a temperature you chose,
  a top-k you chose, and a softcap the architecture applied. Every one of those is a
  decision outside the weights.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 07 | `mlp_forward`, `block_forward` | your MLP and block match the real modules, and you can demonstrate the identity-at-init property and then break it |
| 08 | `predict_param_counts` | you can derive the full parameter breakdown from a config, for four different configs |
| 09 | `next_token_probs`, `greedy_generate` | you can sample correctly: distributions sum to 1, temperature and top-k never change the argmax, greedy is deterministic |

## Time

About three hours.
