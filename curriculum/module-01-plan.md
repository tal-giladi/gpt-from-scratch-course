# Module 01 - From text to tensors

**Lessons 01-03.** The data path, end to end: how text becomes the integers a model trains
on, and what it is asked to predict from them.

## Why this module is first

Every later module manipulates `x` and `y`. Starting anywhere else means handling tensors
whose provenance is a mystery, and it hides the two decisions that constrain everything
downstream: the vocabulary size (which sets the output layer's cost and the model's floor
size) and the packing scheme (which sets what a "training example" even is).

## Learning objectives

After this module a learner can:

1. Explain why BPE exists, and what changes when the vocabulary grows or shrinks.
2. Describe how documents of arbitrary length become a `(B, T)` rectangle at 100%
   utilisation, and implement the packing rule.
3. State the training objective precisely, and predict a model's loss at initialisation
   from its vocabulary size alone.
4. Convert between nats per token and bits per byte, and say why only the second is
   comparable across models.

## Dependencies

None beyond Python and the lab being up. This module is the entry point.

## Misconceptions this module is written to break

- *"The tokenizer is a preprocessing detail."* It sets the size of `lm_head`, which is about
  half the FLOPs of a small model, and it makes per-token loss numbers incomparable between
  models. Lesson 15 later shows it is the binding constraint on model size at CPU budgets.
- *"Padding is how you batch text."* This repo does not pad at all. Seeing an alternative
  makes the cost of padding visible.
- *"One row is one training example."* A row of `T` tokens is `T` training examples, all
  computed in one forward pass - which is why the causal mask in Module 02 matters so much.
- *"Loss is just a number that should go down."* It has a known value at initialisation
  (`ln(vocab)`), and knowing it turns "is training working?" from a feeling into a check.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 01 | `token_stats` | you can drive the real tokenizer and count bytes, not characters |
| 02 | `pack_row` | you can implement best-fit packing exactly, including the cropping branch |
| 03 | `split_row`, `uniform_loss`, `bits_per_byte` | you can state the objective and the metric in code |

Check 03 closes the loop by asserting that a real untrained model's loss on real data equals
your `uniform_loss(8192)`.

## Time

About two hours, including the shell exploration in each lesson.
