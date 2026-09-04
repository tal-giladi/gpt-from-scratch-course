# Module 05 - Measuring

**Lessons 13-15.** The metric, the log, and the arithmetic that says what a budget can buy.

## Why a whole module on measurement

Because the repo is a research loop, and a research loop is only as good as its metric. Two
of this module's three lessons are about not fooling yourself, which is the skill that
separates someone who can run experiments from someone who can trust them.

## Learning objectives

After this module a learner can:

1. Implement `val_bpb` exactly, including the special-token mask and the sum-then-divide
   structure, and match the repo's `evaluate_bpb` to six decimal places.
2. Explain why the validation shard is pinned and excluded from training, and how cheap and
   how fragile that defence is.
3. Explain why the metric is declared read-only, and what the legitimate exception (a
   declared, whole-fork change) looks like.
4. Read every field of a training log line and say what a change to the code would do to
   each.
5. Explain MFU, compute it, and say why it - and not `tok/sec` - tells you whether the fix
   is code or hardware.
6. Reproduce `estimate_flops` and explain the two terms, including why embeddings are
   excluded and `lm_head` is not.
7. Convert a time budget into tokens, compare it against the 20-tokens-per-parameter rule,
   and identify the vocabulary as the binding constraint at CPU scale.

## Dependencies

Modules 01-04. Lesson 13 uses the `reduction="none"` path from lesson 09's forward, and
lesson 14 uses the parameter counting from lesson 08.

## Misconceptions this module is written to break

- *"Lower loss is lower bpb."* Only for a fixed tokenizer. The whole point of a per-byte
  metric is that per-token numbers are not comparable.
- *"Averaging per-batch scores is fine."* It is not: batches cover different numbers of
  bytes.
- *"MFU is a vanity metric."* It is the only number that distinguishes "my code is
  inefficient" from "my machine is small".
- *"Bigger model, better results."* Not at a fixed budget. Lesson 15 does the arithmetic
  that shows the course's own configuration is ~1000x past compute-optimal.
- *"Scaling laws are a law."* Chinchilla's 20:1 is a fitted rule of thumb from one family
  of experiments, and modern practice deliberately departs from it for inference cost.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 13 | `bpb_over_batches` | your metric matches `evaluate_bpb` exactly over the same 16 validation batches, excludes special tokens, and sums rather than averages |
| 14 | `flops_per_token`, `mfu_percent` | your FLOPs estimate matches the model's over five configs, including the vocabulary and window-pattern edge cases |
| 15 | `budget_report` | you can convert a budget into tokens and compare it against compute-optimal, and you have seen the 35x gap on your own configuration |

## Time

About two and a half hours. Check 13 is the slowest in the course (16 forward passes,
roughly a minute).
