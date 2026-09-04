# Course outline

**GPT from scratch: reading, running and improving a real pretraining stack** - 7 modules,
20 lessons, 20 graded exercises, one capstone. Roughly 15-20 hours if you do the exercises,
which is the only way this course works.

## Audience and prerequisites

See [`brief/course-brief.md`](../brief/course-brief.md). In short: an experienced software
engineer with no ML background, Python and Docker on their machine, and high-school linear
algebra.

Required: Docker Desktop, Git Bash (or any bash), and a clone of the CPU fork of
`autoresearch` next to this course. Setup is one command - `bash lab/lab.sh up` - which
verifies the image, the data and the tokenizer are in place. There is nothing to install.

## Design principles

1. **Graded against the real repository.** No lesson is "done" because it was read.
   `bash lab/lab.sh check NN` imports the actual `train.py` from `/autoresearch` and asserts
   your code agrees with it - the same tensors, to 1e-5.
2. **The whole file, in two passes.** The modules teach the ideas in dependency order; the
   [annotated walkthrough](../assets/annotated-train.md) reads the file top to bottom. Doing
   both is how "I understand each line" and "I understand the context" become the same
   thing.
3. **One fixed toy scale.** 2 layers, 128 dimensions, 128 tokens of context, the real
   tokenizer, real text. Every check runs in seconds; nothing is a mock.
4. **Concepts arrive when the code needs them.** Tokens before batches; batches before the
   prediction target; the residual stream before attention; attention before position;
   position before the block; the model before the optimizer; the metric before the
   experiment.
5. **The engineering counts as much as the maths.** Why the row capacity is `T + 1`, why
   every output projection starts at zero, why 0-D tensors are passed to a compiled kernel,
   why the garbage collector is turned off at step 0. These are not trivia; they are most of
   what separates a working training stack from a correct one.
6. **Honest about scale.** The final module measures the noise floor and shows this fork is
   ~1000x short of compute-optimal, so the learner knows exactly how much to believe.

## Module map

| # | Module | Lessons | You can, afterwards |
|---|---|---|---|
| 01 | From text to tensors | 01-03 | Explain BPE, packing and the next-token objective, and predict a model's loss at initialisation |
| 02 | The residual stream and attention | 04-06 | Re-implement the model's forward loop, its attention, its rotary embedding and its norm |
| 03 | The rest of the model | 07-09 | Build a block, count a model's parameters from its config, and turn logits into text |
| 04 | Training | 10-12 | Explain backprop's cost, write an AdamW step and Muon's momentum, and reproduce a big batch exactly |
| 05 | Measuring | 13-15 | Compute `val_bpb`, read a training log, and say what a fixed budget can buy |
| 06 | Speed, and reading a port | 16-17 | Benchmark your own machine and explain what FlashAttention does and what replacing it costs |
| 07 | Research | 18-20 | Run the autonomous loop, measure a noise floor, and defend one honest experiment |

## Lesson map

**Module 01 - From text to tensors**
1. Bytes, tokens, and a vocabulary of 8192 - BPE, why per-byte metrics exist
2. Packing documents into rectangles - best-fit packing, BOS, 100% utilisation
3. What the model is actually asked to do - the shift, cross-entropy, bits per byte

**Module 02 - The residual stream and attention**
4. Embeddings and the residual stream - the one tensor everything writes to
5. Attention: queries, keys, values, and the mask - the only cross-position operation
6. Where am I? Rotary embeddings, and why RMS norm - position as rotation, pre-norm

**Module 03 - The rest of the model**
7. The MLP, the block, and the zero that makes training work - ReLU², identity at init
8. Assembling a GPT, and where the parameters went - the shape table, the 6N rule
9. From logits to text - softcap, temperature, top-k, greedy

**Module 04 - Training**
10. Loss, gradients, and what backward() actually does - accumulation, the zero-grad surprise
11. Two optimizers in one model: AdamW and Muon - orthogonalised updates, the LR table
12. Schedules, the time budget, and faking a big batch - time-driven progress, accumulation

**Module 05 - Measuring**
13. val_bpb: the one number that decides everything - the frozen metric and why it is frozen
14. Reading the training log - tok/sec, MFU, and where the FLOPs go
15. What a fixed budget actually buys - Chinchilla, and the vocabulary floor

**Module 06 - Speed, and reading a port**
16. Why it is slow: dtypes, kernels, and torch.compile - measured on your machine
17. FlashAttention, and reading a port for what it gave up - tiling, masks, honest trades

**Module 07 - Research**
18. The autonomous research loop - program.md, hill climbing with git as memory
19. Designing an experiment you can believe - noise floors, one variable, equal budgets
20. Capstone: one honest experiment - three runs, a hypothesis, a defensible conclusion

## Assessment

- **20 graded exercises**, one per lesson, run with `bash lab/lab.sh check NN`. They assert
  measurable outcomes - "your attention matches the repo's to 1e-5", "your accumulated
  gradient equals the single-batch gradient", "your parameter formula matches the model" -
  not "does it run".
- **7 quizzes**, one per module, in [`assessments/`](../assessments/), weighted toward
  diagnosis and judgement rather than recall. Full answer keys.
- **The capstone** (Lesson 20) is the exam: three real training runs, a hypothesis written
  in advance, and a conclusion the check will reject if the data does not support it.

## What this course does not cover

Distributed training, inference serving and KV caching, fine-tuning, RLHF, quantization,
mixture-of-experts, and multimodality. It teaches one single-device pretraining stack
completely. Lesson 20 closes with where to go next, and the sibling
[abliteration course](https://github.com/tal-giladi/abliteration-course) picks up on the
other side - what you can do to a model's weights once it is trained.
