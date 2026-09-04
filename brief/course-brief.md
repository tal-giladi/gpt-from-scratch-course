# Course brief

## The learner

A senior software engineer with no ML background. They have shipped production systems for
years, they use LLM APIs and agents daily, and they can read any codebase you put in front
of them - but they have never trained a model, never opened a transformer's internals, and
have not written PyTorch. They found [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch),
asked whether it could run on their CPU inside docker, got it running as a
[CPU fork](https://github.com/tal-giladi/autoresearch), and then asked the real question:
they want to understand every line of it and become an AI engineer who can build and improve
models.

Assumed: Python, Git, Docker, and comfort with a terminal. High-school linear algebra - a
dot product, a matrix product, a vector norm. Nothing else. Not assumed: PyTorch, calculus
beyond "the derivative tells you which way is uphill", any prior exposure to transformers,
any statistics.

## The outcome

By the end they can open `train.py` - 700 lines, one file, a complete modern pretraining
stack - and explain what every block is for, what breaks without it, and which parts are
deliberate research choices rather than boilerplate. Concretely they will have written, from
scratch and checked against the real implementation:

the tokenizer statistics, the best-fit packer, the input/target shift, the residual-stream
loop, causal windowed attention with an explicit softmax, rotary embeddings, RMS norm, the
ReLU² MLP, a block, the parameter-count formula, the forward pass to logits, temperature and
top-k sampling, a gradient report, an AdamW step, Muon's Nesterov momentum, the
learning-rate schedule, gradient accumulation, the bits-per-byte metric, the FLOPs and MFU
calculation, a matmul benchmark, the sliding-window mask, the run-log parser, and an
experiment validity checker.

Then they run a real controlled experiment - baseline, repeat, variant - measure their own
noise floor, and defend or retract a hypothesis against it.

## The shape

**Graded against the real repository, not a teaching copy.** Every check runs inside the
same docker image the fork trains in, with the real `train.py` mounted at `/autoresearch`
and the real tokenizer and data shards from the shared cache. When a lesson says "this
matches `apply_rotary_emb`", the check imports the actual function and compares tensors. If
the learner edits the repo, the course follows the edit.

**Twenty graded exercises**, one per lesson, run as `bash lab/lab.sh check NN`. A lesson is
not complete because it was read.

**No second toolchain.** No venv, no host Python, no second torch download. The lab reuses
the CPU image the learner has already built, so `lab.sh up` is a verification step rather
than an installation.

**One fixed toy scale throughout**: 2 layers, 128 dimensions, 128 tokens of context, the
real 8192-token tokenizer, real text. Small enough that every check runs in seconds on a
CPU; real enough that nothing is a mock.

**Every mechanism explained before it is used, and the engineering explained alongside the
maths.** Why the row capacity is `T + 1`. Why every output projection is initialised to
zero and what that does to the first backward pass. Why bf16 is a win on an H100 and a loss
on this CPU. Why `val_bpb` is per *byte*. Why the time budget - not the step count - is what
makes two experiments comparable.

## Constraints the design had to respect

- **CPU only, inside docker.** The learner has no GPU. Every exercise runs on a laptop CPU
  in seconds, and the three capstone training runs take about three minutes each.
- **One data download, already done.** The course reads the same `autoresearch_ar-cache`
  volume `prepare.py` populated; it never downloads anything itself.
- **Windows 11 + Git Bash + Docker Desktop**, matching the learner's machine. `lab.sh` shells
  out to `docker compose`, and the compose file finds the repo by relative path because a
  Git Bash absolute path is not something Docker understands.
- **The fork is the subject, and it is honest about being a toy.** At 0.016 tokens per
  parameter the model is a thousand times short of compute-optimal. The course says so, in
  numbers, rather than letting the learner infer that a `val_bpb` of 2.4 means something.

## Non-goals

Distributed training (this repo is single-device by design), inference serving and KV
caching, fine-tuning, RLHF, quantization internals, mixture-of-experts, multimodality, and a
survey of the literature. This course teaches one pretraining stack, end to end, deeply
enough that the learner can read and extend it - and, having done so, can read the papers
behind the pieces on their own.
