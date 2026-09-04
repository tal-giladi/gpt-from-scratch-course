# GPT from scratch: reading, running and improving a real pretraining stack

**7 modules · 20 lessons · 20 graded exercises · 7 quizzes · 1 capstone**

A hands-on course for software engineers with no ML background who want to understand a
modern GPT completely - every line of it - and be able to change it on purpose. The subject
is [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch): 700 lines of
single-file, single-device pretraining code, plus the agent loop that does research on it.
You do not read this course - you run it. Every lesson ends in an exercise **graded against
the real repository**, running on your CPU inside the same docker image the model trains in.

---

## Start here

You need Docker Desktop, bash, and the CPU fork of the repo cloned next to this course:

```bash
git clone https://github.com/tal-giladi/autoresearch ../autoresearch
```

Then, once:

```bash
bash lab/lab.sh up
```

That builds the CPU image if you do not have it, downloads two data shards and trains the
tokenizer if the cache is empty (~185 MB, once), and runs a self-test proving the lab can see
the repo, the data and the model. Nothing is installed on your machine outside docker.

Then open [`lessons/module-01/lesson-01.md`](lessons/module-01/lesson-01.md), do the exercise
at the bottom, and run:

```bash
bash lab/lab.sh check 01
```

| Command | What it does |
|---|---|
| `bash lab/lab.sh up` | verify the image, the data and the tokenizer |
| `bash lab/lab.sh check NN` | grade exercise NN against the real repository |
| `bash lab/lab.sh hint NN` | point you at the lesson's Hints section |
| `bash lab/lab.sh solve NN` | copy the reference solution into your exercise file |
| `bash lab/lab.sh reset NN` | restore exercise NN to its original stub |
| `bash lab/lab.sh status` | which exercises currently pass |
| `bash lab/lab.sh shell` | a Python shell inside the lab, with the repo importable |

---

## Contents

### Lessons — `lessons/module-NN/lesson-XX.md`

| Module | Lessons | Title |
|---|---|---|
| 01 | 3 | From text to tensors |
| 02 | 3 | The residual stream and attention |
| 03 | 3 | The rest of the model |
| 04 | 3 | Training |
| 05 | 3 | Measuring |
| 06 | 2 | Speed, and reading a port |
| 07 | 3 | Research |

Each lesson: the mechanism explained properly against the real code, a shell session to see
it with your own eyes, a graded exercise, hints, and a full solution.

### The annotated file — `assets/`
[`annotated-train.md`](assets/annotated-train.md) walks `train.py` **top to bottom** in file
order, with a one-page map of how the six stages connect. The lessons teach the ideas in
dependency order; this reads the file in its own order. Doing both is how "I understand each
line" and "I understand the whole thing" become the same sentence.

### Curriculum — `curriculum/`
[`course-outline.md`](curriculum/course-outline.md) and seven module plans with objectives,
dependencies, and the misconceptions each module is written to break.

### Assessments — `assessments/`
Seven quizzes with full answer keys, weighted toward diagnosis and judgement rather than
recall.

### Lab — `lab/`
`lab.sh` (the grader), `lib/common.py` (the handle on the real repo), `exercises/` (your
stubs), `checks/` (20 graders), `solutions/`, `fixtures/`, and `tools/run_experiment.py` for
the capstone.

---

## How it is built

**Graded against the real repository, not a teaching copy.** The lab mounts the actual
`autoresearch` clone at `/autoresearch` and the real tokenizer and data shards from the
docker volume `prepare.py` filled. When lesson 06 says "this matches `apply_rotary_emb`", the
check imports that function and compares tensors to 1e-5. Edit the repo and the course
follows your edit.

**No second toolchain.** No venv, no host Python, no second 200 MB torch download. Every
check runs inside the `autoresearch-cpu` image you already built.

**One fixed toy scale.** 2 layers, 128 dimensions, 128 tokens of context, the real
8192-token tokenizer, real text. Checks run in seconds; nothing is mocked.

**You write the model, then you read the real one.** Across the twenty exercises you
implement the packer, the causal windowed attention, the rotary embedding, RMS norm, the
ReLU² MLP, a block, the parameter-count formula, the forward pass and sampling, a gradient
report, an AdamW step, Muon's momentum, the LR schedule, gradient accumulation, the
bits-per-byte metric, the FLOPs and MFU calculation, a matmul benchmark, the sliding-window
mask, a run-log parser and an experiment-validity checker - each checked against the real
implementation.

**Nothing hand-waved.** Why the row capacity is `T + 1`. Why every output projection is
initialised to zero, and why `c_fc` gets exactly zero gradient on the first step because of
it. Why `lm_head` is untied from `wte` and what the 150x learning-rate gap has to do with it.
Why bf16 is a win on an H100 and measurably slower on your CPU. Why the metric is per *byte*.
Why a fixed *time* budget is what makes two experiments comparable at all.

**Honest about scale.** Lesson 15 does the arithmetic: this fork's ten-minute CPU run is
about 1000x short of compute-optimal, and the binding constraint is the vocabulary size, not
the architecture. Lesson 19 has you measure your own noise floor before you are allowed to
believe any result - including your capstone's.

---

## The capstone

Three real training runs - baseline, an identical repeat, and one changed knob - about three
minutes each. You write your hypothesis and your predicted direction *before* looking at the
variant's number. The grader re-reads your logs, recomputes your noise floor from the two
baselines, works out what the data actually says, and fails you if your conclusion does not
match it. Reporting "inconclusive" is a passing result; claiming a win inside your own noise
floor is not.

---

## Cleaning up

```bash
docker compose -f lab/docker-compose.yml down
rm -rf lab/capstone/runs
```

The data and tokenizer live in the `autoresearch_ar-cache` docker volume, shared with the
repo under study; remove it with `docker volume rm autoresearch_ar-cache` if you want the
disk back.
