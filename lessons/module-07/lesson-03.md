# 20 - Capstone: one honest experiment

Nineteen lessons of mechanism. This one is the exam, and it is not a coding exercise - it is
a research exercise. You will run three real trainings, measure your own noise floor, and
write down a conclusion your own check will refuse to accept if the data does not support
it.

## What you are producing

Three runs and a report:

| Run | What it is | Why |
|---|---|---|
| `baseline` | the protocol, unchanged | the thing to beat |
| `baseline2` | the protocol, unchanged, **again** | your measured noise floor |
| `variant` | the protocol with **one** knob changed | your hypothesis |

And a report stating, *before* you look at the variant's number: what you changed, why you
expect it to help (or not), and which direction `val_bpb` should move.

## Choosing something worth testing

Anything you can express as one `AR_*` override. Some that follow directly from the course:

- **`AR_DEPTH=2` or `AR_DEPTH=8`.** Lesson 15 said this fork is a thousand times short of
  compute-optimal, so a *smaller* model should do better in a fixed budget. Does it?
- **`AR_TOTAL_BATCH_SIZE=4096`.** Fewer, less noisy optimizer steps versus more, noisier
  ones (lesson 12).
- **`AR_MAX_SEQ_LEN=128`.** Cheaper attention, more steps, less context (lesson 14's FLOPs
  split).
- **`AR_HEAD_DIM=32`.** More heads of the same total width - free, or not?
- **`AR_BF16=1`.** Lesson 16 measured bf16 at 0.78x of fp32 on this machine. Does the
  slowdown show up in `val_bpb`?

Pick one you can argue about in advance. "I expect X because Y" is the part being graded;
being right is optional and being *honest about being wrong* is the point.

## Do this

1. Run the three experiments. Each takes about three minutes:

       bash lab/lab.sh shell

       python tools/run_experiment.py baseline
       python tools/run_experiment.py baseline2
       python tools/run_experiment.py variant AR_DEPTH=2

   The logs land in `lab/capstone/runs/`. Everything except your one override is pinned by
   `tools/run_experiment.py`, so the runs really are comparable.

2. Fill in `lab/exercises/lesson_20.py`: `capstone_report()` returns your hypothesis, which
   knob you changed, which direction you expected, and your conclusion.

3. Grade it:

       bash lab/lab.sh check 20

   The check re-reads your three logs, recomputes the noise floor from `baseline` versus
   `baseline2`, works out what the data actually says, and compares that against the
   `conclusion` you wrote. Claiming a win that is inside your own noise floor fails. So does
   claiming your hypothesis was confirmed when the number moved the other way.

## Hints

- `expectation` is `"lower"`, `"higher"` or `"no change"` - what you predicted `val_bpb`
  would do in the variant relative to the baseline.
- `conclusion` is `"confirmed"`, `"refuted"` or `"inconclusive"`:
  - **inconclusive** when `|variant - baseline|` is not bigger than your measured noise
    floor (`|baseline - baseline2|`). This is the most likely outcome, and reporting it is
    a correct result, not a failure.
  - **confirmed** when the difference clears the noise floor and moved the way you
    predicted.
  - **refuted** when it clears the noise floor and moved the other way.
- `hypothesis` must be a real sentence (the check requires 20 characters and does not check
  the prose - your reader does).
- If your variant crashed, that is a result too: fix it or pick a different knob. The check
  requires three completed runs, because a crash tells you about your change, not about the
  model.

## Solution

There is not one. The point is your own experiment, run honestly. The reference file is a
filled-in example of the *shape*:

    def capstone_report():
        return {
            "hypothesis": "At 0.016 tokens per parameter this model is far past "
                          "compute-optimal, so halving the depth should trade capacity "
                          "we cannot fill for tokens we can.",
            "changed": "AR_DEPTH",
            "expectation": "lower",
            "baseline": "baseline",
            "repeat": "baseline2",
            "variant": "variant",
            "conclusion": "inconclusive",
        }

## Where this leaves you

You can read `train.py` line by line and say what each line is for and what breaks without
it. You have written, from scratch and checked against the real thing, the attention, the
rotary embedding, the RMS norm, the MLP, the parameter count, the FLOPs estimate, the AdamW
step, Muon's momentum, the learning-rate schedule, gradient accumulation, the bits-per-byte
metric and the sampling loop. You have measured your own machine, ported a GPU-only kernel
and proved the replacement equivalent, and run a controlled experiment against a frozen
metric.

What is deliberately not here: distributed training (this repo is single-device by design),
inference serving and KV caching, fine-tuning and RLHF, quantization, and mixture-of-experts.
Every one of them assumes exactly what you now have - a pretraining stack you can read.

The obvious next step is the one the repo was built for: run the loop for real. On a rented
H100 for an evening, `program.md` works as written, five minutes an experiment, and you will
be able to tell which of its results you should believe.
