# 20 - Capstone: one honest experiment

Nineteen lessons of mechanism. This one is the exam, and it is not a coding exercise - it is a
research exercise. You will run three real trainings, measure your own noise floor, predict what a
change will do, and write down a conclusion that the check will refuse to accept if your own data
does not support it.

## What you are producing

Three runs and a short report:

| Run | What it is | Why |
|---|---|---|
| `baseline` | the fixed protocol, unchanged | the thing to beat |
| `baseline2` | the fixed protocol, unchanged, **again** | your measured noise floor (lesson 19) |
| `variant` | the fixed protocol with **one** knob changed | your hypothesis |

If you did lesson 19's Do-this, you already have `baseline` and `baseline2`.

The report says what you changed, why you expect it to matter, and which direction `val_bpb` should
move - and you write that part **before** you run the variant. Then, once the numbers are in, you add
the conclusion.

## How the verdict is decided

The check does the same arithmetic you should do by hand. Suppose (illustrative numbers):

    baseline    2.2824
    baseline2   2.2756
    noise floor = |2.2824 - 2.2756| = 0.0068

Three possible variants, and what each one means if you predicted "lower":

    variant 2.2700   moved -0.0124   bigger than 0.0068, and lower     ->  confirmed
    variant 2.2790   moved -0.0034   not bigger than 0.0068            ->  inconclusive
    variant 2.3000   moved +0.0176   bigger than 0.0068, but higher    ->  refuted

Notice the middle one. The variant *did* come out lower, and the hill-climbing loop from lesson 18
would have kept it. But it moved less than two identical runs differ from each other, so the honest
answer is "I can't tell". **Inconclusive is a correct result, not a failure** - and at this scale it
is the most common one.

## Choosing something worth testing

Anything you can express as one `AR_*` setting. Pick one you can argue about in advance - "I expect
X because Y" is the part that matters; being right is optional, and being honest about being wrong
is the point.

Good candidates, each connected to a lesson:

- **`AR_DEPTH=2` or `AR_DEPTH=6`.** Lesson 15 showed this fork sees far too few tokens per parameter:
  a 2-minute run is under 0.02 tokens per parameter against Chinchilla's 20. A smaller model gets
  through more tokens in the same time; a bigger one has more capacity. Which effect wins here?
  (`AR_DEPTH` also sets the width: `n_embd = depth * 64`.)
- **`AR_TOTAL_BATCH_SIZE=4096`.** Two micro-batches per step instead of one: half as many optimizer
  steps, each with a less noisy gradient (lesson 12).
- **`AR_HEAD_DIM=32`.** Same total width, twice as many attention heads, each half as wide (lesson 08).
  Free, or not?

Two tempting choices that do **not** work, and why - both are lessons in themselves:

- **`AR_MAX_SEQ_LEN=128`** looks like a clean "cheaper attention" experiment. But `MAX_SEQ_LEN` is
  also used by the evaluation (lesson 13): the variant would be scored on a different ruler. The check
  rejects it, and it should.
- **`AR_BF16=1`** looks like a quick dtype experiment. Measured on the course machine, a training step
  in bfloat16 on this CPU took about **190 seconds** instead of about 1.2. With a 120-second budget,
  the run would spend over nine minutes on its three untimed warm-up steps (0, 1 and 2, lesson 14)
  and then fit a single timed step - well over ten minutes, to learn what lesson 16 already told you.

## Do this

1. Write your hypothesis first. Open `lab/exercises/lesson_20.py` and fill in `hypothesis`,
   `changed` and `expectation` - before running the variant.

2. Run the three experiments from the course folder (skip the first two if lesson 19 already made
   them). Each takes a little over two minutes:

       bash lab/lab.sh experiment baseline
       bash lab/lab.sh experiment baseline2
       bash lab/lab.sh experiment variant AR_DEPTH=2

   Replace `AR_DEPTH=2` with your own knob. The logs land in `lab/capstone/runs/`. Everything except
   your one override is pinned by `lab/tools/run_experiment.py` - the same 2-minute budget, eval size,
   context and batch - so the runs really are comparable.

   Do not re-run just the variant until you like its number. That is lesson 19's "won once" problem,
   performed by hand.

3. Work out the verdict yourself, exactly as in the table above, and fill in `conclusion`.

4. Grade it:

       bash lab/lab.sh check 20

   The check re-reads your three logs, recomputes the noise floor from `baseline` versus
   `baseline2`, works out what the data actually says, and compares that against the
   `conclusion` you wrote. Claiming a win that is inside your own noise floor fails. So does
   claiming your hypothesis was confirmed when the number moved the other way.

## Hints

- `expectation` is `"lower"`, `"higher"` or `"no change"` - what you predicted `val_bpb` would do in
  the variant, relative to the baseline.
- `conclusion` is `"confirmed"`, `"refuted"` or `"inconclusive"`:
  - **inconclusive** when `|variant - baseline|` is not bigger than your measured noise floor
    `|baseline - baseline2|`.
  - **confirmed** when the difference clears the noise floor and moved the way you predicted.
  - **refuted** when it clears the noise floor and moved the other way - or when it clearly moved
    at all and you had predicted `"no change"`. (So `"no change"` can never come out `"confirmed"`
    here: a difference inside the noise floor is `"inconclusive"` whatever you predicted.)
- `baseline`, `repeat` and `variant` are the names of the logs, without `.log`.
- `hypothesis` must be a real sentence. The check only requires 20 characters and does not read the
  prose; your reader does.
- The check also refuses runs that are not comparable: different time budgets, different
  `vocab_size` or `sequence_len` between baseline and variant, or two baselines with different
  model configs.
- If your variant crashed, that is a result too: fix it or pick a different knob. The check requires
  three completed runs, because a crash tells you about your change, not about the model.

## Solution

There is not one. The point is your own experiment, run honestly. This is a filled-in example of the
*shape*, not of the answer:

    def capstone_report():
        return {
            "hypothesis": "At under 0.02 tokens per parameter this model is far past "
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

You can read `train.py` line by line and say what each line is for and what breaks without it. You
have written, from scratch and checked against the real thing, the attention, the rotary embedding,
the RMS norm, the MLP, the parameter count, the FLOPs estimate, the AdamW step, Muon's momentum, the
learning-rate schedule, gradient accumulation, the bits-per-byte metric and the sampling loop. You
have measured your own machine, read a port for what it gave up, and run a controlled experiment
against a frozen metric.

What is deliberately not here: distributed training (this repo is single-device by design),
inference serving and KV caching, fine-tuning and RLHF, quantization, and mixture-of-experts. Every
one of them assumes exactly what you now have - a pretraining stack you can read.

The obvious next step is the one the repo was built for: run the loop for real. On a rented H100 for
an evening, `program.md` works as written, five minutes an experiment, and you will be able to tell
which of its results you should believe.
