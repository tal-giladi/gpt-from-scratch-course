# 18 - The autonomous research loop

Everything so far has been the machine: how the model is built, trained and measured. This module
is about what the repo is really *for*: a **loop in which an AI coding assistant does the research
on its own**.

An "agent" here is simply an LLM that can run commands - edit a file, run a script, read the
output - and decide what to do next based on what it saw. `program.md` is the set of instructions
that turns such an assistant into an experimenter. It is not documentation for humans; it is a
prompt, checked into the repo. It is short and worth reading in full. Here is its skeleton, and
why each rule is there.

## The loop

    LOOP FOREVER:
      1. look at the git state
      2. change train.py with an experimental idea
      3. git commit
      4. uv run train.py > run.log 2>&1
      5. grep "^val_bpb:" run.log
      6. if the grep is empty, the run crashed - read the traceback, maybe fix
      7. record the result in results.tsv
      8. if val_bpb improved: keep the commit (the branch advances)
      9. if it did not: git reset back

In plain words: try an idea, measure it, keep it if the number went down, undo it if not, repeat.
Git is the memory - the current commit is always "the best version so far" - and `results.tsv` is
the lab notebook. `program.md`'s own example of that notebook:

    commit    val_bpb    memory_gb  status   description
    a1b2c3d   0.997900   44.0       keep     baseline
    b2c3d4e   0.993200   44.2       keep     increase LR to 0.04
    c3d4e5f   1.005000   44.0       discard  switch to GeLU activation
    d4e5f6g   0.000000   0.0        crash    double model width (OOM)

Row 2 went down, so it stays and becomes the new starting point. Row 3 went up, so the code is
reset to row 2's commit. Row 4 did not even finish.

This strategy is called **hill climbing**: from where you stand, take a step; if it is an
improvement, stay there; if not, step back and try another direction. It never plans ahead and
never accepts a temporary loss, which makes it simple and relentless. On a GPU each experiment is
five minutes, so roughly twelve an hour - about a hundred while a human sleeps.

## The three rules that make it work

**1. Fix the metric, fence off the file.** `prepare.py` is read-only. The evaluation, the tokenizer,
the data loading, the time budget and the sequence length all live there. The agent may change
`train.py` and nothing else. Without this rule the loop would optimise the *measurement* - and it
would do so cheerfully, because "reduce `EVAL_TOKENS`" really does change the reported number
(lesson 13).

**2. Fix the budget.** Every run gets the same training time. Then a modelling change and a speed
change are the same kind of thing: both are judged by the `val_bpb` reachable in that time (lesson
12). This single design decision is what makes results from different experiments comparable at
all.

**3. Redirect the output.** `> run.log 2>&1` sends everything the run prints into a file, and the
agent reads back only the lines it needs with `grep`. `program.md` is explicit: "do NOT use tee or
let output flood your context". A run prints one progress line per step - hundreds of lines - and an
LLM can only hold so much text at once (its *context*). If the progress lines pushed out the
agent's memory of what it was testing and why, the next decision would be made blind. Context is a
resource the loop has to manage, exactly like memory on the GPU.

## The judgement rules

Two instructions are about taste rather than mechanics, and they are the ones that keep the code
from rotting over a hundred experiments:

- **Simplicity criterion.** "A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably
  not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep." Without it, hill
  climbing on one number produces an unreadable pile of tiny tweaks, each of which won once - the
  classic failure of automated search against a metric.
- **NEVER STOP.** "Do NOT pause to ask the human if you should continue." The human might be asleep.
  An agent that checks in every hour is not autonomous; it is a slow human.

## The weak spot: "improved" means "lower by any amount"

Step 8 keeps a change if `val_bpb` is lower - by *any* amount. Now remember lesson 13: two identical
2-minute runs on this machine came out at `2.2824` and `2.2756`. Nothing changed between them; the
0.007 gap is pure chance.

So picture a change that does **nothing at all**. Its run is just another roll of that same dice,
and it comes out lower than the baseline about half the time. Hill climbing keeps it, the branch
advances, and the next experiment is compared against a baseline that got lucky. Run a hundred
do-nothing experiments and roughly fifty of them get committed as "improvements".

That is not a reason to distrust the loop - it is exactly the question lesson 19 answers: how big
does a difference have to be before you believe it?

## What it is not

It is not a general research agent. The search space is one file, the objective is one number, and
the feedback arrives within minutes. That combination - small, measurable, fast - is exactly what
makes it tractable, and most research questions have none of those three properties.

And on a CPU it is a demonstration, not research. A 2-minute run here reaches `val_bpb` around 2.28,
with run-to-run noise of several thousandths, and many real effects are smaller than that. You can
watch the mechanism work. You should not trust its conclusions without the extra care of lesson 19.

## Do this

1. Read `program.md` in the autoresearch repo - all of it, it is short.

2. Run one experiment by hand, the way the agent would. From the course folder:

       bash lab/lab.sh experiment try1

   This runs `train.py` inside the lab container with the course's fixed 2-minute protocol, sends
   every line it prints to `lab/capstone/runs/try1.log` (step 4 of the loop), and then shows you
   the summary block at the end. Open the log and find the three kinds of line that contain a colon:
   the `Model config:` dump near the top, the `step ... | loss: ...` progress lines, and the summary
   after `---`. Only the last kind is the result.

3. Fill in `lab/exercises/lesson_18.py`: `parse_summary(text)` turns that block into a dict,
   the way step 5 of the loop does. It must also handle the two failure modes: a crashed run
   (no summary at all) and a diverged run (the `FAIL` fast-exit from lesson 10).

4. Grade it:

       bash lab/lab.sh check 18

## Hints

- The summary is the block after a line containing only `---`. Every line is `key:`, then
  whitespace, then a value:

      ---
      val_bpb:          2.282386
      training_seconds: 120.3
      num_steps:        97

- Return `{"status": "ok", ...}` with `val_bpb`, `training_seconds`, `num_steps`,
  `num_params_M` and `depth` as numbers - floats where the value has a decimal point, ints
  where it does not (`num_steps`, `depth`).
- If the text contains a line that is exactly `FAIL`, return `{"status": "diverged"}` - the
  run detected `NaN` (or an exploding loss) and killed itself.
- If there is no summary and no `FAIL`, return `{"status": "crashed"}`. A traceback is not
  an exception for your parser to raise; it is a result to record.
- Progress lines and the config dump contain colons too. Only accept the keys you know, and
  ignore everything else - including a new metric someone adds to the summary later. A parser
  that breaks on an unexpected line is a parser that stops the loop.
- `line.split(":", 1)` (or `line.partition(":")`) splits on the first colon only, which matters
  if a value ever contains one.

## Solution

    def parse_summary(text: str) -> dict:
        lines = [ln.strip() for ln in text.splitlines()]
        if any(ln == "FAIL" for ln in lines):
            return {"status": "diverged"}

        wanted_int = {"num_steps", "depth"}
        wanted_float = {"val_bpb", "training_seconds", "total_seconds", "mfu_percent",
                        "peak_vram_mb", "total_tokens_M", "num_params_M"}
        out = {}
        for line in lines:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            try:
                if key in wanted_int:
                    out[key] = int(float(value))
                elif key in wanted_float:
                    out[key] = float(value)
            except ValueError:
                continue
        if "val_bpb" not in out:
            return {"status": "crashed"}
        out["status"] = "ok"
        return out

## Summary

The research loop is hill climbing with git as its memory and `val_bpb` as its objective: try an
idea, keep it if the number went down, undo it if not. It works because three things are frozen -
the metric, the file the agent may edit, and the time budget - and because the agent reads results
with `grep` instead of drowning its context in progress lines. The two soft rules, prefer simplicity
and never stop to ask, keep a hundred unsupervised experiments from producing an unreadable codebase.
Its weak spot is that "lower" includes "lower by chance". Next: how to know whether a result from
that loop means anything.
