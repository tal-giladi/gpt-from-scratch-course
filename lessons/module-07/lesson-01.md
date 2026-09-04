# 18 - The autonomous research loop

Everything so far has been the machine. This module is about the thing built on top of it,
which is what makes this repo interesting rather than merely educational: a **loop in which
an LLM does the research**.

`program.md` is not documentation. It is the agent's instructions - a prompt, checked into
the repo, that turns a coding assistant into an experimenter. It is worth reading in full;
here is its skeleton and why each rule is there.

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

That is hill climbing, with git as the state and `val_bpb` as the objective. Roughly twelve
experiments an hour; a hundred while a human sleeps.

## The three rules that make it work

**1. Fix the metric, fence off the file.** `prepare.py` is read-only. The evaluation, the
tokenizer, the data loading, the time budget and the sequence length all live there. The
agent may change `train.py` and nothing else. Without this the loop optimises the
*measurement* - and it would, cheerfully, because "reduce EVAL_TOKENS" does lower the
observed number.

**2. Fix the budget.** Every run is five minutes of training time. Then a modelling change
and a speed change are the same kind of thing: both are judged by the `val_bpb` reachable in
five minutes. This is the single design decision that makes the results comparable at all.

**3. Redirect the output.** `> run.log 2>&1`, and read it with `grep`, explicitly "do NOT
use tee or let output flood your context". A thousand progress lines would evict the agent's
memory of what it was testing. Context is a resource the loop has to manage, exactly like
GPU memory.

## The judgement rules

Two instructions in `program.md` are about taste rather than mechanics, and they are the
ones that keep the codebase from rotting over a hundred experiments:

- **Simplicity criterion.** "A 0.001 improvement that adds 20 lines of hacky code? Probably
  not worth it. A 0.001 improvement from deleting code? Definitely keep." Without this, hill
  climbing on a single scalar produces an unreadable pile of micro-optimisations - the
  classic failure of automated search against a metric.
- **NEVER STOP.** "Do not ask 'should I keep going?' The human might be asleep." An agent
  that checks in every hour is not autonomous; it is a slow human.

## What it is not

It is not a general research agent. The search space is one file, the objective is one
scalar, and the feedback loop is five minutes long. That combination - small, measurable,
fast - is exactly what makes it tractable, and it is worth being honest that most research
questions have none of those three properties.

And on a CPU it is a demonstration, not research: with each run taking ten minutes to reach
`val_bpb ≈ 2.4`, and run-to-run noise of a few hundredths, most real effects are below your
noise floor. You can watch the mechanism work. You cannot trust its conclusions. Lesson 19
is about telling those two situations apart.

## Do this

1. Read `program.md` in the repo - all of it, it is short.

2. Run one experiment by hand, the way the agent would:

       docker compose run --rm -e AR_TIME_BUDGET=120 autoresearch python train.py

   and look at the summary block it prints at the end.

3. Fill in `lab/exercises/lesson_18.py`: `parse_summary(text)` turns that block into a dict,
   the way step 5 of the loop does. It must also handle the two failure modes: a crashed run
   (no summary at all) and a diverged run (the `FAIL` fast-exit from lesson 10).

4. Grade it:

       bash lab/lab.sh check 18

## Hints

- The summary is the block after a line containing only `---`. Every line is
  `key:` then whitespace then a value.
- Return `{"status": "ok", ...}` with `val_bpb`, `training_seconds`, `num_steps`,
  `num_params_M` and `depth` as numbers - floats where the value has a decimal point, ints
  where it does not (`num_steps`, `depth`).
- If the text contains a line that is exactly `FAIL`, return `{"status": "diverged"}` - the
  run detected `NaN` and killed itself.
- If there is no summary and no `FAIL`, return `{"status": "crashed"}`. A traceback is not
  an exception to handle; it is a result to record.
- Ignore unknown keys rather than failing on them - the summary grows when someone adds a
  metric, and a parser that breaks on a new line is a parser that stops the loop.
- `line.split(":", 1)` splits on the first colon only, which matters if a value ever
  contains one.

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

The research loop is hill climbing with git as its memory and `val_bpb` as its objective,
and it works because three things are frozen: the metric, the file the agent may edit, and
the time budget. The two soft rules - prefer simplicity, never stop to ask - are what keep
a hundred unsupervised experiments from producing an unreadable codebase. Next: how to know
whether a result from that loop means anything.
