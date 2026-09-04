# 19 - Designing an experiment you can believe

The loop from lesson 18 will happily report that a change improved `val_bpb` by 0.004. The
question this lesson answers is whether that means anything - and the honest answer, at CPU
scale, is usually no.

## First, measure the noise floor

Run the same code twice. Not a variant - the *same code*. Any difference you see is noise:
non-deterministic thread scheduling, a different number of steps completed inside the time
budget, a slightly different point on the learning-rate schedule when the clock ran out.

That spread is your **noise floor**, and it is the smallest improvement you are entitled to
believe. On this fork's CPU configuration it is on the order of 0.01-0.02 bpb, mostly
because the run is short enough that finishing one extra step out of fifteen visibly moves
the result. Anything smaller than the floor is a coin flip you have chosen to interpret.

Nobody does this often enough. It is two runs, once, and it recalibrates every result you
will read afterwards.

## The rules a comparison has to obey

**One variable.** If you change the depth *and* the learning rate, a better number tells you
nothing about either. This is the rule the autonomous loop enforces structurally: one commit
per idea, reverted if it does not win.

**Equal budgets.** `TIME_BUDGET` and `EVAL_TOKENS` must be identical between the two runs.
Lesson 12 explained why a truncated run is penalised twice: it sees less data *and* it never
completes its learning-rate warmdown, so its weights are left mid-flight.

**The same evaluation.** Same shard, same `MAX_SEQ_LEN`, same masking. Change any of them
and you have changed the ruler, not the model.

**A stated seed.** `train.py` fixes `torch.manual_seed(42)` so initialisation is identical
between runs. That removes one source of variance and hides another: a change that only
helps at seed 42 looks like a real effect. If a result matters, re-run it at a second seed.

**Direction declared in advance.** "I expect this to help because attention is the bottleneck
at this context length" is a prediction that can be wrong. "It came out lower, so it worked"
is a story fitted to a number afterwards - and with enough experiments, some change will
always come out lower.

## Why the last rule matters more than it sounds

Run 100 experiments, keep the ones that improve, and you have run 100 statistical tests
without correcting for it. Some of your "wins" are noise you have promoted to a commit -
and because the branch advances on every keep, the noise compounds into the baseline.
`results.tsv` exists partly so a human can look back and ask "did this one really work, or
did it just win once?"

The defence is cheap and boring: re-run the changes that survived, at the end, against the
original baseline. A real improvement reproduces. A noise win does not.

## A worked example of a bad comparison

    run A: depth 4, TIME_BUDGET=600, val_bpb 2.399
    run B: depth 6, TIME_BUDGET=900, val_bpb 2.310   ->  "deeper is better!"

No. B trained for 50% longer. It is not even an experiment about depth; it is an experiment
about time in which depth also changed. The check for this lesson makes you write the
function that catches exactly this.

## Do this

1. Measure your own noise floor. Two identical short runs:

       docker compose run --rm -e AR_TIME_BUDGET=120 autoresearch python train.py | tail -3
       docker compose run --rm -e AR_TIME_BUDGET=120 autoresearch python train.py | tail -3

   Write the two `val_bpb` values down. That difference is the smallest result you may
   believe from any later comparison.

2. Fill in `lab/exercises/lesson_19.py`: `compare_runs(a, b, changed)`.

3. Grade it:

       bash lab/lab.sh check 19

## Hints

- Each run is a dict like the one lesson 18 produced, plus a `"config"` dict of the settings
  it ran with. Two runs are comparable only if every config key **except** the declared
  `changed` one is equal.
- Return a dict with `"valid"` (bool), `"reason"` (a short string), and `"winner"`
  (`"a"`, `"b"`, or `"tie"`).
- The invalid cases, in the order the check tests them: a run that is not `"ok"`; more than
  one config difference; a config difference that is not the declared `changed` key;
  `changed` not actually differing between the two.
- `NOISE_FLOOR = 0.02`. If the two `val_bpb` values differ by less than that, the winner is
  `"tie"` - the comparison is valid, but it did not separate them.
- Lower `val_bpb` wins.

## Solution

    NOISE_FLOOR = 0.02

    def compare_runs(a, b, changed):
        for name, run in (("a", a), ("b", b)):
            if run.get("status") != "ok":
                return {"valid": False, "reason": f"run {name} did not complete", "winner": None}

        keys = set(a["config"]) | set(b["config"])
        differing = {k for k in keys if a["config"].get(k) != b["config"].get(k)}

        if changed not in differing:
            return {"valid": False, "reason": f"{changed} is the same in both runs", "winner": None}
        extra = differing - {changed}
        if extra:
            return {"valid": False, "reason": f"more than one variable changed: {sorted(extra)}", "winner": None}

        delta = a["val_bpb"] - b["val_bpb"]
        if abs(delta) < NOISE_FLOOR:
            return {"valid": True, "reason": "difference is within the noise floor", "winner": "tie"}
        return {"valid": True, "reason": "one variable, equal budgets", "winner": "b" if delta > 0 else "a"}

## Summary

An improvement is believable when it is bigger than the noise floor you actually measured,
when exactly one thing changed, when both runs got the same budget and the same ruler, and
when you said in advance which way it should go. Everything else is a story told about a
number. Next: the capstone, where you run one honest experiment end to end.
