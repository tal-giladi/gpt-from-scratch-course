# 19 - Designing an experiment you can believe

The loop from lesson 18 will happily report that a change improved `val_bpb` by 0.004. This lesson
is about whether that means anything - and the honest answer, at CPU scale, is often no. None of it
needs statistics beyond subtraction. It needs a handful of rules, and the discipline to follow
them.

## First, measure the noise floor

Run the same code twice. Not a variant - the **same code, same settings**. Any difference you see is
noise.

Where can a difference even come from, with the seed fixed? Mostly from the clock. The run stops
when its time budget is used up, and on a laptop the time per step wobbles - other programs, the
operating system, which CPU cores the threads landed on. The course's two identical 2-minute runs:

    run a:   97 steps   val_bpb 2.2824
    run b:  102 steps   val_bpb 2.2756
                                -------
    difference                  0.0068

Run b happened to fit five more steps into its two minutes, saw a little more data, and scored a
little better. Nothing about the code differed.

That spread is your **noise floor**: the smallest difference you are entitled to believe. If a
variant beats the baseline by 0.004 and two identical runs differ by 0.007, you have learned
nothing - the variant's "win" is smaller than the gap between a run and itself.

Two identical runs is the cheapest measurement in this whole course, and almost nobody does it. It
recalibrates every result you read afterwards.

One honest caveat: two runs give you *one* sample of the noise. It could have come out 0.002 or
0.015 on another pair. That is why the exercise below uses a deliberately generous floor of `0.02`,
about three times what was measured here. When a decision matters, run the repeat more than once.

## The rules a comparison has to obey

**One variable.** If you change the depth *and* the learning rate and the number goes down, you do
not know which one did it - or whether one helped and the other hurt. The autonomous loop enforces
this structurally: one commit per idea, reverted if it does not win.

**Equal budgets.** `TIME_BUDGET` must be identical between the two runs. Lesson 12 explained why a
longer run is not a fair opponent: it sees more data *and* gets a longer, gentler learning-rate
cooldown. A shorter one is penalised twice.

**The same ruler.** Same validation shard, same `EVAL_TOKENS`, same `MAX_SEQ_LEN`, same special-token
masking (lesson 13). Change any of them and you have changed how the score is measured, not how good
the model is.

**A stated seed.** `train.py` fixes `torch.manual_seed(42)`, so the starting weights are identical
between runs. That removes one source of randomness - and hides another: a change that happens to
help only for *these* particular starting weights looks like a real effect. If a result matters,
re-run both sides with a different seed.

**Direction declared in advance.** "I expect a shallower model to do better here, because lesson 15
showed this model sees far too few tokens for its size" is a prediction. It can be wrong, and that
is what makes it worth something. "It came out lower, so it worked" is a story fitted to a number
afterwards - and with enough experiments, *something* always comes out lower.

## Why the last rule matters more than it sounds

Lesson 18 showed that a change which does nothing still comes out lower about half the time. Now run
100 experiments and keep every one that improves. Even if not a single idea was any good, about fifty
get kept - and because the branch advances on every keep, each lucky run becomes the new baseline
that the next idea has to beat. The noise does not just sneak in; it compounds.

`results.tsv` exists partly so a human can look back and ask, "did this one really work, or did it
just win once?" The defence is cheap and boring: at the end, re-run the changes that survived,
against the original baseline. A real improvement reproduces. A lucky one does not.

## A bad comparison, and how to fix it

    run A: depth 4, TIME_BUDGET=600, val_bpb 2.399
    run B: depth 6, TIME_BUDGET=900, val_bpb 2.310   ->  "deeper is better!"

No. B trained for 50% longer. This is not an experiment about depth; it is an experiment about time
in which the depth also changed. Two variables moved, and one of them (the budget) is known to lower
`val_bpb` all by itself.

The fixed version:

    (illustrative numbers)
    run A:  depth 4, TIME_BUDGET=600, val_bpb 2.399
    run A': depth 4, TIME_BUDGET=600, val_bpb 2.388   noise floor = 0.011
    run B:  depth 6, TIME_BUDGET=600, val_bpb 2.371   beats A by 0.028 - more than 0.011

Now only depth differs, both runs had the same ten minutes, and the gap is bigger than the measured
noise. That is a result you are allowed to report - once - and would want to see reproduce. The
function you write for this lesson checks exactly these conditions.

## Do this

1. Measure your own noise floor with two identical 2-minute runs. From the course folder:

       bash lab/lab.sh experiment baseline
       bash lab/lab.sh experiment baseline2

   Each takes a little over two minutes and prints its summary. Write the two `val_bpb` values down
   and subtract. That difference is the smallest result you may believe from any later comparison on
   this machine. Keep these two logs: they are the first two runs of the capstone in lesson 20.

2. Fill in `lab/exercises/lesson_19.py`: `compare_runs(a, b, changed)`.

3. Grade it:

       bash lab/lab.sh check 19

## Hints

- Each run is a dict like the one lesson 18 produced, plus a `"config"` dict of the settings it ran
  with. Two runs are comparable only if every config key **except** the declared `changed` one is
  equal.
- Find the differing keys by looking at the union of both configs' keys, and comparing with
  `.get(k)` - a key present on one side and missing on the other counts as a difference.
- Return a dict with `"valid"` (bool), `"reason"` (a short string), and `"winner"` (`"a"`, `"b"`,
  `"tie"`, or `None` when the comparison is invalid).
- The invalid cases: a run whose status is not `"ok"`; `changed` not actually differing between the
  two; any config difference other than `changed`. When a second key differs, name it in the reason.
- `NOISE_FLOOR = 0.02`. If the two `val_bpb` values differ by less than that, the winner is `"tie"` -
  the comparison was valid, it just did not separate them.
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

Measure the noise first: two identical runs, one subtraction. A difference is believable only when it
is bigger than that noise floor, when exactly one thing changed, when both runs had the same budget
and the same ruler, and when you said in advance which way it should go. Without those rules, a loop
that keeps anything "lower" will keep about half of the ideas that do nothing. Next: the capstone,
where you run one honest experiment end to end.
