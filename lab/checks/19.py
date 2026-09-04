from _lib import check, done, fail, stub_guard

from exercises.lesson_19 import compare_runs

compare_runs = stub_guard(compare_runs, "compare_runs")

BASE = {"depth": 4, "time_budget": 600, "eval_tokens": 32768, "seed": 42, "lr": 0.04}


def run(bpb, status="ok", **config):
    cfg = dict(BASE)
    cfg.update(config)
    return {"status": status, "val_bpb": bpb, "config": cfg}


# 1. A clean comparison: one variable, big enough difference.
r = compare_runs(run(2.40), run(2.30, depth=6), "depth")
if not isinstance(r, dict):
    fail("compare_runs must return a dict")
check(r["valid"] is True, "one variable, equal budgets: valid")
check(r["winner"] == "b", "the lower val_bpb wins")

r = compare_runs(run(2.20), run(2.30, depth=6), "depth")
check(r["winner"] == "a", "...in the other direction too")

# 2. The bad comparison from the lesson: depth AND budget changed.
r = compare_runs(run(2.40), run(2.31, depth=6, time_budget=900), "depth")
check(r["valid"] is False, "changing the time budget as well invalidates the comparison")
check("time_budget" in r["reason"], "the reason names the offending key")
check(r["winner"] is None, "an invalid comparison declares no winner")

# 3. Moving the ruler.
r = compare_runs(run(2.40), run(2.10, depth=6, eval_tokens=4096), "depth")
check(r["valid"] is False, "changing eval_tokens invalidates it: that is the ruler, not the model")

# 4. Nothing actually changed.
r = compare_runs(run(2.40), run(2.30), "depth")
check(r["valid"] is False, "if the declared variable is identical in both runs, there is no experiment")

# 5. A run that did not finish.
r = compare_runs(run(2.40), {"status": "crashed", "config": dict(BASE)}, "depth")
check(r["valid"] is False, "a crashed run cannot be compared")
r = compare_runs({"status": "diverged", "config": dict(BASE)}, run(2.30, depth=6), "depth")
check(r["valid"] is False, "nor can a diverged one")

# 6. The noise floor.
r = compare_runs(run(2.400), run(2.395, depth=6), "depth")
check(r["valid"] is True and r["winner"] == "tie", "a 0.005 difference is a tie, not a win")
r = compare_runs(run(2.400), run(2.375, depth=6), "depth")
check(r["winner"] == "b", "a 0.025 difference clears the noise floor")

# 7. A key present in one config and missing from the other still counts as a difference.
a = run(2.40)
b = run(2.30, depth=6)
del b["config"]["lr"]
r = compare_runs(a, b, "depth")
check(r["valid"] is False, "a config key missing from one side is a second changed variable")

done("19")
