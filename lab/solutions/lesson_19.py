# Lesson 19 - reference solution.

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
