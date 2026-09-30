# Lesson 19 - Designing an experiment you can believe
# Read lessons/module-07/lesson-02.md before filling this in.

NOISE_FLOOR = 0.02


def compare_runs(a, b, changed):
    """Decide whether two runs can be compared, and if so which one won.

    Each run is {"status": "ok"|..., "val_bpb": float, "config": {...}}.
    `changed` names the ONE config key the experiment was about.
    
    Return {"valid": bool, "reason": str, "winner": "a"|"b"|"tie"|None}.

    Invalid when: either run did not complete; `changed` is not actually
    different between the two configs; or any OTHER config key differs.
    Valid but "tie" when the two val_bpb values are within NOISE_FLOOR.
    Lower val_bpb wins.
    """
    # TODO: validate first, compare second.
    
    a_valid = a["status"] == "ok"
    b_valid = b["status"] == "ok"
    for key in ["time_budget", "eval_tokens"]:
        if a["config"][key]!=b["config"][key]:
            return {"valid": False, "reason": key, "winner": None}
    
    if (a["config"].get("lr")==b["config"].get("lr") and a["config"].get("depth")==b["config"].get("depth")):
        return {"valid": False, "reason": "changed", "winner": None}
    if (a["config"].get("lr")!=b["config"].get("lr") and a["config"].get("depth")!=b["config"].get("depth")):
        return {"valid": False, "reason": "changed", "winner": None}

    if a_valid==False or b_valid==False:
        return {"valid": False, "reason": "one of the runs is invalid", "winner": None}
    winner = ""
    diff = a["val_bpb"]-b["val_bpb"]
    if diff<0:
        diff = diff * -1
    if diff <=0.02:
        winner = "tie"
    elif a["val_bpb"]<b["val_bpb"]:
        winner = "a"
    else:
        winner = "b"
    
        
    print(a["config"], b["config"])
    return {"valid": True, "winner": winner}
    raise NotImplementedError
