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
    raise NotImplementedError
