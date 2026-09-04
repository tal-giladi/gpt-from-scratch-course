# Lesson 20 - there is no reference solution for a capstone: the answer is your own
# experiment. This file shows the SHAPE the check expects, filled in for the depth
# experiment described in the lesson. Running `lab.sh solve 20` and grading it will only
# pass if you have actually produced runs named baseline, baseline2 and variant - and the
# conclusion below is very unlikely to match what your data says.


def capstone_report() -> dict:
    return {
        "hypothesis": (
            "At 0.016 tokens per parameter this configuration is far past compute-optimal, "
            "so halving the depth should trade capacity we cannot fill for tokens we can."
        ),
        "changed": "AR_DEPTH",
        "expectation": "lower",
        "baseline": "baseline",
        "repeat": "baseline2",
        "variant": "variant",
        "conclusion": "inconclusive",
    }
