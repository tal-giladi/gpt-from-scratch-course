# Lesson 18 - The autonomous research loop
# Read lessons/module-07/lesson-01.md before filling this in.


def parse_summary(text: str) -> dict:
    """Turn a train.py run log into a result record.

    Three outcomes:
      - a summary block is present -> {"status": "ok", "val_bpb": float, ...}
        with num_steps and depth as ints and the rest as floats
      - a line that is exactly "FAIL" -> {"status": "diverged"}
      - neither                      -> {"status": "crashed"}

    Unknown keys are ignored, not an error.
    """
    # TODO: scan the lines, parse "key: value", classify the outcome.
    raise NotImplementedError
