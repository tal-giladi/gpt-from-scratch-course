# Lesson 18 - The autonomous research loop
# Read lessons/module-07/lesson-01.md before filling this in.

import re

def parse_summary(text: str) -> dict:
    """Turn a train.py run log into a result record.

    Three outcomes:
      - a summary block is present -> {"status": "ok", "val_bpb": float, ...}
        with num_steps and depth as ints and the rest as floats
      - a line that is exactly "FAIL" -> {"status": "diverged"}
      - neither                      -> {"status": "crashed"}

    Unknown keys are ignored, not an error.
    """
    print(text)
    # TODO: scan the lines, parse "key: value", classify the outcome.
    status = ""
    val_bpb = 0.0
    if "FAIL" in text:
        status = "diverged"
    elif text == "" or "---" not in text:
        status = "crashed"
    else:
        values = {
    key.strip(): parse_value(key.strip(), value)
    for line in text.split("---")[1].splitlines()
    if ":" in line and "checkpoint" not in line
    for key, value in [line.split(":", 1)]
}
        values["status"] = "ok"
        return values;

    return {"status": status }
    raise NotImplementedError


def parse_value(key, value):
    value = value.strip()
    if key == "num_steps" or key =="depth":
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value.strip()
