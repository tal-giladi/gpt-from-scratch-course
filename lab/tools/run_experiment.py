"""Run one training experiment and save its log where the capstone check can find it.

Usage (from inside the lab container - lab.sh does that for you):

    python tools/run_experiment.py baseline
    python tools/run_experiment.py variant AR_DEPTH=2
    python tools/run_experiment.py baseline2

Every run uses the same fixed settings unless you override them on the command line, so
whatever you change is the only thing that changed. The log lands in
lab/capstone/runs/<name>.log and the summary is echoed here.
"""

import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
RUNS = LAB / "capstone" / "runs"
REPO = os.environ.get("AR_REPO", "/autoresearch")

# The capstone's fixed protocol. Short enough to run three times in half an hour, long
# enough that the loss visibly moves. Override any of them per run at your own risk - the
# check will notice if the two runs you compare disagree on the budget.
PROTOCOL = {
    "AR_TIME_BUDGET": "120",
    "AR_EVAL_TOKENS": "32768",
    "AR_MAX_SEQ_LEN": "256",
    "AR_DEVICE_BATCH_SIZE": "8",
    "AR_TOTAL_BATCH_SIZE": "2048",
    "AR_UNTIMED_STEPS": "2",
    "AR_DEPTH": "4",
    "AR_HEAD_DIM": "64",
    "AR_SAVE_CHECKPOINT": "0",
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    overrides = dict(kv.split("=", 1) for kv in sys.argv[2:])

    env = dict(os.environ)
    env.update(PROTOCOL)
    env.update(overrides)

    RUNS.mkdir(parents=True, exist_ok=True)
    log_path = RUNS / f"{name}.log"

    print(f"Running {name} with {overrides or 'the default protocol'} ...")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, "train.py"],
            cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
        )

    text = log_path.read_text(encoding="utf-8")
    tail = text.strip().splitlines()[-11:]
    print("\n".join(tail))
    print(f"\nSaved to capstone/runs/{name}.log (exit code {proc.returncode})")


if __name__ == "__main__":
    main()
