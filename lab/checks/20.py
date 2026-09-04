import ast
from pathlib import Path

from _lib import LAB_DIR, check, done, fail, stub_guard

from exercises.lesson_20 import capstone_report

capstone_report = stub_guard(capstone_report, "capstone_report")

RUNS = Path(LAB_DIR) / "capstone" / "runs"
VALID_EXPECT = {"lower", "higher", "no change"}
VALID_CONCLUSION = {"confirmed", "refuted", "inconclusive"}


def parse_log(name):
    path = RUNS / f"{name}.log"
    if not path.exists():
        fail(
            f"capstone/runs/{name}.log is missing - run it first:\n"
            f"        bash lab/lab.sh shell\n"
            f"        python tools/run_experiment.py {name}"
        )
    text = path.read_text(encoding="utf-8")
    out = {"status": "crashed", "config": {}, "budget": None}
    for line in text.splitlines():
        line = line.strip()
        if line == "FAIL":
            out["status"] = "diverged"
        elif line.startswith("Model config:"):
            out["config"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("Time budget:"):
            out["budget"] = line.split(":", 1)[1].strip()
        elif line.startswith("val_bpb:"):
            out["val_bpb"] = float(line.split(":", 1)[1])
            out["status"] = "ok"
    return out


report = capstone_report()
if not isinstance(report, dict):
    fail("capstone_report must return a dict")
for key in ("hypothesis", "changed", "expectation", "baseline", "repeat", "variant", "conclusion"):
    if key not in report:
        fail(f"the report is missing {key!r}")

check(len(str(report["hypothesis"]).strip()) >= 20, "the hypothesis is a real sentence, not a placeholder")
check(str(report["changed"]).startswith("AR_"), f"'changed' names an AR_* knob (got {report['changed']!r})")
check(report["expectation"] in VALID_EXPECT, f"expectation is one of {sorted(VALID_EXPECT)}")
check(report["conclusion"] in VALID_CONCLUSION, f"conclusion is one of {sorted(VALID_CONCLUSION)}")

baseline = parse_log(report["baseline"])
repeat = parse_log(report["repeat"])
variant = parse_log(report["variant"])

for name, run in (("baseline", baseline), ("repeat", repeat), ("variant", variant)):
    check(run["status"] == "ok", f"the {name} run completed and reported a val_bpb ({run['status']})")

check(baseline["budget"] == repeat["budget"] == variant["budget"],
      f"all three runs used the same time budget ({baseline['budget']})")
check(baseline["config"] == repeat["config"], "the two baseline runs used an identical model config")

differing = {k for k in set(baseline["config"]) | set(variant["config"])
             if baseline["config"].get(k) != variant["config"].get(k)}
if report["changed"] in ("AR_BF16", "AR_TOTAL_BATCH_SIZE", "AR_DEVICE_BATCH_SIZE", "AR_UNTIMED_STEPS"):
    check(differing == set(), f"{report['changed']} changes the training loop, not the model config")
else:
    check(differing != set(), f"the variant's model config differs from the baseline (you changed {report['changed']})")

# One knob may legitimately move several config fields - AR_DEPTH changes n_layer, and
# n_embd and n_head with it. What must NOT move is the ruler: same vocabulary, same context.
for field in ("vocab_size", "sequence_len"):
    check(
        baseline["config"].get(field) == variant["config"].get(field),
        f"{field} is identical in both runs - the evaluation must not move between them",
    )

noise = abs(baseline["val_bpb"] - repeat["val_bpb"])
delta = variant["val_bpb"] - baseline["val_bpb"]
print(f"  [data] baseline {baseline['val_bpb']:.4f} | repeat {repeat['val_bpb']:.4f} "
      f"| variant {variant['val_bpb']:.4f}")
print(f"  [data] measured noise floor {noise:.4f}, variant moved {delta:+.4f}")

if abs(delta) <= noise:
    truth = "inconclusive"
elif (delta < 0 and report["expectation"] == "lower") or (delta > 0 and report["expectation"] == "higher"):
    truth = "confirmed"
elif report["expectation"] == "no change":
    truth = "refuted"
else:
    truth = "refuted"

check(
    report["conclusion"] == truth,
    f"the conclusion matches the data: it says {truth!r}"
    + ("" if report["conclusion"] == truth else f", you wrote {report['conclusion']!r}"),
)

done("20 - capstone")
