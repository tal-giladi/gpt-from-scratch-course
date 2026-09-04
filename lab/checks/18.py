from pathlib import Path

from _lib import LAB_DIR, check, done, fail, stub_guard

from exercises.lesson_18 import parse_summary

parse_summary = stub_guard(parse_summary, "parse_summary")

FIXTURES = Path(LAB_DIR) / "fixtures"


def read(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- a successful run ------------------------------------------------------------------
r = parse_summary(read("run_ok.log"))
if not isinstance(r, dict):
    fail("parse_summary must return a dict")
check(r.get("status") == "ok", "a completed run is status 'ok'")
check(isinstance(r.get("val_bpb"), float) and abs(r["val_bpb"] - 2.399582) < 1e-9, "val_bpb == 2.399582, as a float")
check(r.get("num_steps") == 15 and isinstance(r["num_steps"], int), "num_steps == 15, as an int")
check(r.get("depth") == 4 and isinstance(r["depth"], int), "depth == 4, as an int")
check(abs(r.get("training_seconds", 0) - 150.9) < 1e-9, "training_seconds == 150.9")
check(abs(r.get("num_params_M", 0) - 11.5) < 1e-9, "num_params_M == 11.5")
check("checkpoint" not in r, "the non-numeric 'checkpoint:' line is ignored, not a crash")

# The progress lines and the config dump also contain colons - they must not leak in.
check(not any(k.startswith("step") or k in ("loss", "Model config", "Time budget") for k in r),
      "progress and config lines are not mistaken for summary fields")

# --- a crashed run ---------------------------------------------------------------------
c = parse_summary(read("run_crash.log"))
check(c.get("status") == "crashed", "a traceback with no summary is status 'crashed'")
check("val_bpb" not in c, "a crashed run reports no val_bpb at all")

# --- a diverged run --------------------------------------------------------------------
d = parse_summary(read("run_diverged.log"))
check(d.get("status") == "diverged", "a run that printed FAIL is status 'diverged'")

# --- robustness ------------------------------------------------------------------------
check(parse_summary("").get("status") == "crashed", "empty output is a crash, not an exception")
extended = read("run_ok.log") + "\nnew_metric_we_added: 42.0\n"
e = parse_summary(extended)
check(e.get("status") == "ok", "an unknown extra metric does not break the parser")
check(abs(e["val_bpb"] - 2.399582) < 1e-9, "...and the known fields still parse")

done("18")
