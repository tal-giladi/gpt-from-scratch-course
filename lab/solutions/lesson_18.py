# Lesson 18 - reference solution.


def parse_summary(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines()]
    if any(ln == "FAIL" for ln in lines):
        return {"status": "diverged"}

    wanted_int = {"num_steps", "depth"}
    wanted_float = {
        "val_bpb", "training_seconds", "total_seconds", "mfu_percent",
        "peak_vram_mb", "total_tokens_M", "num_params_M",
    }
    out = {}
    for line in lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        try:
            if key in wanted_int:
                out[key] = int(float(value))
            elif key in wanted_float:
                out[key] = float(value)
        except ValueError:
            continue
    if "val_bpb" not in out:
        return {"status": "crashed"}
    out["status"] = "ok"
    return out
