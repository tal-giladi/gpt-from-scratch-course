from _lib import check, done, stub_guard

from exercises.lesson_14 import flops_per_token, mfu_percent
from lib.common import build_model, toy_config

flops_per_token = stub_guard(flops_per_token, "flops_per_token")
mfu_percent = stub_guard(mfu_percent, "mfu_percent")

CONFIGS = [
    ("course scale", toy_config()),
    ("deeper", toy_config(n_layer=6)),
    ("wider", toy_config(n_embd=384)),
    ("long context", toy_config(sequence_len=512)),
    ("all long windows", toy_config(window_pattern="L")),
]

for name, config in CONFIGS:
    model = build_model(config)
    got, want = flops_per_token(model), model.estimate_flops()
    check(got == want, f"{name}: flops_per_token == estimate_flops() == {want:,}")

# The embeddings really are excluded: a bigger vocabulary changes lm_head but nothing else,
# so the FLOPs must rise by exactly 6 * n_embd * (delta vocab).
small = build_model(toy_config(vocab_size=1000))
big = build_model(toy_config(vocab_size=2000))
delta = flops_per_token(big) - flops_per_token(small)
check(delta == 6 * 128 * 1000, f"doubling the vocab adds only the lm_head term ({delta:,} FLOPs/token)")

# Halving every window halves the attention term, and nothing else.
# Windows change only the attention term. "S" is half of "L" everywhere except the last
# layer, which is always forced to long - so the difference is exactly the layers in between.
full = build_model(toy_config(window_pattern="L"))
half = build_model(toy_config(window_pattern="S"))
check(flops_per_token(half) < flops_per_token(full), "short windows cost fewer FLOPs per token than long ones")

# --- mfu_percent -----------------------------------------------------------------------
check(abs(mfu_percent(1e6, 1000, 1.0, 1e9) - 100.0) < 1e-9, "1e6 FLOPs x 1000 tokens in 1s against a 1 GFLOP/s peak is 100%")
check(abs(mfu_percent(1e6, 1000, 2.0, 1e9) - 50.0) < 1e-9, "taking twice as long halves it")
check(mfu_percent(1e6, 1000, 0.0, 1e9) == 0.0, "zero seconds returns 0.0 rather than dividing by zero")
check(mfu_percent(1e6, 1000, 1.0, 0.0) == 0.0, "a zero peak returns 0.0 too")

# The real numbers from this fork's own log: ~300 tok/s at ~30 GFLOP/s is a few percent.
model = build_model()
real = mfu_percent(model.estimate_flops(), 2048, 2048 / 300.0, 30e9)
check(0.1 < real < 30.0, f"the course-scale model's MFU on this machine is a plausible {real:.1f}%")

done("14")
