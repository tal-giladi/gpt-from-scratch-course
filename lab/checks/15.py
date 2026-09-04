from _lib import check, done, fail, stub_guard

from lib.common import build_model, toy_config
from exercises.lesson_15 import budget_report

budget_report = stub_guard(budget_report, "budget_report")

model = build_model()
r = budget_report(model, seconds=600, flops_per_second=30e9)
if not isinstance(r, dict):
    fail("budget_report must return a dict")

params = sum(p.numel() for p in model.parameters())
fpt = model.estimate_flops()
check(r["params"] == params, f"params counts every parameter, embeddings included ({params:,})")
check(r["flops_per_token"] == fpt, f"flops_per_token == estimate_flops() == {fpt:,}")
check(r["tokens"] == int(30e9 * 600 / fpt), f"tokens == FLOP budget / FLOPs per token ({r['tokens']:,})")
check(abs(r["tokens_per_param"] - r["tokens"] / params) < 1e-12, "tokens_per_param is tokens / params")
check(abs(r["chinchilla_params"] - r["tokens"] / 20.0) < 1e-9, "chinchilla_params is tokens / 20")

# Halving the budget halves the tokens.
half = budget_report(model, seconds=300, flops_per_second=30e9)
check(abs(half["tokens"] / r["tokens"] - 0.5) < 0.01, "half the time buys half the tokens")

# A bigger model sees fewer tokens for the same budget - the whole trade-off.
deep = build_model(toy_config(n_layer=8))
deep_r = budget_report(deep, seconds=600, flops_per_second=30e9)
check(deep_r["params"] > r["params"], "a deeper model has more parameters")
check(deep_r["tokens"] < r["tokens"], "...and therefore sees fewer tokens in the same budget")
check(deep_r["tokens_per_param"] < r["tokens_per_param"], "...so it is even further from compute-optimal")

# The uncomfortable finding this lesson exists for.
check(r["tokens_per_param"] < 1.0, f"the course-scale model gets {r['tokens_per_param']:.3f} tokens per parameter, against Chinchilla's 20")
check(r["chinchilla_params"] < r["params"] / 100, "the compute-optimal model for this budget is over 100x smaller than the one we train")

done("15")
