import math

import torch
from _lib import check, done, fail, stub_guard

from exercises.lesson_10 import grad_report
from lib.common import build_model, get_batch

grad_report = stub_guard(grad_report, "grad_report")

model = build_model()
x, y = get_batch(B=2, T=64)

r = grad_report(model, x, y)
if not isinstance(r, dict):
    fail("grad_report must return a dict")

V = model.config.vocab_size
check(abs(r["loss"] - math.log(V)) < 0.05, f"loss at init is ln(vocab) == {math.log(V):.4f}, got {r['loss']:.4f}")
check(r["grad_norm"] > 0 and math.isfinite(r["grad_norm"]), f"grad_norm is finite and positive ({r['grad_norm']:.4f})")
check(r["n_params"] == len(list(model.parameters())), f"n_params == {len(list(model.parameters()))}")
check(r["n_with_grad"] == r["n_params"], "every parameter tensor received a gradient")

# The norm really is over all gradients together.
expected = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters()))
check(abs(r["grad_norm"] - expected) < 1e-3 * max(1.0, expected), "grad_norm is the L2 norm of all gradients as one vector")

# Called twice, the answer must not double: old gradients have to be cleared.
again = grad_report(model, x, y)
check(abs(again["grad_norm"] - r["grad_norm"]) < 1e-3, "calling it twice gives the same answer (gradients were cleared)")

# The lesson-07 claim, on the real model: a zero weight still gets a gradient.
c_proj = model.transformer.h[0].mlp.c_proj.weight
check(c_proj.abs().max().item() == 0.0, "mlp.c_proj is still exactly zero (it has not been updated)")
check(c_proj.grad.abs().max().item() > 0.0, "...and it has a NON-zero gradient, so it will start learning immediately")

# The other half of the same story: c_fc gets NO gradient on the very first step, because
# the gradient reaching it has to pass back through the (still zero) c_proj.
c_fc = model.transformer.h[0].mlp.c_fc.weight
check(c_fc.grad.abs().max().item() == 0.0, "the up-projection gets exactly zero gradient while c_proj is zero")

with torch.no_grad():
    model.transformer.h[0].mlp.c_proj.weight.normal_(0, 0.02)
grad_report(model, x, y)
check(c_fc.grad.abs().max().item() > 0.0, "...and starts receiving one the moment c_proj is non-zero")

done("10")
