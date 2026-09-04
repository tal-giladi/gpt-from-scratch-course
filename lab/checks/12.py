import torch
from _lib import check, done, stub_guard

from exercises.lesson_12 import accumulated_grad_norm, lr_multiplier
from lib.common import build_model, get_batch

lr_multiplier = stub_guard(lr_multiplier, "lr_multiplier")
accumulated_grad_norm = stub_guard(accumulated_grad_norm, "accumulated_grad_norm")

# --- lr_multiplier: the repo's own settings (no warmup, half warmdown, to zero) ---------
EXPECTED = {0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.6: 0.8, 0.75: 0.5, 0.9: 0.2, 1.0: 0.0}
for progress, want in EXPECTED.items():
    got = lr_multiplier(progress)
    check(abs(got - want) < 1e-9, f"lr_multiplier({progress}) == {want} (got {got:.4f})")

# With a warmup phase.
check(abs(lr_multiplier(0.0, warmup=0.1) - 0.0) < 1e-9, "with warmup=0.1, progress 0 starts at 0.0")
check(abs(lr_multiplier(0.05, warmup=0.1) - 0.5) < 1e-9, "...and is halfway up at progress 0.05")
check(abs(lr_multiplier(0.2, warmup=0.1) - 1.0) < 1e-9, "...and flat once warmup is over")

# A non-zero floor.
check(abs(lr_multiplier(1.0, final_frac=0.1) - 0.1) < 1e-9, "final_frac=0.1 ends at 0.1, not 0")
check(abs(lr_multiplier(0.75, final_frac=0.1) - 0.55) < 1e-9, "...and interpolates linearly to it")

# Monotonic on the way down.
values = [lr_multiplier(i / 20) for i in range(21)]
check(all(a >= b - 1e-12 for a, b in zip(values, values[1:])), "the schedule never goes back up")

# --- gradient accumulation must be exact -----------------------------------------------
x, y = get_batch(B=4, T=64)

model = build_model()
one_shot = accumulated_grad_norm(model, x, y, 1)

for micro in (2, 4):
    model = build_model()
    got = accumulated_grad_norm(model, x, y, micro)
    check(
        abs(got - one_shot) < 1e-4 * max(1.0, one_shot),
        f"{micro} micro-batches give the same gradient as one big batch ({got:.6f} vs {one_shot:.6f})",
    )

# And the bug this exists to catch: forgetting to divide by the number of micro-batches
# would multiply the gradient by that number.
check(one_shot > 0, "the gradient norm is non-zero, so the comparison is meaningful")
model = build_model()
model.zero_grad(set_to_none=True)
for xb, yb in zip(x.chunk(4), y.chunk(4)):
    model(xb, yb).backward()  # deliberately WITHOUT the /4
unscaled = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None).sqrt().item()
check(
    abs(unscaled / one_shot - 4.0) < 0.05,
    f"...and that mistake really would inflate the gradient 4x ({unscaled / one_shot:.2f}x), which your code avoids",
)

done("12")
