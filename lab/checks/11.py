import torch
from _lib import check, check_close, done, stub_guard

from exercises.lesson_11 import adamw_update, nesterov_momentum
from lib.common import train_defs

defs = train_defs()
adamw_update = stub_guard(adamw_update, "adamw_update")
nesterov_momentum = stub_guard(nesterov_momentum, "nesterov_momentum")

torch.manual_seed(0)
LR, B1, B2, EPS, WD = 0.01, 0.8, 0.95, 1e-10, 0.1


def t(v):
    return torch.tensor(float(v))


# --- adamw_update: three consecutive steps must track the repo's kernel exactly ---------
p_mine = torch.randn(16, 8)
p_ref = p_mine.clone()
m_mine, v_mine = torch.zeros_like(p_mine), torch.zeros_like(p_mine)
m_ref, v_ref = torch.zeros_like(p_ref), torch.zeros_like(p_ref)

for step in (1, 2, 3):
    grad = torch.randn(16, 8) * (0.1 * step)
    adamw_update(p_mine, grad.clone(), m_mine, v_mine, step, LR, B1, B2, EPS, WD)
    with torch.no_grad():
        defs.adamw_step_fused(p_ref, grad.clone(), m_ref, v_ref, t(step), t(LR), t(B1), t(B2), t(EPS), t(WD))
    check_close(p_mine, p_ref, f"parameter matches train.py's AdamW after step {step}", tol=1e-6)
    check_close(m_mine, m_ref, f"exp_avg matches after step {step}", tol=1e-6)
    check_close(v_mine, v_ref, f"exp_avg_sq matches after step {step}", tol=1e-6)

# Weight decay really is decoupled: with a zero gradient, p just shrinks by (1 - lr*wd).
p = torch.ones(4)
adamw_update(p, torch.zeros(4), torch.zeros(4), torch.zeros(4), 1, LR, B1, B2, EPS, WD)
check_close(p, torch.full((4,), 1 - LR * WD), "a zero gradient leaves only the weight decay", tol=1e-6)

# --- nesterov_momentum -----------------------------------------------------------------
buf = torch.randn(6)
grads = torch.randn(6)
buf_before, grads_before = buf.clone(), grads.clone()
out = nesterov_momentum(buf, grads, 0.95)

expected_buf = buf_before * 0.95 + grads_before * 0.05
check_close(buf, expected_buf, "momentum_buffer becomes momentum*buf + (1-momentum)*grad", tol=1e-6)
check_close(out, grads_before * 0.05 + expected_buf * 0.95, "the returned direction looks one step ahead (Nesterov)", tol=1e-6)
check(torch.equal(grads, grads_before), "grads is left untouched")

# With momentum 0 there is no smoothing at all.
buf = torch.zeros(6)
g = torch.randn(6)
check_close(nesterov_momentum(buf, g, 0.0), g, "momentum=0 returns the raw gradient", tol=1e-6)

done("11")
