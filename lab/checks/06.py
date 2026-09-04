import torch
from _lib import check, check_close, done, stub_guard

from exercises.lesson_06 import apply_rotary, rms_norm
from lib.common import build_model, train_defs

defs = train_defs()
apply_rotary = stub_guard(apply_rotary, "apply_rotary")
rms_norm = stub_guard(rms_norm, "rms_norm")

model = build_model()
T = 12
cos, sin = model.cos[:, :T], model.sin[:, :T]
head_dim = model.config.n_embd // model.config.n_head

torch.manual_seed(0)
x = torch.randn(2, T, model.config.n_head, head_dim)

# --- apply_rotary ----------------------------------------------------------
got = apply_rotary(x, cos, sin)
check(tuple(got.shape) == tuple(x.shape), "apply_rotary keeps the input shape")
check_close(got, defs.apply_rotary_emb(x, cos, sin), "matches train.py's apply_rotary_emb", tol=1e-5)
check_close(got.norm(dim=-1), x.norm(dim=-1), "a rotation preserves each vector's length", tol=1e-4)
check_close(got[:, 0], x[:, 0], "position 0 is rotated by angle 0, i.e. unchanged", tol=1e-5)

# The property that makes RoPE worth it: scores depend on RELATIVE distance only.
one = torch.randn(1, 1, 1, head_dim).expand(1, T, 1, head_dim).contiguous()
rot = apply_rotary(one, cos, sin)[0, :, 0]           # (T, head_dim), same vector at each position
dots = rot @ rot.T                                    # dots[m, n] for identical q and k
pairs_same_offset = [(dots[m, m - 3], dots[m + 1, m - 2]) for m in range(3, T - 1)]
check(
    all(abs(a - b) < 1e-4 for a, b in pairs_same_offset),
    "the dot product depends only on the distance between positions, not on where they are",
)
check(
    abs(dots[5, 5 - 3] - dots[5, 5 - 6]) > 1e-3,
    "...and it really does change when the distance changes",
)

# --- rms_norm --------------------------------------------------------------
y = torch.randn(4, 9) * 7.0
n = rms_norm(y)
check(tuple(n.shape) == tuple(y.shape), "rms_norm keeps the input shape")
check_close(n, defs.norm(y), "matches train.py's norm()", tol=1e-4)
check_close(n.pow(2).mean(-1), torch.ones(4), "each row comes out with root-mean-square 1", tol=1e-4)
check_close(rms_norm(y * 100.0), n, "scaling the input by 100 changes nothing (it is scale-invariant)", tol=1e-3)

done("06")
