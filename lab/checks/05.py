import torch
from _lib import check, check_close, done, stub_guard

from exercises.lesson_05 import manual_attention
from lib.common import train_defs

defs = train_defs()
manual_attention = stub_guard(manual_attention, "manual_attention")

torch.manual_seed(0)
B, T, H, D = 2, 16, 3, 8
q, k, v = (torch.randn(B, T, H, D) for _ in range(3))

out = manual_attention(q, k, v, T)
check(tuple(out.shape) == (B, T, H, D), "output keeps the (B, T, n_head, head_dim) layout")

# 1. Agreement with the repo's own attention(), at several windows.
for window in (T, T // 2, 3, 1):
    with torch.no_grad():
        want = defs.attention(q, k, v, (window, 0))
    got = manual_attention(q, k, v, window)
    check_close(got, want, f"matches train.py attention() at window={window}", tol=1e-5)

# 2. Causality, tested the only way that counts: change the future, see if the past moves.
q2, k2, v2 = q.clone(), k.clone(), v.clone()
k2[:, -1] += 10.0
v2[:, -1] += 10.0
before = manual_attention(q, k, v, T)
after = manual_attention(q2, k2, v2, T)
check_close(before[:, :-1], after[:, :-1], "editing the last token leaves every earlier position untouched", tol=1e-6)
check(not torch.allclose(before[:, -1], after[:, -1]), "...but the last position itself does change")

# 3. Position 0 can only see itself, so its output is exactly v[0].
check_close(manual_attention(q, k, v, T)[:, 0], v[:, 0], "position 0 attends only to itself, so out[0] == v[0]", tol=1e-5)

# 4. A window of 0 means "self only" everywhere.
check_close(manual_attention(q, k, v, 0), v, "window=0 makes every position return its own value", tol=1e-5)

done("05")
