"""Imported by every checks/NN.py: puts /lab on sys.path and gives every checker the same
PASS/FAIL vocabulary, so a check reads as a list of claims about your code.
"""

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

import torch  # noqa: E402  (after the sys.path fix, deliberately)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def check(condition: bool, msg: str) -> None:
    ok(msg) if condition else fail(msg)


def check_close(got, want, msg, tol=1e-4):
    """Numeric agreement, for tensors or plain numbers, with a readable failure."""
    got_t = got if isinstance(got, torch.Tensor) else torch.tensor(got)
    want_t = want if isinstance(want, torch.Tensor) else torch.tensor(want)
    if got_t.shape != want_t.shape:
        fail(f"{msg} — shape {tuple(got_t.shape)}, expected {tuple(want_t.shape)}")
    diff = (got_t.float() - want_t.float()).abs().max().item()
    check(diff <= tol, f"{msg} (max abs difference {diff:.2e} <= {tol:.0e})")


def stub_guard(fn, name):
    """Call a student function, turning the untouched stub into a clear message."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            fail(f"{name}() is still a stub — fill in the TODO in its exercise file")
    return wrapped


def done(lesson: str) -> None:
    print(f"Lesson {lesson}: all checks passed.")
