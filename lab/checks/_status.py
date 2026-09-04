"""`bash lab/lab.sh status` — run every check in one container and report a tick list."""

import io
import contextlib
import runpy
import sys
from pathlib import Path

CHECKS = sorted(p for p in Path(__file__).parent.glob("[0-9][0-9].py"))

passed = 0
for path in CHECKS:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            runpy.run_path(str(path), run_name="__main__")
        mark, passed = "x", passed + 1
    except SystemExit as e:
        mark = "x" if e.code in (0, None) else " "
        passed += 1 if mark == "x" else 0
    except Exception:
        mark = " "
    print(f"  [{mark}] lesson {path.stem}")

print(f"\n{passed}/{len(CHECKS)} exercises passing.")
sys.exit(0 if passed == len(CHECKS) else 1)
