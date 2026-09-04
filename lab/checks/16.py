import torch
from _lib import check, done, fail, stub_guard

from exercises.lesson_16 import benchmark_matmul

benchmark_matmul = stub_guard(benchmark_matmul, "benchmark_matmul")

r = benchmark_matmul(512, torch.float32, iters=5)
if not isinstance(r, dict):
    fail("benchmark_matmul must return a dict")
for key in ("n", "dtype", "seconds", "gflops"):
    if key not in r:
        fail(f"missing key {key!r}")

check(r["n"] == 512, "n is echoed back")
check(r["seconds"] > 0, f"seconds is positive ({r['seconds']*1000:.2f} ms per iteration)")
expected_gflops = 2 * 512 ** 3 / r["seconds"] / 1e9
check(
    abs(r["gflops"] - expected_gflops) < 1e-6 * max(1.0, expected_gflops),
    f"gflops == 2*n^3 / seconds / 1e9 ({r['gflops']:.1f} GFLOP/s measured here)",
)
check(0.1 < r["gflops"] < 5000, "the measured throughput is in a physically plausible range")

# Bigger matrices amortise the per-call overhead, so they reach a higher fraction of peak.
small = benchmark_matmul(64, torch.float32, iters=50)
large = benchmark_matmul(1024, torch.float32, iters=3)
check(
    large["gflops"] > small["gflops"],
    f"a 1024x1024 matmul beats a 64x64 one ({large['gflops']:.1f} vs {small['gflops']:.1f} GFLOP/s)",
)

# The warm-up matters: without it, the first-call overhead shows up as a much slower result.
repeat = benchmark_matmul(512, torch.float32, iters=5)
ratio = max(r["gflops"], repeat["gflops"]) / min(r["gflops"], repeat["gflops"])
check(ratio < 3.0, f"two identical runs agree within 3x ({ratio:.2f}x) - the warm-up is doing its job")

# bfloat16 on this CPU: measured, not assumed.
bf = benchmark_matmul(512, torch.bfloat16, iters=5)
check(bf["gflops"] > 0, f"bfloat16 also runs here, at {bf['gflops']:.1f} GFLOP/s vs float32's {r['gflops']:.1f}")
print(f"  [note] bf16/fp32 throughput ratio on this machine: {bf['gflops'] / r['gflops']:.2f}x")

done("16")
