import torch
from _lib import check, check_close, done, stub_guard

from exercises.lesson_07 import block_forward, mlp_forward
from lib.common import build_model, train_defs

defs = train_defs()
mlp_forward = stub_guard(mlp_forward, "mlp_forward")
block_forward = stub_guard(block_forward, "block_forward")

model = build_model()
block = model.transformer.h[0]
n_embd = model.config.n_embd
torch.manual_seed(0)
x = torch.randn(2, 8, n_embd)
cos_sin = (model.cos[:, :8], model.sin[:, :8])
window = model.window_sizes[0]

# --- mlp_forward -----------------------------------------------------------
with torch.no_grad():
    check_close(mlp_forward(block.mlp, x), block.mlp(x), "mlp_forward matches the real MLP module", tol=1e-5)

# c_proj is zero at init, so give it real weights and check again - otherwise both sides
# are trivially zero and the test proves nothing.
with torch.no_grad():
    block.mlp.c_proj.weight.normal_(0, 0.02)
    got = mlp_forward(block.mlp, x)
    check_close(got, block.mlp(x), "...still matches once c_proj is non-zero", tol=1e-5)
    check(got.abs().max() > 0, "the output is not all zeros, so the comparison means something")

# --- block_forward ---------------------------------------------------------
with torch.no_grad():
    check_close(block_forward(block, x, None, cos_sin, window), block(x, None, cos_sin, window),
                "block_forward matches the real Block module", tol=1e-5)

# The identity property, on a freshly initialised block.
fresh = build_model().transformer.h[0]
with torch.no_grad():
    out = block_forward(fresh, x, None, cos_sin, window)
check_close(out, x, "at initialisation the block is the identity (both c_proj are zero)", tol=1e-6)

with torch.no_grad():
    fresh.attn.c_proj.weight.normal_(0, 0.02)
    moved = block_forward(fresh, x, None, cos_sin, window)
check(not torch.allclose(moved, x, atol=1e-6), "...and stops being the identity as soon as attn.c_proj is non-zero")

done("07")
