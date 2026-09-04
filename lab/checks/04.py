import torch
from _lib import check, check_close, done, fail, stub_guard

from exercises.lesson_04 import residual_stream
from lib.common import build_model, get_batch, train_defs

defs = train_defs()
norm = defs.norm
residual_stream = stub_guard(residual_stream, "residual_stream")

model = build_model()
idx, _ = get_batch(B=2, T=64)

with torch.no_grad():
    streams = residual_stream(model, idx)

if not isinstance(streams, (list, tuple)):
    fail("residual_stream must return a list of tensors")

n_layer = model.config.n_layer
check(len(streams) == n_layer + 1, f"one stream per stage: n_layer + 1 == {n_layer + 1}")
check(
    all(tuple(s.shape) == (2, 64, model.config.n_embd) for s in streams),
    f"every stream is (B, T, n_embd) == (2, 64, {model.config.n_embd})",
)

with torch.no_grad():
    expected_x0 = norm(model.transformer.wte(idx))
check_close(streams[0], expected_x0, "stream[0] is the normalised token embedding", tol=1e-5)

check(
    not torch.allclose(streams[-1], streams[0], atol=1e-6),
    "the blocks actually changed the stream",
)

# The real test: your final stream must reproduce the model's own logits.
with torch.no_grad():
    softcap = 15
    logits_from_yours = model.lm_head(norm(streams[-1])).float()
    logits_from_yours = softcap * torch.tanh(logits_from_yours / softcap)
    logits_from_model = model(idx)
check_close(
    logits_from_yours,
    logits_from_model,
    "final norm + lm_head + softcap on YOUR stream reproduces model(idx) exactly",
    tol=1e-4,
)

done("04")
