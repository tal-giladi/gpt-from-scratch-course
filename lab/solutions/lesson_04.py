# Lesson 04 - reference solution.

from lib.common import train_defs


def residual_stream(model, idx):
    norm = train_defs().norm
    T = idx.size(1)
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    x = norm(model.transformer.wte(idx))
    x0 = x
    streams = [x]
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i])
        streams.append(x)
    return streams
