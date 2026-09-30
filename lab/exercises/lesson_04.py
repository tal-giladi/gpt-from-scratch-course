# Lesson 04 - Embeddings and the residual stream
# Read lessons/module-02/lesson-01.md before filling this in.

from lib.common import train_defs


def residual_stream(model, idx):
    """Re-implement GPT.forward's loop and return the stream at every stage.

    Returns a list of n_layer + 1 tensors, each (B, T, n_embd):
        [0]   the normalised token embedding (this is x0)
        [i+1] the stream after block i has added its contribution

    Per iteration, in this order:
        x = resid_lambdas[i] * x + x0_lambdas[i] * x0
        x = block(x, ve, cos_sin, window_size)
    where ve is model.value_embeds[str(i)](idx) if that key exists else None,
    cos_sin is (model.cos[:, :T], model.sin[:, :T]) and window_size is
    model.window_sizes[i].
    """
    
    x = model.transformer.wte(idx)
    x = train_defs().norm(x)	
    x0 = x
    T = idx.size(1)
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    result = [x]
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        window_size = model.window_sizes[i]
        
        
        x = block(x, ve, cos_sin, window_size)
        result.append(x)
    return result
