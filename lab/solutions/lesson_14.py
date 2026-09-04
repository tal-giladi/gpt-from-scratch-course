# Lesson 14 - reference solution.


def flops_per_token(model):
    config = model.config
    nparams = sum(p.numel() for p in model.parameters())
    excluded = (
        model.transformer.wte.weight.numel()
        + sum(ve.weight.numel() for ve in model.value_embeds.values())
        + model.resid_lambdas.numel()
        + model.x0_lambdas.numel()
    )
    h = config.n_head
    q = config.n_embd // config.n_head
    t = config.sequence_len
    attn_flops = 0
    for window, _ in model.window_sizes:
        effective = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective
    return 6 * (nparams - excluded) + attn_flops


def mfu_percent(flops_per_token, tokens, seconds, peak_flops):
    if seconds <= 0 or peak_flops <= 0:
        return 0.0
    return 100.0 * flops_per_token * tokens / seconds / peak_flops
