# Lesson 14 - Reading the training log
# Read lessons/module-05/lesson-02.md before filling this in.


def flops_per_token(model):
    """Reproduce model.estimate_flops() without calling it.

    6 * (all parameters EXCEPT wte, value_embeds, resid_lambdas, x0_lambdas)
    plus, per layer, 12 * n_head * head_dim * min(window, sequence_len).
    A negative window means "no limit".
    """
    # TODO: count parameters, subtract the excluded ones, add the attention term.
    nparams = sum(p.numel() for p in model.parameters())
    #m.config.n_layer
    nparams_exclude = sum(p.numel() for p in model.transformer.wte.parameters()) + sum(p.numel() for p in model.value_embeds.parameters()) + model.resid_lambdas.numel() + model.x0_lambdas.numel()
    config = model.config
    h = config.n_head
    q = config.n_embd // config.n_head
    t = config.sequence_len
    attn_flops = sum(12 * h * q * min(window, t) for window,_ in model.window_sizes)
   
    return 6 * (nparams - nparams_exclude) + attn_flops


def mfu_percent(flops_per_token, tokens, seconds, peak_flops):
    """Model FLOPs Utilisation, as a percentage.

    useful FLOPs performed / FLOPs the hardware could have performed.
    Return 0.0 if seconds or peak_flops is <= 0.
    """
    # TODO: one division, times 100.
   
    return 0.0 if seconds <=0 or peak_flops <=0 else (100 * flops_per_token * tokens / seconds / peak_flops)
