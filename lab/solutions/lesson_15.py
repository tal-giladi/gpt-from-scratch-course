# Lesson 15 - reference solution.


def budget_report(model, seconds, flops_per_second):
    flops_per_token = model.estimate_flops()
    params = sum(p.numel() for p in model.parameters())
    tokens = int(flops_per_second * seconds / flops_per_token)
    return {
        "params": params,
        "flops_per_token": flops_per_token,
        "tokens": tokens,
        "tokens_per_param": tokens / params,
        "chinchilla_params": tokens / 20.0,
    }
