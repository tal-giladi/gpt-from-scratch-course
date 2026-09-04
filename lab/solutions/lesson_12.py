# Lesson 12 - reference solution.

import torch


def lr_multiplier(progress, warmup=0.0, warmdown=0.5, final_frac=0.0):
    if warmup > 0 and progress < warmup:
        return progress / warmup
    if progress < 1.0 - warmdown:
        return 1.0
    cooldown = (1.0 - progress) / warmdown
    return cooldown * 1.0 + (1.0 - cooldown) * final_frac


def accumulated_grad_norm(model, x, y, micro_batches):
    model.zero_grad(set_to_none=True)
    for xb, yb in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        loss = model(xb, yb) / micro_batches
        loss.backward()
    total_sq = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
    return total_sq.sqrt().item()
