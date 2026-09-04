# Lesson 10 - reference solution.

import torch


def grad_report(model, x, y) -> dict:
    model.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    params = list(model.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    total_sq = sum(g.pow(2).sum() for g in grads)
    return {
        "loss": loss.item(),
        "grad_norm": total_sq.sqrt().item(),
        "n_with_grad": len(grads),
        "n_params": len(params),
    }
