# Lesson 10 - Loss, gradients, and what backward() actually does
# Read lessons/module-04/lesson-01.md before filling this in.

import torch


def grad_report(model, x, y) -> dict:
    """Run one forward+backward pass and report on the gradients.

    Returns:
        loss         float, the mean cross-entropy for this batch
        grad_norm    float, the L2 norm of ALL gradients as one vector
        n_with_grad  int, how many parameter tensors have a .grad
        n_params     int, how many parameter tensors the model has

    Clear old gradients first, so calling this twice gives the same answer.
    """
    # TODO: zero_grad, forward, backward, then measure.
    model.zero_grad(set_to_none = True)
    loss = model(x,y)
    loss.backward()
  
    n_with_grad = sum(p.grad is not None for p in model.parameters())
    n_params = sum(1 for p in model.parameters())
    grad_norm = sum(
        (p.grad.norm() ** 2)
        for p in model.parameters()
        if p.grad is not None  
    ).sqrt().item()
    return {"loss": loss, "grad_norm": grad_norm, "n_with_grad": n_with_grad, "n_params": n_params}