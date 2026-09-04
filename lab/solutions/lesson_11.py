# Lesson 11 - reference solution.

import torch


def adamw_update(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
    p.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-lr / bias1)
    return p


def nesterov_momentum(momentum_buffer, grads, momentum):
    momentum_buffer.lerp_(grads, 1 - momentum)
    return torch.lerp(grads, momentum_buffer, momentum)
