# Lesson 11 - AdamW and Muon
# Read lessons/module-04/lesson-02.md before filling this in.

import torch


def adamw_update(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
    """One AdamW step, in place, in the same order train.py's kernel does it:

        1. decoupled weight decay on p
        2. update exp_avg    (EMA of grad)
        3. update exp_avg_sq (EMA of grad squared)
        4. bias-correct both, then step p

    step counts from 1. eps is added AFTER the square root.
    Mutates p, exp_avg and exp_avg_sq; returns p.
    """
    # TODO: the four stages above.
    p.mul_(1-lr * wd)
    exp_avg.lerp_(grad, 1-beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-lr/bias1)
    return p


def nesterov_momentum(momentum_buffer, grads, momentum):
    """Muon's momentum, the first two lines of muon_step_fused.

        buf = buf + (1 - momentum) * (grads - buf)     # in place
        return grads + momentum * (buf - grads)        # the update direction

    Update momentum_buffer in place. Do NOT modify grads.
    """
    # TODO: two lerps, one in place and one not.
    momentum_buffer.add_((1 - momentum) * (grads - momentum_buffer))
    return grads + momentum * (momentum_buffer - grads)
