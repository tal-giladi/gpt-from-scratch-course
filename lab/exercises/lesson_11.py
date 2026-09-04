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
    raise NotImplementedError


def nesterov_momentum(momentum_buffer, grads, momentum):
    """Muon's momentum, the first two lines of muon_step_fused.

        buf = buf + (1 - momentum) * (grads - buf)     # in place
        return grads + momentum * (buf - grads)        # the update direction

    Update momentum_buffer in place. Do NOT modify grads.
    """
    # TODO: two lerps, one in place and one not.
    raise NotImplementedError
