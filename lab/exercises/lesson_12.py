# Lesson 12 - Schedules, the time budget, and gradient accumulation
# Read lessons/module-04/lesson-03.md before filling this in.

import torch


def lr_multiplier(progress, warmup=0.0, warmdown=0.5, final_frac=0.0):
    """The learning-rate multiplier at a given progress in [0, 1].

    - below `warmup`:            linear ramp from 0 to 1 (and 1.0 if warmup == 0)
    - up to `1 - warmdown`:      flat 1.0
    - after that:                linear from 1.0 down to `final_frac` at progress 1
    """
    # TODO: three branches, and do not divide by a zero warmup.
    raise NotImplementedError


def accumulated_grad_norm(model, x, y, micro_batches):
    """Global gradient norm from splitting one batch into micro-batches.

    Split x and y into `micro_batches` equal chunks along the batch dimension,
    run forward+backward on each WITHOUT clearing gradients in between, and
    scale each micro-batch's loss so the result equals what a single
    forward+backward on the whole batch would have produced.
    """
    # TODO: zero once, loop over chunks, divide the loss, backward, then measure.
    raise NotImplementedError
