# Lesson 17 - FlashAttention, and reading a port
# Read lessons/module-06/lesson-02.md before filling this in.

import torch


def window_mask(T, window):
    """The (T, T) boolean mask the CPU port has to build by hand.

    True at [i, j] when key j is visible from query i, i.e. 0 <= i - j <= window.
    Write it yourself; do not import _causal_window_mask.
    """
    # TODO: one arange, one difference, two conditions.
    raise NotImplementedError


def mask_memory_bytes(B, n_head, T, bytes_per_element=4):
    """Bytes of the score tensor SDPA must materialise when an explicit mask
    forces it onto the general math kernel: B * n_head * T * T elements.
    """
    # TODO: multiply out and return an int.
    raise NotImplementedError
