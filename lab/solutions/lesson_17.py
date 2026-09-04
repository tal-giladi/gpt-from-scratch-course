# Lesson 17 - reference solution.

import torch


def window_mask(T, window):
    i = torch.arange(T)
    delta = i[:, None] - i[None, :]
    return (delta >= 0) & (delta <= window)


def mask_memory_bytes(B, n_head, T, bytes_per_element=4):
    return int(B * n_head * T * T * bytes_per_element)
