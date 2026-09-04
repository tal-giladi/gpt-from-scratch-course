# Lesson 06 - Rotary embeddings and RMS norm
# Read lessons/module-02/lesson-03.md before filling this in.

import torch


def apply_rotary(x, cos, sin):
    """Rotate the (B, T, H, head_dim) tensor x by the precomputed angles.

    The pairing is FIRST HALF with SECOND HALF:
        x1 = x[..., :d], x2 = x[..., d:]   with d = head_dim // 2
        y1 = x1*cos + x2*sin
        y2 = -x1*sin + x2*cos
    and the result is [y1, y2] concatenated on the last dimension.

    cos and sin are (1, T, 1, d) and broadcast on their own.
    """
    # TODO: split, rotate, concatenate.
    raise NotImplementedError


def rms_norm(x, eps=1e-6):
    """Root-mean-square normalisation over the last dimension.

    x / sqrt(mean(x^2) + eps) - no mean subtraction, no learned gain, no bias.
    Write the arithmetic; do not call F.rms_norm.
    """
    # TODO: one line with x.pow(2).mean(-1, keepdim=True).
    raise NotImplementedError
