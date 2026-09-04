# Lesson 07 - reference solution.

import torch

from lib.common import train_defs


def mlp_forward(mlp, x):
    h = mlp.c_fc(x)
    h = torch.relu(h).square()
    return mlp.c_proj(h)


def block_forward(block, x, ve, cos_sin, window_size):
    norm = train_defs().norm
    x = x + block.attn(norm(x), ve, cos_sin, window_size)
    x = x + block.mlp(norm(x))
    return x
