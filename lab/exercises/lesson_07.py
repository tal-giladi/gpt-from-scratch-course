# Lesson 07 - The MLP, the block, and zero init
# Read lessons/module-03/lesson-01.md before filling this in.

import torch

from lib.common import train_defs


def mlp_forward(mlp, x):
    """Run the MLP by hand: up-project, ReLU squared, down-project.

    mlp has .c_fc (n_embd -> 4*n_embd) and .c_proj (4*n_embd -> n_embd),
    both callable nn.Linear layers with no bias.
    """
    # TODO: three lines.
    raise NotImplementedError


def block_forward(block, x, ve, cos_sin, window_size):
    """Run one transformer block by hand.

        x = x + attn(norm(x), ve, cos_sin, window_size)
        x = x + mlp(norm(x))

    Use block.attn and block.mlp; norm is train_defs().norm.
    """
    # TODO: two residual additions, pre-norm on the way in.
    raise NotImplementedError
