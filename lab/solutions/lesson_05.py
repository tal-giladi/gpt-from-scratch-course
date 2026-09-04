# Lesson 05 - reference solution.

import math

import torch


def manual_attention(q, k, v, window):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    T, head_dim = q.size(2), q.size(3)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    i = torch.arange(T, device=q.device)
    delta = i[:, None] - i[None, :]
    allowed = (delta >= 0) & (delta <= window)
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v).transpose(1, 2)
