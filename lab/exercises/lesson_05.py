# Lesson 05 - Attention: queries, keys, values, and the mask
# Read lessons/module-02/lesson-02.md before filling this in.

import math

import torch


def manual_attention(q, k, v, window):
    """Causal, windowed, scaled dot-product attention, written out by hand.

    q, k, v: (B, T, n_head, head_dim)
    window:  int. Key j is visible from query i iff 0 <= i - j <= window.
             window >= T means "causal only".
    Returns: (B, T, n_head, head_dim)

    No F.scaled_dot_product_attention - the point is to write the four lines:
    scaled scores, mask, softmax, weighted sum of values.
    """
   
    # TODO: transpose to (B, n_head, T, head_dim), score, mask, softmax,
    head_dim = q.shape[-1]
    T = q.shape[1]
    k = k.transpose(1,2)
    q = q.transpose(1,2)
    
    k = k.transpose(2,3)
    scores = (q @ k) / math.sqrt(head_dim)
    positions = torch.arange(T)
    delta = positions[:, None] - positions[None, :]
    masked = (delta < 0) | (delta > window)
   # scores[masked] = -float("inf")
    scores = scores.masked_fill(masked, -float("inf"))
    attn = torch.softmax(scores, dim=-1)
    v = v.transpose(1,2)
    out = attn @ v
    return out.transpose(1,2)
    
