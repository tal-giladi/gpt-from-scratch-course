# Lesson 13 - reference solution.

import math

import torch


@torch.no_grad()
def bpb_over_batches(model, batches, token_bytes):
    total_nats, total_bytes = 0.0, 0
    for x, y in batches:
        loss_flat = model(x, y, reduction="none").view(-1)
        nbytes = token_bytes[y.view(-1)]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)
