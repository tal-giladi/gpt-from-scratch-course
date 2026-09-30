# Lesson 13 - val_bpb, the fixed metric
# Read lessons/module-05/lesson-01.md before filling this in.

import math

import torch


def bpb_over_batches(model, batches, token_bytes):
    """Bits per byte over a sequence of (x, y) batches.

    For each batch: per-position loss with reduction="none", look up how many
    UTF-8 bytes each TARGET token decodes to, drop the positions whose byte
    count is 0 (the special tokens) from the nats sum, and accumulate both
    sums across all batches.

    Return total_nats / (ln(2) * total_bytes), or inf if no bytes were scored.
    """
    # TODO: accumulate the two sums, then divide once at the end.
    total_bytes = 0
    total_nats = 0.0
    for x,y in batches:
        loss_flat = model(x, y, reduction = "none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes>0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
        
    

    return (float("inf") if total_bytes == 0 else (total_nats / (math.log(2) * total_bytes)))
