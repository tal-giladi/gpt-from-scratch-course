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
    raise NotImplementedError
