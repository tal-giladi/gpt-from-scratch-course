# Lesson 03 - What the model is actually asked to do
# Read lessons/module-01/lesson-03.md before filling this in.

import math


def split_row(row):
    """Split one packed row of length T+1 into (inputs, targets), each length T.

    Position i of inputs must predict position i of targets. Works for a list
    or a 1-D tensor - use slicing only, do not convert types.
    """
    # TODO: return the row without its last element, and without its first.
    raise NotImplementedError


def uniform_loss(vocab_size: int) -> float:
    """Cross-entropy in nats of a model that puts equal probability on every
    token in the vocabulary - i.e. the loss you should see at step 0.
    """
    # TODO: one call to math.log.
    raise NotImplementedError


def bits_per_byte(total_nats: float, total_bytes: int) -> float:
    """Convert a SUM of per-token losses in nats, plus the total number of
    target bytes those tokens covered, into bits per byte.

    Return float("inf") when total_bytes is 0.
    """
    # TODO: nats -> bits is a division by ln(2); then divide by the bytes.
    raise NotImplementedError
