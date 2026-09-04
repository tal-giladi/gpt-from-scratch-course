# Lesson 03 - reference solution.

import math


def split_row(row):
    return row[:-1], row[1:]


def uniform_loss(vocab_size: int) -> float:
    return math.log(vocab_size)


def bits_per_byte(total_nats: float, total_bytes: int) -> float:
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)
