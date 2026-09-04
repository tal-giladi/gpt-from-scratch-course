import math

import torch
from _lib import check, done, stub_guard

from exercises.lesson_03 import bits_per_byte, split_row, uniform_loss
from lib.common import build_model, get_batch, tokenizer

split_row = stub_guard(split_row, "split_row")
uniform_loss = stub_guard(uniform_loss, "uniform_loss")
bits_per_byte = stub_guard(bits_per_byte, "bits_per_byte")

# --- split_row -------------------------------------------------------------
inp, tgt = split_row([10, 11, 12, 13, 14])
check(inp == [10, 11, 12, 13] and tgt == [11, 12, 13, 14], "split_row shifts a list by one")

row = torch.arange(9)
inp, tgt = split_row(row)
check(
    isinstance(inp, torch.Tensor) and torch.equal(inp, row[:-1]) and torch.equal(tgt, row[1:]),
    "split_row works on a tensor without converting it",
)
check(len(inp) == len(tgt) == len(row) - 1, "a row of T+1 gives T inputs and T targets")

# --- uniform_loss ----------------------------------------------------------
V = tokenizer().get_vocab_size()
check(abs(uniform_loss(V) - math.log(V)) < 1e-9, f"uniform_loss({V}) == ln({V}) == {math.log(V):.4f}")
check(abs(uniform_loss(2) - math.log(2)) < 1e-9, "uniform_loss(2) == ln(2): one bit of uncertainty")

# The claim that makes it real: an untrained model on real data scores exactly this.
x, y = get_batch(B=2, T=128)
loss = build_model()(x, y).item()
check(abs(loss - uniform_loss(V)) < 0.05, f"an untrained model's real loss ({loss:.4f}) matches uniform_loss")

# --- bits_per_byte ---------------------------------------------------------
check(abs(bits_per_byte(math.log(2), 1) - 1.0) < 1e-9, "ln(2) nats over 1 byte == 1.0 bits per byte")
check(abs(bits_per_byte(8 * math.log(2), 4) - 2.0) < 1e-9, "the sums are divided, not averaged (2.0 bpb)")
check(bits_per_byte(0.0, 10) == 0.0, "a perfect model scores 0.0 bpb")
check(bits_per_byte(123.0, 0) == float("inf"), "no bytes means inf, not 0.0")

# End to end on real data: the untrained model must score close to ln(V)/ln(2)/bytes-per-token.
from prepare import get_token_bytes  # noqa: E402

token_bytes = get_token_bytes()
nbytes = token_bytes[y.view(-1)]
mask = nbytes > 0
# The mean loss times the number of scored tokens is their summed loss in nats.
total_nats = loss * mask.sum().item()
bpb = bits_per_byte(total_nats, nbytes.sum().item())
check(2.5 < bpb < 6.0, f"an untrained model scores a plausible {bpb:.2f} bits per byte on real text")

done("03")
