import math

import torch
from _lib import check, done, stub_guard

from exercises.lesson_13 import bpb_over_batches
from lib.common import build_model, tokenizer

from prepare import EVAL_TOKENS, MAX_SEQ_LEN, evaluate_bpb, get_token_bytes, make_dataloader

bpb_over_batches = stub_guard(bpb_over_batches, "bpb_over_batches")

model = build_model()
tok = tokenizer()
token_bytes = get_token_bytes()
B = 8
steps = EVAL_TOKENS // (B * MAX_SEQ_LEN)

loader = make_dataloader(tok, B, MAX_SEQ_LEN, "val")
batches = []
for _ in range(steps):
    x, y, _ = next(loader)
    batches.append((x.clone(), y.clone()))

got = bpb_over_batches(model, batches, token_bytes)
want = evaluate_bpb(model, tok, B)

check(isinstance(got, float) and math.isfinite(got), f"returns a finite float ({got:.6f})")
check(abs(got - want) < 1e-6, f"matches prepare.evaluate_bpb over the same {steps} batches ({got:.6f} vs {want:.6f})")
check(3.0 < got < 4.0, "an untrained model lands near ln(vocab)/ln(2) spread over the average token's bytes")

# Special tokens must be excluded: forcing every target to BOS leaves nothing to score.
bos = tok.get_bos_token_id()
check(token_bytes[bos].item() == 0, "the BOS token has a byte length of 0")
all_bos = [(x, torch.full_like(y, bos)) for x, y in batches[:1]]
check(bpb_over_batches(model, all_bos, token_bytes) == float("inf"), "a batch of only special tokens scores no bytes -> inf")

# It is a SUM over batches, not a mean of per-batch scores: doubling the batch list by
# repeating it must leave the score unchanged.
check(abs(bpb_over_batches(model, batches[:2] * 2, token_bytes) - bpb_over_batches(model, batches[:2], token_bytes)) < 1e-9,
      "repeating the same batches does not change the score (sums, not means)")

done("13")
