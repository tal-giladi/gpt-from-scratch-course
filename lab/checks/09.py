import torch
from _lib import check, check_close, done, stub_guard

from exercises.lesson_09 import greedy_generate, next_token_probs
from lib.common import build_model, get_batch

next_token_probs = stub_guard(next_token_probs, "next_token_probs")
greedy_generate = stub_guard(greedy_generate, "greedy_generate")

model = build_model()
V = model.config.vocab_size
idx, _ = get_batch(B=2, T=32)

# Give the untrained model some opinions, otherwise every distribution is flat and none of
# the tests below can tell a right answer from a wrong one.
with torch.no_grad():
    model.lm_head.weight.normal_(0, 0.05)

p = next_token_probs(model, idx)
check(tuple(p.shape) == (2, V), f"next_token_probs returns (B, vocab_size) == (2, {V})")
check_close(p.sum(dim=-1), torch.ones(2), "each row is a probability distribution (sums to 1)", tol=1e-4)
check((p >= 0).all().item(), "no negative probabilities")

# Temperature
cold = next_token_probs(model, idx, temperature=0.25)
hot = next_token_probs(model, idx, temperature=4.0)
check(cold.max(dim=-1).values.mean() > p.max(dim=-1).values.mean(), "temperature 0.25 sharpens the distribution")
check(hot.max(dim=-1).values.mean() < p.max(dim=-1).values.mean(), "temperature 4.0 flattens it")
check(
    torch.equal(cold.argmax(dim=-1), p.argmax(dim=-1)),
    "temperature never changes which token is most likely",
)

# Top-k
k = 5
tk = next_token_probs(model, idx, top_k=k)
check(int((tk[0] > 0).sum()) == k, f"top_k={k} leaves exactly {k} tokens with non-zero probability")
check_close(tk.sum(dim=-1), torch.ones(2), "the surviving probabilities still sum to 1", tol=1e-4)
check(
    torch.equal(tk.argmax(dim=-1), p.argmax(dim=-1)),
    "top-k does not change the most likely token either",
)
check(
    torch.allclose(next_token_probs(model, idx, top_k=V), p, atol=1e-6),
    "top_k == vocab_size is the same as no filtering",
)

# Greedy generation
start = [int(v) for v in idx[0, :8]]
out = greedy_generate(model, start, 6)
check(len(out) == len(start) + 6, "greedy_generate returns the prompt plus n_new tokens")
check(out[: len(start)] == start, "the prompt is preserved unchanged at the front")
check(out == greedy_generate(model, start, 6), "greedy decoding is deterministic")
check(list(start) == [int(v) for v in idx[0, :8]], "the caller's list is not mutated")

first_new = greedy_generate(model, start, 1)[-1]
expected = int(next_token_probs(model, torch.tensor([start])) [0].argmax())
check(first_new == expected, "the first generated token is the argmax of the distribution")

done("09")
