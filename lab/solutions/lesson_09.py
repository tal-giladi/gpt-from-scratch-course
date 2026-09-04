# Lesson 09 - reference solution.

import torch


def next_token_probs(model, idx, temperature=1.0, top_k=None):
    with torch.no_grad():
        logits = model(idx)[:, -1, :].float()
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.softmax(logits, dim=-1)


def greedy_generate(model, ids, n_new):
    ids = list(ids)
    seq_len = model.config.sequence_len
    for _ in range(n_new):
        idx = torch.tensor([ids[-seq_len:]], dtype=torch.long)
        probs = next_token_probs(model, idx)
        ids.append(int(probs[0].argmax()))
    return ids
