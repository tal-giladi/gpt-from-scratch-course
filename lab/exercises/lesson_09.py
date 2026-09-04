# Lesson 09 - From logits to text
# Read lessons/module-03/lesson-03.md before filling this in.

import torch


def next_token_probs(model, idx, temperature=1.0, top_k=None):
    """Probability distribution over the NEXT token, shape (B, vocab_size).

    idx is (B, T) of token ids. Take the logits at the last position, divide
    by the temperature, optionally keep only the top_k highest scores (set the
    rest to -inf), then softmax.

    top_k=None means no filtering.
    """
    # TODO: last position, temperature, top-k mask, softmax - in that order.
    raise NotImplementedError


def greedy_generate(model, ids, n_new):
    """Append n_new tokens by always taking the most likely one.

    ids is a list[int]; return the full list (original + generated).
    Never feed the model more than model.config.sequence_len tokens.
    """
    # TODO: loop n_new times, argmax the distribution, append.
    raise NotImplementedError
