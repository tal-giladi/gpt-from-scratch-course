# Lesson 01 - reference solution.

from lib.common import tokenizer


def token_stats(text: str) -> dict:
    tok = tokenizer()
    ids = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(ids)
    return {
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "bytes_per_token": (n_bytes / n_tokens) if n_tokens else 0.0,
        "roundtrip": tok.decode(ids) == text,
    }
