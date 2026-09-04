# Lesson 01 - Bytes, tokens, and a vocabulary of 8192
# Read lessons/module-01/lesson-01.md before filling this in.

from lib.common import tokenizer


def token_stats(text: str) -> dict:
    """Return {"n_tokens": int, "n_bytes": int, "bytes_per_token": float,
    "roundtrip": bool} for one string.

    n_bytes is the length of the text in UTF-8 BYTES, not characters.
    bytes_per_token is 0.0 for the empty string.
    roundtrip is True when decode(encode(text)) == text.
    """
    # TODO: encode the text, count tokens and utf-8 bytes, compute the ratio,
    # and check that decoding the ids gives the original string back.
    raise NotImplementedError
