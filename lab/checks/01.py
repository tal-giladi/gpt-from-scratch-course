from _lib import check, done, fail, stub_guard

from exercises.lesson_01 import token_stats
from lib.common import tokenizer

tok = tokenizer()
token_stats = stub_guard(token_stats, "token_stats")

ASCII = "The quick brown fox jumps over the lazy dog."
UNICODE = "Tal — שלום, מה נשמע? 42°"

r = token_stats(ASCII)
if not isinstance(r, dict):
    fail("token_stats must return a dict")

expected_ids = tok.encode(ASCII)
check(r.get("n_tokens") == len(expected_ids), f"n_tokens == {len(expected_ids)} for the ASCII sentence")
check(r.get("n_bytes") == len(ASCII.encode("utf-8")), "n_bytes counts utf-8 bytes")
check(r.get("roundtrip") is True, "roundtrip is True for plain ASCII")
ratio = r.get("bytes_per_token")
check(
    isinstance(ratio, float) and abs(ratio - r["n_bytes"] / r["n_tokens"]) < 1e-9,
    "bytes_per_token == n_bytes / n_tokens",
)
check(2.0 < ratio < 6.0, f"bytes_per_token is in the plausible range for English ({ratio:.2f})")

u = token_stats(UNICODE)
check(
    u.get("n_bytes") == len(UNICODE.encode("utf-8")) and u["n_bytes"] > len(UNICODE),
    "n_bytes uses utf-8 bytes, not len(str), on non-ASCII text",
)
check(u.get("roundtrip") is True, "non-ASCII text still round-trips through the tokenizer")

e = token_stats("")
check(e.get("n_tokens") == 0 and e.get("bytes_per_token") == 0.0, "the empty string gives 0 tokens and 0.0 ratio")

done("01")
