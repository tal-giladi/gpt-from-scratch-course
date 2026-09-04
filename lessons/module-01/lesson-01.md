# 01 - Bytes, tokens, and a vocabulary of 8192

A language model does not see text. It sees integers. Everything in this course - the
embeddings, the attention, the loss - operates on a sequence of integers, and the first
thing worth understanding properly is where those integers come from, because a
surprising amount of a model's behaviour is decided here, before the model exists.

The obvious two options are both bad. **One integer per character** gives you a tiny
vocabulary (say 256 for bytes) but very long sequences - and since attention cost grows
with the square of sequence length, that is expensive, and the model spends its capacity
learning to spell. **One integer per word** gives you short sequences but an unbounded
vocabulary: every typo, every product name, every language is a new word, and anything the
tokenizer has not seen becomes `<unk>` and is gone forever.

**BPE** (byte-pair encoding) is the compromise everyone landed on. Start with raw bytes -
so nothing is ever unrepresentable - then repeatedly find the most frequent adjacent pair
of symbols in a large text sample and merge it into one new symbol. Do that 8,000-odd
times and you get a vocabulary where common words are a single token, rare words are two
or three, and arbitrary binary junk still round-trips as individual bytes.

## What `prepare.py` actually did

When you ran `prepare.py`, step two was `train_tokenizer()`. It streamed ~100M characters
of the downloaded text through `rustbpe`, learned the merges, and saved the result as a
`tiktoken` encoding under the shared cache:

    VOCAB_SIZE = 8192
    SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
    BOS_TOKEN = "<|reserved_0|>"

Three facts to carry forward:

1. **8192 tokens is small.** GPT-4 uses ~100k, Llama 3 ~128k. Small vocab means longer
   sequences for the same text, but a much cheaper output layer - and the output layer
   (`lm_head`: `n_embd x vocab_size`) is one of the biggest matrices in a small model. At
   this course's scale it is about half the total compute. Vocabulary size is not a
   detail; it is a compute decision.
2. **Four special tokens** are reserved, and `<|reserved_0|>` is used as **BOS**
   (beginning of sequence). It is not text; it is a marker the model learns to treat as
   "a document starts here".
3. **The split pattern matters.** `prepare.py` uses a GPT-4-style regex with one change -
   `\p{N}{1,2}` instead of `{1,3}` - so numbers are chunked at most two digits at a time.
   That single character in a regex changes how the model does arithmetic.

## Bytes per token is the unit that matters

Later, the course's only quality metric will be **bits per byte**, not "loss per token".
The reason starts here: change the tokenizer and every per-token number changes with it. A
model with a bigger vocabulary sees fewer, fatter tokens and gets a lower loss per token
without being any better at predicting text. Per-*byte* numbers are comparable across
tokenizers; per-token numbers are not.

So the first thing to measure is the exchange rate: how many bytes of text does one token
buy you? For English prose with an 8192-token vocab, expect somewhere around 3-4.

## Do this

1. Make sure the lab works (once):

       bash lab/lab.sh up

2. Look around in a Python shell inside the lab container:

       bash lab/lab.sh shell

       from lib.common import tokenizer
       tok = tokenizer()
       tok.get_vocab_size()                      # 8192
       ids = tok.encode("The quick brown fox jumps over the lazy dog.")
       ids                                        # a list of ints
       len(ids), len("The quick brown fox jumps over the lazy dog.".encode("utf-8"))
       tok.decode(ids)                            # back to the original string
       [tok.decode([i]) for i in ids]             # what each token *is*

   That last line is the one to stare at. Note where the spaces go - most tokens begin
   with the space that preceded the word, which is why `"dog"` and `" dog"` are different
   tokens.

3. Fill in `lab/exercises/lesson_01.py`: `token_stats(text)` returns the token count, the
   UTF-8 byte count, the bytes-per-token ratio, and whether the text survives a
   round-trip through encode/decode.

4. Grade it:

       bash lab/lab.sh check 01

## Hints

- `tokenizer()` from `lib.common` is cached - calling it repeatedly is free.
- `tok.encode(text)` takes a `str` and returns a `list[int]`. Do not pass `prepend=`; this
  exercise is about the raw text.
- Byte count means **UTF-8 bytes**: `len(text.encode("utf-8"))`, not `len(text)`. For
  ASCII they are the same, and the check uses a non-ASCII string precisely to catch that.
- `bytes_per_token` is bytes divided by tokens - guard against dividing by zero for the
  empty string, where the honest answer is `0.0`.

## Solution

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

## Summary

Text becomes integers through a BPE vocabulary that was learned - not designed - from the
same corpus the model trains on. The vocabulary size is a compute decision that shows up
in the size of the output layer, and it makes per-token metrics incomparable across
models, which is why this course will grade everything per byte instead. Next: how those
token ids get packed into the rectangular batches a GPU (or in our case, a CPU) can chew
on.
