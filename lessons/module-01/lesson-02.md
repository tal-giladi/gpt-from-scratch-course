# 02 - Packing documents into rectangles

The model wants a rectangle: a tensor of shape `(B, T)`, `B` rows of exactly `T` token ids.
The data is not a rectangle. It is a stream of documents of wildly different lengths - a
tweet, a Wikipedia article, a page of source code. Getting from one to the other is a
design decision with real consequences, and `prepare.py` makes an unusual choice.

The usual choice is **padding**: put one document per row, pad the short ones with a `<pad>`
token, mask the padding out of the loss. It is simple and it wastes compute in proportion
to how uneven your documents are - on web text, easily 30-50% of every batch is padding
that costs full price in FLOPs and teaches the model nothing.

`prepare.py` instead does **best-fit packing with cropping**, and achieves 100%
utilisation: every position in every row is a real token that contributes to the loss.

## The algorithm, exactly

Each row has capacity `T + 1` (one extra, and lesson 03 explains why). A buffer of
1,000 tokenized documents is kept topped up. Then, per row:

    pos = 0
    while pos < capacity:
        remaining = capacity - pos
        pick the LONGEST document in the buffer whose length <= remaining
        if one exists:
            remove it from the buffer, write it at pos, pos += len(doc)
        else:
            remove the SHORTEST document, write its first `remaining` tokens, pos = capacity

Two details that are easy to skim past and are both deliberate:

- **Longest-that-fits, not first-that-fits.** Greedily taking the biggest document that
  still fits leaves the smallest possible hole, which is what makes the tail of the row
  fillable without much cropping. This is the classic best-fit bin-packing heuristic.
- **When nothing fits, crop the shortest.** Cropping throws tokens away, so you want to
  throw away as little as possible - the shortest document in the buffer is the one whose
  tail you lose least of. (Its remainder is discarded, not carried over. Simplicity beats
  a few hundredths of a percent of data here.)

## Every row starts with BOS

Before packing, every document is encoded with `prepend=bos_token`. So each document in the
buffer already begins with `<|reserved_0|>`, and therefore every row begins with BOS too -
the first document written to a row brings it along.

This matters more than it looks. The model has no idea where documents begin unless you
tell it, and "the previous document ended, forget everything" is exactly the kind of thing
a next-token predictor needs to represent. BOS is that signal. Without it the model spends
capacity learning that text sometimes jumps topic mid-sentence for no reason.

Note what it does *not* do: it does not stop attention from crossing the boundary. Tokens
after a BOS can still attend to the previous document's tokens. Preventing that needs a
document mask, which costs kernel complexity; upstream decided the BOS marker is enough.
That is a real, arguable engineering choice sitting in your dataloader - the kind of thing
an agent doing autonomous research on this file might try to change.

## Do this

1. In the lab shell, watch a real batch get built:

       bash lab/lab.sh shell

       from lib.common import get_batch, tokenizer
       tok = tokenizer()
       x, y = get_batch(B=2, T=128)
       x.shape, x.dtype                      # torch.Size([2, 128]) torch.int64
       x[0, :12]                             # first row, first 12 ids
       tok.decode(x[0].tolist())             # read it as text
       bos = tok.get_bos_token_id(); bos     # the id every row starts with
       (x[:, 0] == bos).all()                # True: both rows begin with BOS
       (x[0] == bos).sum()                   # how many documents got packed into row 0

2. Fill in `lab/exercises/lesson_02.py`: `pack_row(docs, capacity)` implements exactly the
   algorithm above and returns `(row, leftover_docs)`.

3. Grade it:

       bash lab/lab.sh check 02

## Hints

- Do not mutate the caller's list. Copy it first (`docs = [list(d) for d in docs]` or at
  least `list(docs)`), then pop from your copy and return it as the leftover.
- "Longest that fits" with ties broken by **first occurrence**: iterate with an index, keep
  the best index and best length, and only replace when the new length is **strictly**
  greater. That matches `prepare.py`, and the check tests exactly this case.
- The crop branch takes `doc[:remaining]` from the **shortest** document (by `len`), and
  the remainder is thrown away, not pushed back.
- The returned row must have length exactly `capacity` - assert it yourself while
  developing.
- Assume the buffer never runs out for this exercise; the real dataloader refills it, you
  do not have to.

## Solution

    def pack_row(docs, capacity):
        remaining_docs = [list(d) for d in docs]
        row = []
        while len(row) < capacity:
            remaining = capacity - len(row)
            best_idx, best_len = -1, 0
            for i, doc in enumerate(remaining_docs):
                if best_len < len(doc) <= remaining:
                    best_idx, best_len = i, len(doc)
            if best_idx >= 0:
                row.extend(remaining_docs.pop(best_idx))
            else:
                shortest = min(range(len(remaining_docs)), key=lambda i: len(remaining_docs[i]))
                row.extend(remaining_docs.pop(shortest)[:remaining])
        return row, remaining_docs

## Summary

Documents become rectangles by best-fit packing, not padding: every position is a real
token, every row starts with BOS, and the only loss is the tail of a cropped short
document. You now know what is in `x`. Next: what the model is asked to *do* with it, and
why the row capacity was `T + 1`.
