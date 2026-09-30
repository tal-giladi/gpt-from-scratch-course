# Lesson 02 - Packing documents into rectangles
# Read lessons/module-01/lesson-02.md before filling this in.


def pack_row(docs, capacity):
    """Pack documents into one row of exactly `capacity` token ids.

    docs: list of documents, each a list[int] (BOS already prepended).
    Returns (row, leftover_docs):
        row           list[int] of length exactly `capacity`
        leftover_docs the documents that were not consumed, in their original
                      order, as a NEW list (do not mutate the caller's).

    Rule, repeated until the row is full:
      - remaining = capacity - len(row)
      - if any document has len <= remaining, take the LONGEST such document
        (earliest one on a tie) and append all of it
      - otherwise take the SHORTEST document and append only its first
        `remaining` tokens; the rest is discarded
    """
    row = []
    # A copy in the CALLER's order: both tie-breaks and the leftovers are
    # defined by that order, so it must not be sorted away.
    docs = [list(d) for d in docs]
    while len(row) < capacity:
        remaining = capacity - len(row)
        i = longest_fitting(docs, remaining)
        if i is not None:
            row.extend(docs.pop(i))
        else:
            i = shortest(docs)
            row.extend(docs.pop(i)[:remaining])
    return (row, docs)


def longest_fitting(docs, remaining):
    """Index of the longest document that fits in `remaining`, or None."""
    best = None
    for i, doc in enumerate(docs):
        if len(doc) <= remaining and (best is None or len(doc) > len(docs[best])):
            best = i
    return best


def shortest(docs):
    """Index of the shortest document; `min` keeps the earliest on a tie."""
    return min(range(len(docs)), key=lambda i: len(docs[i]))
