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
    # TODO: implement the loop described above.
    raise NotImplementedError
