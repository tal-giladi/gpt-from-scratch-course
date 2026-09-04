# Lesson 02 - reference solution.


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
