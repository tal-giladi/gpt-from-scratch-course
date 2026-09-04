from _lib import check, done, stub_guard

from exercises.lesson_02 import pack_row

pack_row = stub_guard(pack_row, "pack_row")

# 1. Best fit, in order: 5 fits first, then 3, then 2. Nothing is cropped.
docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]]
row, left = pack_row(docs, 10)
check(row == [6, 7, 8, 9, 10, 1, 2, 3, 4, 5], "longest-that-fits order: 5, then 3, then 2")
check(left == [[11]], "the document that never fitted is returned as leftover")

# 2. The caller's list is untouched.
check(docs == [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10], [11]], "the input list is not mutated")

# 3. Nothing fits in the tail -> crop the SHORTEST document.
row, left = pack_row([[1, 2, 3, 4, 5], [9, 9, 9]], 7)
check(row == [1, 2, 3, 4, 5, 9, 9], "when nothing fits, the shortest document is cropped to fill the row")
check(left == [], "a cropped document is consumed, not returned")

# 4. Cropping picks the shortest, not the first.
row, left = pack_row([[7, 7, 7, 7], [8, 8]], 3)
check(row == [8, 8, 7], "the shortest document is chosen for cropping (8,8 whole, then one 7)")

# 5. Ties go to the earliest document.
row, left = pack_row([[1, 2], [3, 4]], 2)
check(row == [1, 2] and left == [[3, 4]], "equal-length candidates: the earliest one wins")

# 6. The row is always exactly `capacity` long.
for capacity in (1, 4, 9, 10):
    row, _ = pack_row([[1, 2, 3], [4, 5, 6, 7], [8], [9, 10]], capacity)
    check(len(row) == capacity, f"row length is exactly capacity ({capacity})")

done("02")
