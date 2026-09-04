import torch
from _lib import check, done, stub_guard

from exercises.lesson_17 import mask_memory_bytes, window_mask
from lib.common import train_defs

defs = train_defs()
window_mask = stub_guard(window_mask, "window_mask")
mask_memory_bytes = stub_guard(mask_memory_bytes, "mask_memory_bytes")

device = torch.device("cpu")

for T, w in ((8, 2), (16, 5), (32, 31), (7, 0)):
    got = window_mask(T, w)
    check(tuple(got.shape) == (T, T), f"T={T}, window={w}: the mask is (T, T)")
    check(got.dtype == torch.bool, f"T={T}, window={w}: the mask is boolean")
    check(torch.equal(got, defs._causal_window_mask(T, w, device)), f"T={T}, window={w}: matches the port's mask")
    expected_true = T * (w + 1) - w * (w + 1) // 2
    check(int(got.sum()) == expected_true, f"T={T}, window={w}: exactly {expected_true} visible (query, key) pairs")

# The two properties that make it a causal window.
m = window_mask(10, 3)
check(bool(m.diagonal().all()), "every position can always see itself")
check(not bool(m.triu(diagonal=1).any()), "nothing can see the future")
check(not bool(m[9, 5]), "a key 4 positions back is outside a window of 3")
check(bool(m[9, 6]), "...but one 3 positions back is inside it")

# --- the memory the mask costs ---------------------------------------------------------
check(mask_memory_bytes(1, 1, 10) == 400, "1x1x10x10 float32 scores is 400 bytes")
check(mask_memory_bytes(8, 8, 256) == 8 * 8 * 256 * 256 * 4, "the course-scale batch materialises 16 MB of scores")
big = mask_memory_bytes(8, 8, 2048, bytes_per_element=2)
check(big == 8 * 8 * 2048 * 2048 * 2, f"upstream's shape in bf16 would be {big / 1e6:.0f} MB - what FlashAttention avoids")
check(
    mask_memory_bytes(8, 8, 512) == 4 * mask_memory_bytes(8, 8, 256),
    "doubling the context quadruples the memory: this is the O(T^2) everyone worries about",
)

done("17")
