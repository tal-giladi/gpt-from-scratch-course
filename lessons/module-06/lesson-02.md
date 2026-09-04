# 17 - FlashAttention, and reading a port for what it gave up

The single line that made this repo GPU-only:

    y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

`fa3` is FlashAttention-3, loaded from the Hugging Face `kernels` hub as a compiled CUDA
kernel tuned for Hopper. There is no CPU build. There cannot be a CPU build - the whole
design is about a GPU's memory hierarchy.

## What FlashAttention actually does

Lesson 05's four lines have a problem that only shows up at scale:

    scores  = q @ k.T / sqrt(d)     # (B, H, T, T)  <- this tensor
    weights = softmax(scores)       # (B, H, T, T)  <- and this one
    out     = weights @ v

The `T x T` score matrix is **materialised in memory**. At `B=8, H=8, T=2048` in bf16 that
is 512 MB, written out and read back twice, and the backward pass needs it again. Attention
becomes bound by memory bandwidth long before it is bound by arithmetic.

FlashAttention never builds it. It tiles the computation: load a block of queries and a
block of keys into fast on-chip SRAM, compute that block's scores, and fold them into a
running **online softmax** (a streaming formulation that keeps a running max and running
sum, so the result is exact rather than approximate). Move to the next block. The `T x T`
matrix exists only a tile at a time.

The result is the same numbers, computed with `O(T)` extra memory instead of `O(T^2)`, and
2-4x faster - not because it does less arithmetic, but because it does far less memory
traffic. Version 3 adds Hopper-specific asynchrony and FP8 support, and the `causal` and
`window_size` arguments mean the tiles that fall entirely outside the mask are simply never
computed at all.

## What the port had to do instead

    if window is None or window <= 0 or window >= T:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=_causal_window_mask(T, window, q.device))

Two branches, because the sliding window has no first-class support in the PyTorch API:

- **`is_causal=True`** lets SDPA use its own efficient path - the mask is implied, not
  passed, so it can skip the upper triangle without ever building a boolean matrix.
- **An explicit `attn_mask`** is the only way to express "causal *and* within `window`". It
  is a real `(T, T)` boolean tensor, and it forces SDPA onto its general math kernel: the
  `T x T` scores get materialised after all. The port caches the mask per `(T, window)` so
  at least it is built once.

So the port is honest but not free: the sliding windows, which on a GPU are a pure saving,
become a cost on CPU. At `T=256` the mask is 64 KB and nobody cares. At `T=2048` it would be
4 MB per shape plus a materialised score tensor - which is precisely the thing
FlashAttention exists to avoid. **A "faster" architecture choice can invert when the kernel
under it changes**, and that is the real lesson of reading a port.

## Reading a port well

The fork marks every deviation with `# CHANGED (talg, 2026-09-04):` and a `# Reason:`. That
convention is worth stealing. When you pick up an unfamiliar fork, the question is never
"what does this code do" - it is **"what did this fork give up, and would I have made the
same trade?"** For this one:

| Given up | Cost | Would you? |
|---|---|---|
| FlashAttention-3 | slower, and windows now cost memory | forced - no CPU build exists |
| bf16 | none on this CPU; it was slower | yes (lesson 16) |
| `torch.compile` | 2-3x on the memory-bound ops | arguable - a toolchain in the image would change it |
| 512K-token batches | much noisier gradients | forced by wall-clock |
| upstream's eval budget | numbers no longer comparable upstream | forced, and declared loudly |

Only two of those five were genuinely forced. The others are trades, and a reader who
cannot tell the difference cannot maintain the fork.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import train_defs
       defs = train_defs()
       defs.USE_FLASH                    # False on CPU
       m = defs._causal_window_mask(8, 2, torch.device("cpu")).int(); m
       m.sum()                            # how many (query, key) pairs survive
       defs._mask_cache.keys()            # it is cached per (T, window, device)

2. Fill in `lab/exercises/lesson_17.py`: `window_mask(T, window)` and
   `mask_memory_bytes(B, n_head, T)`.

3. Grade it:

       bash lab/lab.sh check 17

## Hints

- The mask is `(T, T)` boolean, `True` where the key is visible:
  `delta = i[:, None] - i[None, :]`, then `(delta >= 0) & (delta <= window)`.
- Count the `True` entries by hand before you run it: for a window `w < T` the answer is
  `T*(w+1) - w*(w+1)/2`, because the first `w` rows are shorter than the rest. The check
  compares your mask's `.sum()` against that formula.
- `mask_memory_bytes` is the score tensor SDPA has to materialise when a mask forces the
  math path: `B * n_head * T * T` elements at 4 bytes each for float32. Return an int.
- Do not import `_causal_window_mask` - write it.

## Solution

    import torch

    def window_mask(T, window):
        i = torch.arange(T)
        delta = i[:, None] - i[None, :]
        return (delta >= 0) & (delta <= window)

    def mask_memory_bytes(B, n_head, T, bytes_per_element=4):
        return int(B * n_head * T * T * bytes_per_element)

## Summary

FlashAttention is not a different algorithm - it is the same softmax computed in tiles so
the `T x T` score matrix never reaches memory, which is why it is a CUDA kernel and not a
formula you can port. Replacing it with a masked SDPA call is correct and measurably
equivalent, and it quietly turns the sliding window from a saving into a cost. Next module:
the loop that lets an agent make changes like these on its own.
