# 17 - FlashAttention, and reading a port for what it gave up

Upstream, one line made this whole repo GPU-only:

    y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

`fa3` is **FlashAttention-3**, downloaded from the Hugging Face `kernels` hub as a compiled GPU
program tuned for NVIDIA's Hopper chips (the H100). There is no CPU version, and there cannot
really be one: the whole design is built around how a GPU's memory is laid out. This lesson is
about what that kernel does, what the CPU port replaced it with, and how to read a port for what
it gave up.

## The problem: the score matrix is huge

Lesson 05's attention, written the straightforward way:

    scores  = q @ k.T / sqrt(d)     # (B, H, T, T)   <- one score per (query, key) pair
    weights = softmax(scores)       # (B, H, T, T)   <- and a second tensor the same size
    out     = weights @ v

That `T x T` score matrix has to exist in memory, in full. Its size grows with the **square** of
the context length:

    B = 8, H = 8, T = 256,   float32      8 * 8 * 256 * 256 * 4 bytes   =  16 MB
    B = 8, H = 8, T = 2048,  bfloat16     8 * 8 * 2048 * 2048 * 2 bytes = 512 MB
    B = 8, H = 8, T = 4096,  bfloat16                                    ~  2 GB

Double the context, four times the memory. And it is not just stored - it is written, read back for
the softmax, written again, and needed again in the backward pass. At long context, attention is
limited by how fast memory can be moved, long before it is limited by arithmetic.

## What FlashAttention does instead

FlashAttention computes **exactly the same numbers** without ever building the full matrix.

A GPU has two kinds of memory: a large, relatively slow main memory (the "80 GB" on the box), and
a tiny amount of extremely fast memory right on the chip (SRAM, a few megabytes per core). Moving
data between them is the expensive part.

So FlashAttention works in **tiles**:

1. Take a block of queries - say, positions 0-127 - and a block of keys, small enough to fit in
   the fast on-chip memory.
2. Compute that block's scores there.
3. Fold them into a **running softmax** and a running weighted sum of values, then throw the
   block's scores away.
4. Move to the next block.

The trick in step 3 is that softmax can be computed in pieces. For one query, the probabilities
need `exp(score_j) / sum of all exp(score)` - and you do not need all the scores at once to get
that. Keep a running total as you go, and when a new block raises the total, rescale what you have
accumulated so far. The final answer is exact, not an approximation.

The result:

- **Memory**: only a tile of scores exists at any moment - memory grows with `T`, not `T x T`.
- **Speed**: typically 2-4x faster, not because it does less arithmetic, but because it moves far
  less data between slow and fast memory.
- **Windows for free**: because the kernel is told `causal=True, window_size=(w, 0)`, it knows
  which tiles are entirely outside the allowed band (lesson 05) and **never computes them at all**.
  A short-window layer genuinely costs less.

Version 3 adds Hopper-specific optimisations on top, but the tiling is the core idea.

## What the port had to do instead

    if window is None or window <= 0 or window >= T:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=_causal_window_mask(T, window, q.device))

`F.scaled_dot_product_attention` (SDPA) is PyTorch's built-in attention. It picks the best kernel
it has for the situation. There are two branches because PyTorch's API has no "sliding window"
option:

- **Full context** (`window >= T`, the `L` layers): pass `is_causal=True`. PyTorch knows the mask
  shape without being given one.
- **Short window** (the `S` layers): the only way to say "causal *and* at most `window` back" is to
  build the `(T, T)` boolean mask from lesson 05 and pass it in. The port caches it per
  `(T, window)`, so it is built once.

How much does that cost? Measured in the lab container (PyTorch 2.9.1, CPU), for one attention call
at `B = 1`, `H = 8`, `T = 4096` - where a full score matrix would be 512 MB:

    call                                   extra memory at peak
    SDPA, is_causal=True                          16 MB
    SDPA, attn_mask (window 512)                 179 MB
    lesson 05's manual version                 1,072 MB

And a profile of both SDPA calls shows PyTorch using the same CPU kernel for both,
`_scaled_dot_product_flash_attention_for_cpu` - a CPU flash-style implementation that, like the GPU
one, does not build the full score matrix.

So the honest picture on this PyTorch version is:

- **Neither SDPA path builds the full 512 MB of scores.** The manual version builds it twice - the
  scores and the masked copy.
- **The explicit mask still costs memory.** The boolean mask itself is `4096 x 4096` bytes = 16 MB,
  and the kernel works from a floating-point version of it, so the masked call used about ten times
  the memory of the causal one.
- **The window saves no compute.** Unlike FlashAttention-3, the CPU kernel does not skip the tiles
  outside the window; it computes every pair and applies the mask. At `T = 256`, a window-128 call
  (25% fewer allowed pairs) measured *slightly slower* than the full causal call - 17 ms vs 14 ms
  forward+backward.

(Older PyTorch versions went further and fell back to a plain math kernel whenever a mask was
passed, materialising the full score matrix. The check's `mask_memory_bytes` computes that
worst case.)

Put together: on a GPU with FlashAttention-3, the sliding windows are a **pure saving**. On this CPU
port they are **a small cost** - more memory, no less compute. At `T = 256` nobody notices. At long
context it would matter. **An architecture choice that is "faster" can become slower when the kernel
underneath it changes** - and that is the real lesson of reading a port.

## Reading a port well

The fork marks every change to upstream with a comment like

    # CHANGED (talg, 2026-09-04): <what changed>. Reason: <why>.

That convention is worth copying. When you pick up an unfamiliar fork, the useful question is not
"what does this code do?" but **"what did this fork give up, and would I have made the same trade?"**
Every change falls into one of two kinds:

- **forced** - there was no alternative on this hardware;
- **a trade** - a choice someone made, which a different person with different constraints might
  reverse.

For this fork:

| Given up | What it costs | Forced or a trade? |
|---|---|---|
| FlashAttention-3 | no tile-skipping; short windows cost a little instead of saving | **forced** - no CPU build exists |
| 524K-token batches | ~256x fewer tokens per step, much noisier gradients (lesson 12) | **forced** - a CPU cannot process that many tokens in the time budget |
| upstream's eval size | `val_bpb` no longer comparable with upstream (lesson 13) | **forced** by the time budget, and declared loudly |
| bfloat16 | nothing on this CPU - it was no faster, and sometimes much slower (lesson 16) | a trade, and the right one here |
| `torch.compile` | the 2-3x fusion win on memory-bound operations (lesson 16) | a trade - adding a C++ compiler to the image would reverse it |

Three of the five were forced; two are trades. A reader who cannot tell which is which cannot
maintain the fork - they will either "fix" something that cannot be fixed, or leave a speedup on the
table because they assumed it was impossible.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import train_defs
       defs = train_defs()
       defs.USE_FLASH                    # False on CPU
       m = defs._causal_window_mask(8, 2, torch.device("cpu")).int(); m
       m.sum()                           # 21 (query, key) pairs survive out of 64
       defs._mask_cache.keys()           # cached per (T, window, device)

2. See which kernel PyTorch picks, on your version:

       import torch.nn.functional as F
       from torch.profiler import profile
       q, k, v = (torch.randn(1, 4, 256, 64) for _ in range(3))
       mask = defs._causal_window_mask(256, 128, q.device)
       with profile() as p: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
       [e.key for e in p.key_averages() if "attention" in e.key]

3. Fill in `lab/exercises/lesson_17.py`: `window_mask(T, window)` and
   `mask_memory_bytes(B, n_head, T)`.

4. Grade it:

       bash lab/lab.sh check 17

## Hints

- The mask is `(T, T)` boolean, `True` where the key is visible:
  `delta = i[:, None] - i[None, :]`, then `(delta >= 0) & (delta <= window)` - exactly lesson 05.
- Count the `True` entries by hand before you run it. With no window, row `i` can see `i + 1`
  positions, so the total is `1 + 2 + ... + T`. A window `w` caps every row at `w + 1`; only the
  first `w` rows are shorter than that. That gives `T*(w+1) - w*(w+1)/2`. For `T = 8, w = 2`:
  `8*3 - 2*3/2 = 21`. The check compares your mask's `.sum()` against that formula.
- `mask_memory_bytes` is the size of the full score tensor the naive path would build:
  `B * n_head * T * T` elements at 4 bytes each for float32. Return an int.
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

The attention score matrix grows with the square of the context, and moving it around is what makes
naive attention slow. FlashAttention computes the same softmax in tiles that fit in fast on-chip
memory, never builds the full matrix, and skips the tiles a window rules out - which is why it is a
GPU kernel and not a formula you can port. The CPU port's SDPA calls also avoid the full matrix on
current PyTorch, but the explicit window mask costs extra memory and saves no compute, turning the
sliding window from a saving into a small cost. Reading a port means sorting its changes into forced
and traded. Next module: the loop that lets an agent make changes like these on its own.
