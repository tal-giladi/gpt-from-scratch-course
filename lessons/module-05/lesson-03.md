# 15 - What a fixed budget actually buys

The rule the whole repo is built on: **five minutes, one GPU, lowest `val_bpb` wins.** That
one constraint decides more about the model than any architecture choice, and this lesson is
about seeing why.

A fixed time budget is a fixed number of FLOPs. Spend them how you like:

    tokens_seen = (achieved FLOP/s * seconds) / flops_per_token

A bigger model costs more FLOPs per token, so it sees fewer tokens in the same budget. A
smaller model sees more tokens but has less capacity to store what it saw. Somewhere in
between is the model that reaches the lowest loss for that budget - and finding it is the
question "scaling laws" answers.

## Chinchilla, in one line

The 2022 Chinchilla result, stripped of its detail: for a compute-optimal model, train on
about **20 tokens per parameter**. Fewer and you have bought capacity you cannot fill; many
more and you would have been better off with a bigger model.

That number is a rule of thumb from one family of experiments, not a law of nature - later
work pushes far past 20 on purpose, because inference cost matters too and a smaller
over-trained model is cheaper to serve forever. But as a sanity check it is excellent, and
it is brutal about this course's setup.

## Do the arithmetic on your own run

The CPU fork's defaults: 11.5M parameters, ~300 tokens/second, 600-second budget.

    tokens   = 300 * 600            = 180,000
    ratio    = 180,000 / 11,500,000 = 0.016 tokens per parameter

Chinchilla says 20. You are at **0.016** - roughly a thousandth of compute-optimal. The
model is not undertrained by a bit; it has seen essentially none of the data it was built
for. Which is exactly why lesson 09's sampled text is word-shaped nonsense, and why nobody
should read anything into a `val_bpb` of 2.4 beyond "the loop works end to end".

Turn the arithmetic around and it tells you what you *should* run on a CPU:

    at 180,000 tokens, compute-optimal size = 180,000 / 20 = 9,000 parameters

Nine thousand. That is smaller than the embedding table for a single token vocabulary of
any useful size, which reveals the real constraint: with an 8192-token vocabulary, the two
`(vocab, n_embd)` tables alone are 2M parameters at `n_embd = 128` and you cannot go below
them. **The vocabulary sets a floor on model size, and at CPU budgets that floor is already
far above compute-optimal.** If you genuinely wanted the best possible model out of ten CPU
minutes, the first thing to change is not the architecture - it is `VOCAB_SIZE`.

That is a real, defensible research finding about this fork, derived in four lines of
arithmetic, and it is the kind of thing the capstone asks you to produce.

## Why upstream picked depth 8

Upstream's H100 run: ~50M parameters at depth 8, ~500K tokens per step, five minutes at
maybe 40% MFU on a 989 TFLOP/s machine. That is roughly `989e12 * 0.4 * 300 = 1.2e17` FLOPs,
which at `6 * 50e6 = 3e8` FLOPs per token buys about **400M tokens** - a ratio of 8 tokens
per parameter. Still under Chinchilla, deliberately: at a fixed *time* budget rather than a
fixed compute-optimal budget, and with a warmdown schedule that rewards finishing, slightly
oversized models tend to win the short race.

Depth 8 was not chosen from theory. It was found by running the loop.

## Do this

1. In the lab shell, check the arithmetic against your own run's summary:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       for depth in (2, 4, 8):
           m = build_model(toy_config(n_layer=depth))
           n = sum(p.numel() for p in m.parameters())
           f = m.estimate_flops()
           print(depth, f"{n/1e6:.1f}M params", f"{f/1e6:.1f} MFLOP/token")

   Note how the parameter count barely moves while the FLOPs roughly double - because the
   embeddings dominate the count but contribute nothing to the FLOPs.

2. Fill in `lab/exercises/lesson_15.py`: `budget_report(model, seconds, flops_per_second)`.

3. Grade it:

       bash lab/lab.sh check 15

## Hints

- `tokens = flops_per_second * seconds / flops_per_token`, floored to an int.
- `params` is every parameter tensor, embeddings included - Chinchilla's N counts them all.
- `tokens_per_param = tokens / params`.
- `chinchilla_params = tokens / 20` - the model size those tokens would have been
  compute-optimal for.
- Use `model.estimate_flops()` here; lesson 14 already made you write it by hand.
- Return plain Python numbers (`int` / `float`), not tensors.

## Solution

    def budget_report(model, seconds, flops_per_second):
        flops_per_token = model.estimate_flops()
        params = sum(p.numel() for p in model.parameters())
        tokens = int(flops_per_second * seconds / flops_per_token)
        return {
            "params": params,
            "flops_per_token": flops_per_token,
            "tokens": tokens,
            "tokens_per_param": tokens / params,
            "chinchilla_params": tokens / 20.0,
        }

## Summary

A time budget is a FLOPs budget, and spending it means trading model size against tokens
seen. The 20-tokens-per-parameter rule says this fork's CPU configuration is about a
thousand times short of compute-optimal, and that the binding constraint is the vocabulary
size setting a floor under the embedding tables. Next module: why the CPU is slow in the
first place, and what the port gave up to run at all.
