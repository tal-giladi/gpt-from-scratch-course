# 15 - What a fixed budget actually buys

The rule the whole repo is built on: **a fixed amount of training time, lowest `val_bpb` wins.**
Upstream that is five minutes on one GPU; in this fork, ten minutes on a CPU. That single rule
decides more about the best model than any clever architecture choice, and this lesson is about
seeing why - with nothing more than multiplication and division.

## Time is a FLOPs budget

Lesson 14 showed that a machine does some number of FLOPs per second, and lesson 08 that a model
costs some number of FLOPs per token. Put them together:

    tokens_seen = FLOPs_per_second * seconds / FLOPs_per_token

So a fixed time budget is a fixed number of FLOPs, and you choose how to spend it:

- **A bigger model** costs more FLOPs per token, so it gets through **fewer tokens**. It has more
  room to store patterns, but sees less text to learn them from.
- **A smaller model** gets through **more tokens**, but has less room to store what it saw.

A tiny illustration. A budget of 1,000,000 FLOPs, and two models:

    model A:  1,000 FLOPs per token   ->  1,000 tokens     big, barely trained
    model B:    100 FLOPs per token   -> 10,000 tokens     small, well trained

Neither extreme is best. Somewhere in between is the size that reaches the lowest loss for this
budget. Working out where is the question **scaling laws** answer.

## Chinchilla, in one line

A 2022 paper from DeepMind, known by its model's name, Chinchilla, trained hundreds of models of
different sizes on different amounts of data and asked: for a given amount of compute, what mix
gives the lowest loss? Stripped of detail, the answer was:

> Train on about **20 tokens per parameter**.

A 1-million-parameter model wants about 20 million tokens; a 1-billion-parameter model about 20
billion. Much fewer tokens than that and you paid for capacity you never filled; many more, and
the same compute would have gone further in a bigger model.

That is a rule of thumb from one family of experiments, not a law of nature. Many modern models
deliberately train far past 20 tokens per parameter, because a smaller model is cheaper to *run*
for every user, forever, and that matters more than the training bill. But as a sanity check,
20 is excellent - and it is brutal about this course's setup.

## Do the arithmetic on your own run

The CPU training run from lessons 13-14: an **11.5M-parameter** model, about **1,650 tokens per
second** (97 steps of 2,048 tokens in 120 seconds), and the fork's default **600-second** budget.

    tokens seen      = 1,650 * 600             = 990,000
    tokens per param = 990,000 / 11,534,600    = 0.086

Chinchilla says 20. This run is at **0.086** - about one two-hundred-and-thirtieth of that. The
model is not a little undertrained. It has seen almost none of the text a model its size is built
for. That is exactly why lesson 09's samples are word-shaped nonsense, and why nobody should read a
CPU `val_bpb` in the low 2s as anything more than "the whole pipeline works end to end".

Turn the arithmetic around, and it says which model you *should* have trained with 990,000 tokens:

    compute-optimal size = 990,000 / 20 = 49,500 parameters

**About fifty thousand parameters.** Now try to build a model that small. Every model here has an
embedding table and an output head, each of shape `(vocab_size, n_embd)`, and the vocabulary is
8,192 tokens. Even at a tiny width of `n_embd = 8` - eight numbers to describe everything about a
token:

    wte + lm_head = 8,192 * 8 + 8,192 * 8 = 131,072 parameters

That is already 2.6 times the compute-optimal size, before a single transformer block, and at a
width where the model could barely learn anything. At the real width of 256, those two tables are
4.2M.

(This fork used to be slower still. With PyTorch on its default thread count the same machine did
~255 tokens per second, which put the run at 0.013 tokens per parameter and the ideal model at
7,650 parameters - smaller than the two tables even at `n_embd = 1`. Lesson 16 is about that
fix.)

So the arithmetic has found something real: **the vocabulary sets a floor on model size, and at
CPU budgets that floor is far above compute-optimal.** If you genuinely wanted the best possible
model out of ten CPU minutes, the first thing to change is not the architecture - it is
`VOCAB_SIZE`. That is a defensible finding about this fork, derived in a few lines of arithmetic,
and it is exactly the kind of thing the capstone asks you to produce.

## Upstream's run, for comparison

Now the same arithmetic for upstream's shape: depth 8, `n_embd = 512`, context 2,048. That model is
**50.3M parameters** and costs **239 million FLOPs per token** (the FLOPs count leaves out the
embedding tables, lesson 08). Assume a 5-minute run on an H100 at about 40% MFU:

    FLOPs available  = 989.5e12 FLOPs/s * 0.40 * 300 s  = 1.19e17
    tokens seen      = 1.19e17 / 239,078,400            = ~500 million
    tokens per param = 500,000,000 / 50,332,176         = ~10

About 10 tokens per parameter - under Chinchilla's 20, but in the same neighbourhood, and more than
a hundred times closer than the CPU run. (The 40% is an assumption; a real run's MFU moves this
by a factor of two either way.) Landing a little under 20 is not unusual for a race against the
clock: with the learning rate decaying to zero at the finish (lesson 12), a slightly oversized model
often wins a short race. Upstream's depth was settled empirically - by running experiments - not
derived from the rule.

## Parameters and FLOPs are not the same thing

It is tempting to treat "model size" as one number. It is two, and they move differently. At the
course's width (`n_embd = 128`), changing only depth:

    depth    parameters    FLOPs per token
      2         3.54M          8.95M
      4         4.98M         11.50M
      8         7.86M         16.71M

Going from depth 2 to 8, parameters grow 2.2x but FLOPs per token only 1.9x. Two reasons, both from
lesson 08:

- **Value embeddings add parameters that cost no FLOPs.** Each extra layer that gets one adds a
  whole 1M-parameter lookup table, and a lookup does no multiplication.
- **`lm_head` costs the same FLOPs at every depth.** It is a fixed 6.3M FLOPs per token here, so
  adding blocks grows the total more slowly than it grows the block count.

Chinchilla's rule counts **parameters**, and tokens are bought with **FLOPs**. Keep them separate.

## Do this

1. In the lab shell, check the arithmetic:

       bash lab/lab.sh shell

       from lib.common import build_model, toy_config
       for depth in (2, 4, 8):
           m = build_model(toy_config(n_layer=depth))
           n = sum(p.numel() for p in m.parameters())
           f = m.estimate_flops()
           print(depth, f"{n/1e6:.2f}M params", f"{f/1e6:.2f}M FLOPs/token")

       run = build_model(toy_config(n_layer=4, n_embd=256, sequence_len=256))
       n = sum(p.numel() for p in run.parameters()); n     # 11,534,600
       1650 * 600 / n                                       # 0.086 tokens per parameter
       1650 * 600 / 20                                      # 49,500: the compute-optimal size

2. Fill in `lab/exercises/lesson_15.py`: `budget_report(model, seconds, flops_per_second)`.

3. Grade it:

       bash lab/lab.sh check 15

## Hints

- `tokens = flops_per_second * seconds / flops_per_token`, rounded down to an int with `int(...)`.
- `params` is every parameter tensor, embeddings included - Chinchilla's parameter count includes
  them all.
- `tokens_per_param = tokens / params`.
- `chinchilla_params = tokens / 20` - the model size those tokens would have been compute-optimal
  for.
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

A time budget is a FLOPs budget, and spending it means trading model size against tokens seen. The
20-tokens-per-parameter rule of thumb puts this fork's CPU run at 0.086 - over two hundred times
short of compute-optimal - and shows the binding constraint is the vocabulary, whose two tables
alone outgrow the ideal model at any usable width. Upstream's GPU run lands near 10, in the right
neighbourhood. Parameters and FLOPs per token are different measures of size, and it pays to keep
them apart. Next module: why the CPU is slow in the first place, and what the port gave up to run
at all.
