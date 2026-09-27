# 10 - Loss, gradients, and what backward() actually does

Modules 1-3 built a model that makes predictions. It starts out predicting at random. This
module is about how it gets better, and it starts with the core loop, which in `train.py` is
literally four lines:

    loss = model(x, y)      # forward:  how wrong are we?
    loss.backward()         # backward: which way should every parameter move?
    optimizer.step()        # update:   move them a little
    model.zero_grad(...)    # reset:    forget this step's gradients

Everything else in the file - the schedules, the accumulation, the timing - is bookkeeping
around those four lines.

## The loss: one number for "how wrong"

`model(x, y)` runs the forward pass and compares the predictions with the real next tokens
`y`. For each position it looks up the probability the model gave to the correct token and
takes `-ln` of it (cross-entropy, lesson 03), then averages over all positions:

    model gave the right token probability 0.9     ->  -ln(0.9)    = 0.105   good
    model gave it probability 0.1                  ->  -ln(0.1)    = 2.303   bad
    model gave it 1/8192 (a uniform guess)         ->  -ln(1/8192) = 9.011   untrained

The whole of training is: make this number smaller.

## What a gradient is, on one weight

Forget the model for a moment and take the smallest possible one: a single weight `w`, one
input `x`, one target `y`, and the loss `(w*x - y)^2`.

    w = 2,  x = 3,  y = 5
    prediction = w * x = 6
    loss = (6 - 5)^2 = 1

The **gradient** `d(loss)/d(w)` answers: *if I nudge `w` up a tiny bit, how much does the loss
change, per unit of nudge?* Here it is `2 * (w*x - y) * x = 2 * 1 * 3 = 6`. Check it by
actually nudging:

    w = 2.01  ->  prediction 6.03  ->  loss 1.0609      the loss rose by ~0.06 = 6 * 0.01

A positive gradient means "raising `w` raises the loss". So to *lower* the loss, move `w`
**against** the gradient:

    w_new = w - learning_rate * gradient = 2 - 0.05 * 6 = 1.7
    prediction = 5.1,  loss = 0.01                      much better

That is all of training, repeated for every number in the model at once. The gradient says
which direction; the optimizer (lesson 11) decides how far.

## What `backward()` does

A real model has millions of weights, and the loss depends on them through a long chain of
operations. `loss.backward()` computes the gradient for **every** parameter in one sweep and
stores it next to the parameter:

    p.grad          same shape as p; entry [i, j] is d(loss)/d(p[i, j])

It does this with the **chain rule**: a gradient through a chain of steps is the product of
each step's local gradient. The toy example is already a two-step chain - `h = w*x`, then
`loss = (h - y)^2`:

    d(loss)/d(h) = 2 * (h - y) = 2        how the loss reacts to h
    d(h)/d(w)    = x           = 3        how h reacts to w
    d(loss)/d(w) = 2 * 3       = 6        multiply them - same answer as before

During the forward pass PyTorch records every operation it performed. `backward()` walks that
record in reverse, from the loss back to the first layer, multiplying local gradients as it
goes. Each layer only needs to know its own local rule; the chain does the rest.

## What "activation" means

Both practical consequences below (memory, and cost) turn on a word that has been used
loosely so far: an **activation**. An activation is any tensor of *values* produced during the
forward pass - most often, one layer's output. It is not a **parameter** (a parameter is a
learned weight, like `c_proj.weight`, that stays attached to the model between batches and
between training steps) and it is not a **gradient** (a gradient is a derivative - one number
per parameter, computed *by* the backward pass, not a value flowing *through* it). The token
embeddings, the output of every block, the wide `(B, T, 4*n_embd)` tensor inside an MLP, the
attention weights, the final logits - all of these are activations. None of them persists
from one training step to the next: they are computed fresh on every forward pass and thrown
away once nothing downstream still needs them, which is exactly why they only cost memory for
as long as some later computation - usually `backward()` - still has to look at them.

One more overload of the word, worth flagging so it does not trip you up elsewhere in this
course: "activation" is also used as shorthand for the *function* applied inside a layer -
lesson 07 called ReLU² "this model's MLP activation", meaning the nonlinearity itself, not the
numbers it produces. Both senses are standard; context tells you which one is meant (an
activation *function* versus *the activations*), but it is worth knowing both exist before you
hit a sentence that assumes you do.

## Two practical consequences

**Memory.** To compute a local gradient, a layer usually needs the input it saw on the way
forward (for `h = w*x`, the gradient for `w` needs `x`) - that is, it needs an *activation*.
So every one of those tensors has to be kept alive until `backward()` has consumed it. That,
not the parameters, usually decides how big a batch fits.

Different tensors cost different amounts, because they have different shapes. The residual
stream (lesson 04) is `(B, T, n_embd)` - call a tensor of exactly that shape **stream-sized**.
For the real CPU training run (`B = 8`, `T = 256`, `n_embd = 256`):

    one stream-sized activation    8 * 256 * 256       =    524,288 floats  ~  2 MB

The MLP (lesson 07) briefly widens the stream to `4 * n_embd` before narrowing it back down,
so the tensor living inside it - between `c_fc` and `c_proj` - is four times as many numbers,
and therefore four times the memory of a stream-sized tensor:

    one MLP-sized activation       8 * 256 * 1024      =  2,097,152 floats  ~  8 MB

And the logits (lesson 09) are `(B, T, vocab_size)`, not `(B, T, n_embd)`. Here
`vocab_size = 8192` is 32 times wider than `n_embd = 256`, so this one tensor is 32 times a
stream-sized one:

    the logits                     8 * 256 * 8192      = 16,777,216 floats  ~ 64 MB

The reason the logits are singled out is not that they are special in kind - they are exactly
as much an activation as any other tensor here - it is that `vocab_size` happens to be far
larger than `n_embd` at this model's scale, and a tensor's memory depends entirely on its
shape.

Now the part that is easy to miss: **no single tensor is what determines whether a batch
fits.** Backward needs the *whole chain* of activations that led to the loss, and forward has
to have created and held onto every one of them before backward can even begin consuming the
first - so peak memory is the moment right after the full forward pass finishes, and it is the
**sum** of everything still alive at that instant, not the size of the largest individual
tensor. Every block keeps several stream-sized tensors (the input to its norms, its `q`/`k`/`v`
projections, its attention weights) *and* one MLP-sized tensor, and with `n_layer` blocks
stacked, that running total adds up across every layer - well before you ever reach the 64 MB
logits tensor.

Measured directly on this course's CPU config (comparing one forward pass run under
`torch.no_grad()`, which keeps nothing around for a backward pass, against the same forward
pass run normally): the *ungraded* pass used about 215 MB of resident memory beyond the
model and the batch; the *graded* one - identical in every other way, just with the
activations kept alive for a `backward()` that has not run yet - used about 431 MB. The
difference, roughly 216 MB, is what keeping the activation chain alive actually cost on this
one `B=8, T=256` batch: several times more than the single 64 MB logits tensor on its own,
because it is a sum across dozens of retained tensors, not the size of the biggest one.

**Cost.** For each matrix multiply, backward computes *two* things: the gradient for the
layer's input (so the gradient can keep flowing back to earlier layers) and the gradient for
the weight (so this layer can learn). Here is exactly where that "twice the forward pass"
figure comes from, worked on numbers small enough to check by hand.

Take a linear layer with `out = 4`, `in = 3` (`12` parameters - about as small as a "matrix
multiply" gets), fed a tiny batch of `B = 2` tokens:

    X (2, 3)  @  W^T (3, 4)  ->  Y (2, 4)          # forward

Counting FLOPs the way everyone in ML does - each multiply is one FLOP, each add is another,
and a length-`k` dot product is treated as `k` multiply-add *pairs*, i.e. `2k` FLOPs - one
output entry of `Y` costs `2 * in = 6` FLOPs, and there are `B * out = 8` output entries, so:

    forward FLOPs  =  2 * B * out * in  =  2 * 2 * 4 * 3  =  48

Backward needs two more matrix multiplies of the *same total size*, computed by the chain
rule: `dL/dX = dL/dY @ W` (shape `(2, 3)`, so the gradient can keep flowing to whatever fed
`X`) and `dL/dW = dL/dY^T @ X` (shape `(4, 3)`, matching `W` itself, so this layer can update).
By the identical `2 * m * n * k` counting rule, each of those costs `48` FLOPs too:

    dL/dX FLOPs  =  2 * 2 * 3 * 4  =  48
    dL/dW FLOPs  =  2 * 4 * 3 * 2  =  48
    backward total = 48 + 48 = 96  =  2 x forward
    training total = 48 + 96 = 144 =  3 x forward

Divide by parameters (`12`) and tokens (`B = 2`): that is `48 / (12*2) = 2` FLOPs per
parameter per token forward, and `96 / (12*2) = 4` FLOPs per parameter per token backward -
exactly the `2N` forward `+ 4N` backward `= 6N` rule from lesson 08, now derived rather than
just asserted, with `2` of the backward's `4` coming from `dL/dX` and the other `2` from
`dL/dW`.

Two honest caveats, both about counting the operations, not just multiplying two of them:

- **Additions count too, and the "2k" shortcut slightly overcounts them.** A length-`k` dot
  product is *exactly* `k` multiplies and `k - 1` adds (the last multiply has no partner left
  to add to) - `2k - 1` operations, not `2k`. For the forward pass above, the *exact* count is
  `8 output entries * (2*3 - 1) = 8 * 5 = 40` FLOPs, not the `48` the `2k` shortcut gives. The
  shortcut overcounts by exactly `1` addition per output entry (`8` entries `x` `1` = `8`,
  and `48 - 40 = 8`). At `in = 3` that is a real, `17%` difference. At a real model's
  `n_embd` - hundreds or thousands, not `3` - the same one-addition-per-entry discrepancy is a
  rounding error, which is *why* the shortcut is the one everyone uses.
- **This whole calculation only counts matrix multiplies.** It says nothing about the
  activation function, the normalisation, the residual addition, or the softmax - each of
  those costs real FLOPs too, just vastly fewer than the matmuls at any reasonable width
  (lesson 08's `attn_flops` term is the one exception big enough to matter, and it gets counted
  separately for exactly that reason). "`6N` FLOPs per token" is a large-matrix approximation
  of the dominant cost, not a literal, cycle-accurate count of every operation a real training
  step performs - accurate exactly where it matters, at the scale where the shared dimension
  `k` is large, and visibly rough on a toy example small enough to check by hand like this one.

## Gradients add up - and that is deliberate

`p.grad` is **added to**, never overwritten. If one backward pass gives a weight gradient
`0.3` and a second, without clearing, gives `0.5`, the stored gradient becomes `0.8`, not `0.5`.

This is useful on purpose: it is exactly how gradient accumulation (lesson 12) works. A batch
too big to fit in memory is split into several smaller *micro-batches*; each one gets its own
forward and backward pass, and because the gradients from each micro-batch simply add into the
same `.grad` tensors, the end result after all of them is identical to having run one single
backward pass on the whole big batch at once - accumulation is what makes that equivalence
possible in the first place.

But it also means every *independent* optimizer step needs a clean slate. If step 2's
gradients silently included step 1's leftovers, the update would be computed from a mixture of
two different points in training - partly the current parameters' gradient, partly a stale
gradient computed before `optimizer.step()` already moved those parameters once. So the loop
must clear the gradients after every update it actually takes (as opposed to every micro-batch
within one accumulated step, where clearing would defeat the whole point):

    model.zero_grad(set_to_none=True)

### `set_to_none=True`, concretely

`set_to_none=True` sets each parameter's `.grad` attribute to `None`, discarding the gradient
tensor entirely, rather than the older default behaviour of overwriting that tensor's values
with zeros while keeping the tensor itself allocated. The difference is not just semantic -
it is a real amount of memory. `wte`, this model's token-embedding table, has shape
`(8192, 256)` at this course's CPU scale: `2,097,152` numbers, `~8 MB` in float32. With the
old zeroing behaviour, that 8 MB stays allocated (now full of zeros) the instant after
`zero_grad()` runs, and only gets reused once the next `backward()` happens to overwrite it in
place. With `set_to_none=True`, that 8 MB is freed *immediately*, and the next `backward()`
allocates a brand-new gradient tensor from the values it computes, rather than performing a
wasted "add these values onto an existing tensor of zeros" step. Both effects are small,
free wins with no downside, which is why `set_to_none=True` is the default recommendation.

### A gradient tensor existing is not the same as it being non-zero

`p.grad is not None` tells you only that a gradient *tensor exists* for that parameter - that
`backward()` reached it and allocated something. It says nothing about what is *inside* that
tensor. A gradient tensor can exist and be entirely zero, and this course's own model produces
exactly that case on purpose: `mlp.c_fc.weight.grad` is a real tensor, the same shape as
`c_fc.weight`, and every one of its entries is `0.0` at step 0 - not because `backward()`
skipped it, but because the gradient that reaches it genuinely evaluates to zero (the next
section explains why). Checking `p.grad is not None` and checking `p.grad.abs().max() == 0`
are two different questions, and it is worth being able to tell them apart on sight: one asks
"did this parameter participate in the computation at all", the other asks "did the loss turn
out to be sensitive to it".

## Measuring "how big" the gradients are

A useful single number for the gradient's overall size is the **global gradient norm**: treat
*every* gradient in the model - across every parameter tensor, whatever its own shape - as one
long list of numbers, and take the length of that combined list (the square root of the sum of
its squares).

The reason this can be computed tensor by tensor, rather than actually having to flatten and
concatenate every parameter's gradient into one giant vector first, is a small piece of algebra
worth seeing once. Take two small vectors, `a = [3, 4]` and `b = [12]`. Concatenated, they form
one vector `c = [3, 4, 12]`, whose L2 norm is:

    ||c|| = sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13

Split the sum back apart at exactly the point where `a` ends and `b` begins:

    3^2 + 4^2 + 12^2  =  (3^2 + 4^2)  +  (12^2)  =  ||a||^2  +  ||b||^2

So `||c|| = sqrt(||a||^2 + ||b||^2)` - the norm of the *concatenation* is recoverable from the
*separate* squared norms of `a` and `b`, added together, with one square root taken at the very
end. Nothing about this depended on `a` and `b` being the same shape, or coming from the same
tensor - it works for any number of pieces, of any shapes, which is exactly the situation with
a model's gradients: `wte.weight.grad` is `(8192, 256)`, `resid_lambdas.grad` is `(4,)`, and so
on for every parameter tensor in the model, all different shapes, none of which need to be
reshaped or moved into one physical array to compute one combined norm. Sum each tensor's own
`(gradient ** 2).sum()`, add all of those sums together, and take one square root at the end:

    sqrt(3^2 + 4^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13

A sudden jump in this number is often the first warning of a run about to blow up. The exercise
computes it.

## The zero-initialised layers, one more time

Lesson 07 showed that `mlp.c_proj` starts at exactly zero and still learns, because its weight
gradient does not involve its own current value at all:

    grad_W = grad_out * activation

With numbers (lesson 07's own example): `grad_out = 2`, and the activation feeding into
`c_proj` - the ReLU² output of the random, non-zero `c_fc` - happens to be `5`, so
`grad_W = 2 * 5 = 10`, even though `c_proj`'s weight `W` is exactly `0`. The weight's own value
simply never enters this formula, so starting it at zero costs nothing: it still gets a real,
non-zero gradient on the very first step, purely from the activation flowing into it.

The same rule has a mirror image, and the check tests both. The gradient passed *backwards
through* a layer, to the layer feeding it, is `W^T * grad_out` - and unlike `grad_W`, that one
**does** involve the weight's own current value. With `c_proj`'s `W = 0`:

    gradient reaching c_fc = 0^T * grad_out = 0

So on step 0, `c_proj` gets a real, non-zero gradient (as just shown), while `c_fc` (the layer
feeding it) gets a gradient tensor that exists but is **exactly zero** throughout - precisely
the "tensor exists, but every entry is 0" case from a few sections up. After one update
`c_proj` is no longer zero, and from step 1 on, `c_fc` receives a real gradient too. Normally
"this weight has zero gradient" means a bug. Here it is by design, and it fixes itself after a
single step.

Zero `c_fc` instead of `c_proj`, and nothing ever fixes itself - trace the same two formulas
through in the other order:

    c_fc's own output      = 0                       for every input (its weight is zero)
    activation reaching c_proj = relu(0)^2  = 0        so c_proj's grad_W = grad_out * 0 = 0
    c_fc's own grad_W       = grad_out * (slope of relu^2 at 0) = grad_out * 0 = 0

`c_proj` gets no gradient this time, because the activation feeding into it is zero. `c_fc`
gets no gradient either, because ReLU² has slope exactly `2 * relu(0) = 0` at zero. Neither
weight moves, so on the next step the same two zeros produce the same two zero gradients again,
forever. The branch is permanently dead. That is why the zero has to go on the layer that
**writes out** (whose gradient depends on someone else's activation), never on the layer that
**reads in** (whose output is what the next layer's gradient depends on).

## Fast fail

    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

Two distinct ways a run can break, and why either one is grounds to stop immediately rather
than let the run continue to its scheduled end:

- **`NaN`** ("not a number") - the result of an operation like `0/0` or `inf - inf`, both of
  which genuinely have no defined numeric answer. Any further arithmetic involving a `NaN`
  produces another `NaN` (`NaN + 1 = NaN`, `NaN * 0 = NaN`), so once a single one appears
  anywhere in the model, it spreads to every parameter it touches within one backward pass -
  there is no way for training to recover on its own, and every subsequent number the run
  reports (loss, `val_bpb`, everything) is contaminated and meaningless.
- **An exploding loss.** An untrained model sits at `~9` (lesson 03's `ln(vocab)`). A loss
  above `100` means the weights have diverged into a region training will not climb back out
  of on its own - the run's final answer, whenever it finishes, would still be garbage.

Continuing either kind of broken run to its natural end wastes exactly the time budget that
was supposed to buy a real comparison (lesson 12), and only *then* reports a useless result -
so in an autonomous research loop (lesson 18) that is running experiments unattended, catching
this in the first few seconds and moving on to the next idea is strictly better than
discovering a `NaN` `val_bpb` ten minutes later, having burned the whole budget on nothing.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, get_batch
       model = build_model(); x, y = get_batch(B=2, T=64)

       loss = model(x, y); loss.item()                  # ~9.011, a uniform guess
       [p.grad for p in model.parameters()][:2]          # all None before backward
       loss.backward()
       model.transformer.wte.weight.grad.abs().max()     # non-zero now

       c_proj = model.transformer.h[0].mlp.c_proj
       c_proj.weight.abs().max()                         # 0.0 - the weight is zero
       c_proj.weight.grad.abs().max()                    # but its gradient is NOT
       c_fc = model.transformer.h[0].mlp.c_fc
       c_fc.weight.grad is not None                       # True - the tensor exists...
       c_fc.weight.grad.abs().max()                       # ...and is 0.0 - blocked by c_proj

       before = c_proj.weight.grad.norm().item()
       model(x, y).backward()                            # again, without clearing
       c_proj.weight.grad.norm().item() / before         # ~2.0 - gradients accumulate

   And the one-weight example, so the chain rule is something you have run, not read:

       w = torch.tensor(2.0, requires_grad=True)
       loss = (w * 3 - 5) ** 2; loss.backward(); w.grad   # tensor(6.)

   The tiny matmul from the FLOPs derivation, to see the shapes for yourself:

       X = torch.randn(2, 3, requires_grad=True)
       W = torch.randn(4, 3, requires_grad=True)
       Y = X @ W.T; Y.shape                               # torch.Size([2, 4])
       Y.sum().backward()
       X.grad.shape, W.grad.shape                         # ([2, 3]), ([4, 3]) - dL/dX, dL/dW

2. Fill in `lab/exercises/lesson_10.py`: `grad_report(model, x, y)`.

3. Grade it:

       bash lab/lab.sh check 10

## Hints

- Clear any leftover gradients **first** (`model.zero_grad(set_to_none=True)`), or a second
  call to your function returns the sum of two backward passes and the check will catch it.
- The global gradient norm is the L2 norm of all gradients treated as one long vector:
  `sqrt(sum(g.pow(2).sum() for g in grads))`. Use `.item()` to return a Python float.
- `n_with_grad` counts parameter **tensors** whose `.grad is not None` - existence, not
  whether the tensor happens to be all zeros (`c_fc` at step 0 still counts).
- `model.parameters()` is a generator - it can only be looped over once. If you need it twice,
  make a list first.
- Do not call `optimizer.step()`; this exercise stops at the gradients.

## Solution

    import torch

    def grad_report(model, x, y) -> dict:
        model.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]
        total_sq = sum(g.pow(2).sum() for g in grads)
        return {
            "loss": loss.item(),
            "grad_norm": total_sq.sqrt().item(),
            "n_with_grad": len(grads),
            "n_params": len(params),
        }

## Summary

The loss is one number for how wrong the predictions are. A gradient says how the loss reacts
to nudging one parameter, and training moves every parameter a little against its gradient.
An activation is a value flowing through the forward pass - not a parameter, not a gradient -
and `backward()` needs the whole chain of them alive at once, which is why memory is a sum
across many retained tensors rather than the size of the biggest one. Backward computes two
matrix multiplies for every one the forward pass did (one for the input, one for the weight),
which is where the "twice the forward pass, `6N` FLOPs total" rule actually comes from, and
why it is an approximation that is excellent at real model widths and visibly rough on a
hand-checkable toy example. Gradients add up until you clear them; a gradient tensor existing
is not the same as it being non-zero; and a zero-initialised output layer still learns while
briefly blocking the gradient to the layer before it. Next: what the optimizer does with those
gradients, and why this repo uses two different optimizers at once.
