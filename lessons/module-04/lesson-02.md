# 11 - Two optimizers in one model: AdamW and Muon

Lesson 10 ended with a gradient for every parameter. The **optimizer** is the code that turns
those gradients into an actual change to the weights. The simplest possible optimizer is one
line:

    p -= lr * grad          # plain gradient descent

Nobody trains a transformer that way, and the reason is easiest to see with two numbers.
Suppose one weight has gradient `100` and another has gradient `0.001`, and `lr = 0.01`:

    step for weight 1:  0.01 * 100    = 1.0          huge - probably overshoots
    step for weight 2:  0.01 * 0.001  = 0.00001      tiny - effectively never learns

No single `lr` suits both. Different parameters need step sizes that differ by orders of
magnitude, and the right size changes as training goes on. Every modern optimizer is an answer
to one question: **how big a step, for this parameter, right now?**

This repo answers it **twice**: AdamW for embeddings and the few 1-D scalars, Muon for the 2-D
weight matrices inside the blocks. That is unusual, and it is the most modern thing in the
file.

## AdamW, one idea at a time

The repo's AdamW kernel is these lines:

    p.mul_(1 - lr * wd)                              # weight decay
    exp_avg.lerp_(grad, 1 - beta1)                   # m: smoothed gradient
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)       # v: smoothed squared gradient
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-lr / bias1)

It looks dense. It is four separate ideas, and this lesson traces **one single parameter**
through all four of them with one consistent set of numbers, so you can follow the same `p`
from the top of the function to the bottom.

**The running example.** One parameter `p = 1.0`. Its gradient this step is `grad = 0.6`.
`lr = 0.1`, `beta1 = 0.8`, `beta2 = 0.95`, `wd = 0.1`, `eps` negligibly small. Both averages
(`exp_avg` and `exp_avg_sq`) start at `0`, because this is the parameter's very first
optimizer step (`step = 1`).

### Idea 1: momentum - smooth out the noise

Each batch is a different handful of text, so each gradient is noisy: the true direction plus
batch-specific jitter. `exp_avg` (call it `m`) is a running average that remembers past
gradients and slowly forgets them.

`a.lerp_(b, w)` is PyTorch's in-place "linear interpolation": it means `a = a + w * (b - a)`,
which rearranges to `a = (1 - w) * a + w * b` - a weighted blend of the old value and the new
one. Applied here, `exp_avg.lerp_(grad, 1 - beta1)` sets:

    m = beta1 * m + (1 - beta1) * grad

With `beta1 = 0.8`, each step keeps 80% of the old average and mixes in 20% of the new
gradient. On our running example, starting from `m = 0`:

    m = 0.8 * 0 + 0.2 * 0.6 = 0.12

To see *why* this smooths noise, watch `m` react to a gradient that keeps flipping sign over
several steps - `grad: 1, -1, 1, 1` (a separate illustration, not our running example):

    grad:   1       -1       1        1
    m:      0.200   -0.040   0.168    0.334

The `-1` barely dents it. Over several steps, directions that are consistent build up and
directions that flip back and forth cancel out.

### Idea 2: divide by the typical size - one step size for everyone

`exp_avg_sq` (call it `v`) is the same kind of running average, but of the **squared**
gradient:

    v = beta2 * v + (1 - beta2) * grad^2

Squaring throws away the sign. A gradient of `+5` and a gradient of `-5` contribute identically
to `v` (both square to `25`), so `v` only ever tracks how *big* the gradient typically is,
never which way it points. That is the real distinction between `m` and `v`: `m` is a smoothed
**direction** (it can be positive, negative, or hover near zero if the sign keeps flipping); `v`
is a smoothed **magnitude** (it only ever grows toward whatever size the recent gradients have
been, regardless of sign).

Run the same flip-flopping sequence from above through `v` (`beta2 = 0.95`, starting from `0`):

    grad:   1       -1       1        1
    m:      0.200   -0.040   0.168    0.334
    v:      0.050    0.098    0.143   0.185

Notice `v` never dips or changes sign the way `m` does at step 2 - it climbs steadily toward
`grad^2 = 1`, unaffected by the sign flip. `sqrt(v)` is therefore a running estimate of how
big this parameter's gradient *usually* is, independent of direction.

On our running example (single step, `grad = 0.6`, starting `v = 0`):

    v = 0.95 * 0 + 0.05 * 0.6^2 = 0.05 * 0.36 = 0.018

The step is `m / sqrt(v)`: the smoothed gradient divided by its own typical size. This is what
makes the two weights from the top of the lesson comparable - a *different* illustration, this
time comparing two different parameters rather than one parameter over time:

    weight 1: gradient always ~100     m ~ 100,    sqrt(v) ~ 100,    m/sqrt(v) ~ 1
    weight 2: gradient always ~0.001   m ~ 0.001,  sqrt(v) ~ 0.001,  m/sqrt(v) ~ 1

Both now take a step of about `lr`. The size of the raw gradient no longer matters, only its
direction and consistency. That per-parameter normalisation is what makes Adam robust. `eps`
(here `1e-10`) is added only to avoid dividing by zero.

### Idea 3: bias correction - fix the cold start

Both averages start at zero. After the first step, `m` is only `0.2 * grad` (as computed
above: `0.12`, when `grad = 0.6`) - five times too small, purely because of the zero it
started from. `bias1 = 1 - beta1^step` undoes exactly that:

    step 1:  bias1 = 1 - 0.8^1 = 0.2

Divide our running example's `m` by it:

    m / bias1 = 0.12 / 0.2 = 0.6           <- exactly the raw gradient again

At step 1, from a cold start, dividing by `bias1` exactly recovers the original gradient - the
correction fully undoes the damping caused by starting `m` at zero. `bias2` does the same job
for `v`:

    step 1:  bias2 = 1 - 0.95^1 = 0.05
    v / bias2 = 0.018 / 0.05 = 0.36        <- exactly grad^2 (0.6^2 = 0.36) again

Ten steps in, the correction has almost nothing left to do:

    step 10: bias1 = 1 - 0.8^10 = 0.89     barely any correction left

The combined effect is neat: on step 1, every parameter moves by exactly `lr` in the direction
of its gradient's sign, however large or small that gradient was - which is exactly what Idea
2's normalisation already promised, now confirmed with bias correction in the mix too.

### The `denom` line, step by step

    denom = (exp_avg_sq / bias2).sqrt() + eps

Three steps, with our running numbers:

    bias correction:   v / bias2  =  0.018 / 0.05  =  0.36
    square root:        sqrt(0.36)  =  0.6
    add eps:             0.6 + 1e-10  ~  0.6     (eps only matters when this would be ~0)

`eps` exists for the case `v` (and therefore this whole expression) is still exactly or nearly
zero - for instance a brand-new parameter that has seen no gradient yet, or one that has
received several exactly-zero gradients in a row. Without `eps`, dividing by that zero-ish
`denom` in the next line would produce `inf` or `NaN`. In our example it changes nothing
(`0.6` either way), because `v` is safely away from zero.

### Idea 4: the W - decoupled weight decay

**Weight decay** (`wd`) is a hyperparameter that pulls every weight a little toward zero at
every step, *independent of the gradient*. It defends against weights growing without bound
over a long training run, which tends to hurt generalisation and can destabilise training.

    p.mul_(1 - lr * wd)

With `lr = 0.1` and `wd = 0.1`, the multiplier is `1 - 0.1 * 0.1 = 0.99`. Applied to our
running example's `p = 1.0`:

    p = 1.0 * 0.99 = 0.99

This is the very first line of the kernel - it runs before anything involving the gradient.

The **W** in AdamW stands for doing this *separately* ("decoupled") from the gradient. The older
way added `wd * p` into the gradient - but then Idea 2's `sqrt(v)` divides it away, so
parameters with big gradients got almost no decay. Doing it as its own line, on `p` directly,
gives every weight the same relative shrink regardless of its gradient history.

**In this repo, every AdamW group has `weight_decay = 0.0`.** So this line multiplies `p` by
`1 - lr * 0 = 1` for every AdamW-optimised parameter (the embeddings and the scalars) - it does
nothing there. Decay is used only by the Muon groups, through a different mechanism (cautious
weight decay, further down).

### Putting it together: the final update, on our running example

The last line is:

    p.add_(exp_avg / denom, alpha=-lr / bias1)

Read as ordinary arithmetic, this is:

    p = p - lr * (exp_avg / bias1) / denom

and since `denom` is (up to the negligible `eps`) equal to `sqrt(exp_avg_sq / bias2)`, this is
the same update most people write when they explain Adam:

    p = p - lr * (m / bias1) / sqrt(v / bias2)

Plug in our running numbers - `p` is already `0.99` after decay, `m / bias1 = 0.6`,
`sqrt(v / bias2) = 0.6`:

    p = 0.99 - 0.1 * (0.6 / 0.6)
      = 0.99 - 0.1 * 1
      = 0.89

`p` moved from `1.0` to `0.89` on this step - a decay-shrink of `0.01`, plus a gradient-step of
exactly `0.1` in the direction the gradient pointed. (This matches what `adamw_step_fused`
actually computes; the Do-this section has you verify it in the lab.)

### What each idea contributes

- **Momentum** (`m`) - smooths a noisy per-batch gradient into a steadier direction to move in.
- **Normalisation** (dividing by `sqrt(v)`) - makes the step size comparable across parameters
  whose gradients happen to be huge or tiny, so one `lr` works for all of them.
- **Bias correction** (`bias1`, `bias2`) - removes the artificial damping caused by starting
  both averages at zero, so early steps are not systematically too small.
- **Decoupled weight decay** (`wd`) - shrinks weights toward zero at a rate that does not
  depend on the gradient's size, unlike the older L2-penalty approach.

## Muon: make every direction of a matrix learn at the same speed

Muon is used for exactly the 2-D weight matrices inside `transformer.h`: `c_q`, `c_k`, `c_v`,
both `c_proj`s, `c_fc`, and `ve_gate`.

### Where Muon fits in the training loop

Muon is not a separate training process - it is one more step at the very end of the loop you
already know from lessons 10-11:

    x -> W -> prediction -> loss -> backward() -> gradient G -> Muon -> updated W

Everything up to "gradient `G`" is exactly what lesson 10 covered: a forward pass, a loss, and
`backward()` filling in `.grad` for every parameter. Muon's whole job starts *after* that - it
takes the gradient matrix `G` that backprop already computed for one weight matrix `W`, and
decides how to turn it into an update.

Here is a complete tiny example, small enough to do by hand, with **two** training examples (one
example alone would only ever produce a rank-1 gradient, which hides half of what Muon does -
lesson 08's arithmetic on outer products explains why).

    W (2x2, current weights, start at the identity):
        W = [[1, 0],
             [0, 1]]

    Example 1:  x1 = [1, 0]     y1 = [2, 0]   (target)
    Example 2:  x2 = [0, 1]     y2 = [0, 3]   (target)

**Forward pass**, `prediction = W @ x`:

    p1 = W @ x1 = [1, 0]
    p2 = W @ x2 = [0, 1]

**Loss** (mean squared error, `0.5 * sum((prediction - target)^2)`, one example at a time):

    loss1 = 0.5 * ((1-2)^2 + (0-0)^2) = 0.5 * 1 = 0.5
    loss2 = 0.5 * ((0-0)^2 + (1-3)^2) = 0.5 * 4 = 2.0
    total loss = 0.5 + 2.0 = 2.5

**Backward pass.** For `prediction = W @ x` and this loss, the standard result (lesson 10's
chain rule, applied to a matrix) is `dL/dW = (prediction - target) outer x` for each example,
summed over the batch:

    error1 = p1 - y1 = [-1, 0]
    error2 = p2 - y2 = [0, -2]

    G = outer(error1, x1) + outer(error2, x2)
      = [[-1, 0], [0, 0]]  +  [[0, 0], [0, -2]]
      = [[-1,  0],
         [ 0, -2]]

This `G` is the gradient - `backward()`'s actual output for this weight matrix. Note carefully:
**`G` is not `W`.** They happen to be the same shape (2x2) because a gradient always matches
the shape of the parameter it belongs to, but `G` holds "how the loss wants every entry of `W`
to change", not a weight value. `W` is still `[[1, 0], [0, 1]]` at this point - nothing has been
updated yet.

`G`'s singular values (defined below) are `2` and `1` - a real, if modest, imbalance: one
direction the loss wants to push twice as hard as the other. A real training step's gradient
is the same idea at a much bigger scale - averaged over a whole batch of many tokens, not two
toy examples - and its singular values are typically far more unequal than 2-to-1.

Everything from here on - momentum, orthogonalising, scaling by `lr` - operates on `G` (and the
things derived from it). **`W` is only ever touched once, at the very last step, when the
finished update is subtracted from it.** Muon never inspects `W`'s own values while deciding
what the update should look like.

### The problem it solves

A weight matrix is not just a bag of independent numbers - it maps input directions to output
directions. Any matrix can be broken down (by the *singular value decomposition*, SVD) into a
set of independent directions, each with a strength called a **singular value** (the next
section explains this properly).

The gradient for a matrix is itself a matrix, and for transformers its singular values are
usually wildly unequal - much more so than our tiny 2-to-1 example above. A deliberately extreme
illustration, purely to make the effect obvious:

    G = [[10,  0  ],
         [ 0,  0.1]]

This gradient wants to push hard along the first direction and barely at all along the second:
a 100-to-1 ratio. Take the step as-is and the matrix mostly learns along one or two dominant
directions, while the rest crawl.

### SVD, intuitively - and why "equal singular values" is not "entries equal to 1"

Any matrix `G` can be written as `G = U @ Sigma @ V^T`, where:

- `U` and `V` are **orthogonal** matrices - pure rotations (or reflections), with unit-length,
  mutually perpendicular rows and columns. They never stretch or shrink anything, only turn it.
- `Sigma` is **diagonal**, holding the **singular values** - one non-negative number per
  direction, saying how much that direction gets *stretched*.

So `G` acting on a vector is: rotate it (`V^T`), stretch each axis independently by that axis's
singular value (`Sigma`), rotate again (`U`). The singular values are the only part of this
picture that says "how strong" each direction is; `U` and `V` say "which directions".

This is worth pinning down with a concrete counter-example, because it is easy to assume
"singular values near 1" means "matrix entries near 1" - it does not. Take this pure rotation
matrix:

    R = [[ 0.6, -0.8],
         [ 0.8,  0.6]]

None of its four entries is anywhere near `1`. Yet `R`'s singular values are **exactly** `1`
and `1` (you can check: `R^T @ R` comes out to the identity matrix, which is exactly what
"singular values all equal 1" means for a square matrix). `R` only rotates - it does not
stretch anything - so every direction is preserved at strength exactly 1, regardless of what
its individual entries look like.

Compare that with our diagonal `G = [[10, 0], [0, 0.1]]` above. For a **diagonal** matrix
specifically, the rotation parts `U` and `V` are trivial (the identity, give or take signs), so
the singular values happen to equal the absolute values of the diagonal entries directly - `10`
and `0.1`. That convenient coincidence is *why* this lesson's toy examples use diagonal
matrices: it lets you read the singular values straight off the page. It is a special case, not
the general rule - for the earlier 2x2 pipeline example (which was diagonal too, by
construction of that particular `x1`, `x2`), the same shortcut applied and gave singular values
`2` and `1` directly from the entries `-1` and `-2`. A general, non-diagonal gradient does not
offer that shortcut, and its singular values have to be computed from `G^T @ G`'s eigenvalues,
not read off `G` itself.

**What Muon actually changes**, in this language: it replaces `Sigma` with (approximately) the
identity matrix, while leaving `U` and `V` - *which* directions matter - untouched. It does not
touch `W`'s entries directly at all, and it does not make `G`'s own entries equal to 1 either;
it makes `G`'s *singular values* close to 1, the same way `R` above has entries nowhere near 1
but singular values of exactly 1.

### The fix: orthogonalise

Muon replaces the gradient with the closest matrix that has **all singular values equal to 1**:
same directions (`U`, `V` unchanged), equal strength (`Sigma = I`, an *orthogonal* matrix, or
semi-orthogonal when it is not square). For the illustrative 100-to-1 example above:

    G after Muon ~ [[0.91, 0   ],
                    [0,    1.00]]

The 100-to-1 ratio is gone (down to about 1.1-to-1). Every direction now gets a comparable
update. In practice this trains small transformers noticeably faster per step than Adam, which
is why it is here.

### How, without an SVD: the Newton-Schulz code, line by line

Computing an SVD directly is slow, especially on a GPU. Muon gets the same answer
approximately with a **Newton-Schulz iteration**, which uses only matrix multiplies:

    X = G / (||G|| * 1.02 + 1e-6)              # scale so the largest singular value < 1
    for a, b, c in polar_express_coeffs[:5]:
        A = X.T @ X
        B = b*A + c*(A @ A)
        X = a*X + X @ B

Read it one line at a time, on the `G = [[10, 0], [0, 0.1]]` example:

- **`X = G / (||G|| * 1.02 + 1e-6)`.** `||G||` (the matrix norm) is at least as large as `G`'s
  biggest singular value, so dividing by a slightly-inflated version of it (`* 1.02`) guarantees
  every singular value of `X` starts strictly below 1. This matters because the polynomial
  below only pulls singular values *toward* 1 from inside a certain range - starting above 1
  can make it diverge instead of converge. (`+ 1e-6` only guards against `||G|| = 0`.)
- **`A = X.T @ X`.** This is a standard fact about the SVD: for any matrix `X = U Sigma V^T`,
  `X^T @ X` works out to `V Sigma^2 V^T` - a symmetric matrix whose eigenvalues are exactly the
  *squares* of `X`'s singular values, and whose eigenvectors are `X`'s right singular vectors
  `V`. So `A` is a compact stand-in for "the singular values, squared, with the directions
  still attached."
- **`B = b*A + c*(A @ A)`.** A polynomial in `A`. Because `A` and `A @ A` share the exact same
  eigenvectors as `A` (any polynomial of a matrix has the same eigenvectors as the matrix
  itself), this step can only ever *rescale* each singular direction - it cannot rotate or mix
  directions together. That is precisely why the singular vectors (`U`, `V` - which directions
  matter) come out of the whole loop unchanged, while only the singular values (`Sigma` - how
  strong each direction is) move.
- **`X = a*X + X @ B`.** Combines everything into the next `X`. Written out for a single
  singular value `s` of `X`, one pass of this whole loop body is equivalent to computing
  `f(s) = a*s + b*s^3 + c*s^5` - an odd polynomial in `s` alone. The five `(a, b, c)` triples in
  `polar_express_coeffs` are pre-tuned (the "Polar Express" method) so that applying this
  polynomial five times in a row pushes any starting `s` in the valid range toward `1`, faster
  than the textbook Newton-Schulz coefficients would.

Running the actual five passes on the `G = [[10, 0], [0, 0.1]]` example (verified in the lab)
shows it is **not** a smooth march toward 1 - it overshoots first:

    start:          10.00    0.10       (as singular values, before normalising)
    after pass 1:    1.19    0.08
    after pass 2:    1.27    0.32
    after pass 3:    0.94    1.16
    after pass 4:    1.46    1.09
    after pass 5:    0.91    1.00

Five passes get close enough - on a random badly-conditioned matrix, most singular values land
between roughly 0.75 and 1.15, and directions that started essentially at zero stay small. That
is close enough, and far cheaper than an SVD.

### Momentum, before orthogonalising: why, what it represents, and the two lerps

The very first thing that happens to `G` in the real optimizer, before any of the Newton-Schulz
machinery above, is the same idea as AdamW's Idea 1: **raw per-batch gradients are noisy**, and
orthogonalising a noisy gradient just gives you a confidently-wrong direction. So Muon smooths
`G` with a momentum buffer first - a per-parameter-matrix running average, persisted in the
optimizer's state across steps, exactly analogous to AdamW's `exp_avg`.

The code is two lines:

    momentum_buffer.lerp_(grads, 1 - momentum)
    return torch.lerp(grads, momentum_buffer, momentum)

The **first line** is ordinary momentum, identical in spirit to AdamW's `m`:

    buf = momentum * buf + (1 - momentum) * grad

The **second line** is what makes it *Nesterov* momentum rather than plain momentum: instead of
just using `buf` itself as the direction to orthogonalise, it looks one step further ahead by
blending the *raw current gradient* toward the *just-updated* `buf`:

    direction = grad + momentum * (buf - grad) = (1 - momentum) * grad + momentum * buf

Plain momentum would hand Newton-Schulz the (slightly stale) `buf` on its own. Nesterov's
extra blend reacts a little faster to a genuine change in gradient direction, because it always
includes a slice of *this step's* raw gradient rather than relying entirely on the lagging
average.

**Worked on our pipeline example**, continuing with `G = [[-1, 0], [0, -2]]` from above,
`momentum = 0.85` (the value this repo starts Muon's momentum at), and `buf` starting at zero
(this parameter's very first step):

    buf = 0.85 * 0 + 0.15 * G = 0.15 * [[-1, 0], [0, -2]] = [[-0.15, 0], [0, -0.3]]

    direction = G + 0.85 * (buf - G)
              = [[-1, 0], [0, -2]] + 0.85 * [[0.85, 0], [0, 1.7]]
              = [[-1, 0], [0, -2]] + [[0.7225, 0], [0, 1.445]]
              = [[-0.2775, 0], [0, -0.555]]

(On a *first* step specifically, with `buf` starting at zero, this always works out to
`direction = (1 - momentum^2) * G` - here `1 - 0.85^2 = 0.2775`, matching the arithmetic above.
That shortcut only holds for the very first step; from the second step on, `buf` carries real
history and the two lines have to be run in full.)

This `direction` matrix - not the original `G` - is what gets fed into the Newton-Schulz loop
from the previous section. Its singular values are `0.555` and `0.2775` (still the same 2-to-1
ratio as `G`, just uniformly scaled down - Nesterov's blend on a first step never changes the
*ratio* between singular values, only their overall size); orthogonalising pushes both toward
1, exactly as the diagonal example demonstrated.

### Two more refinements on top

The rest of `muon_step_fused` adds two more steps after orthogonalising:

- **NorMuon variance reduction** - Idea 2 from the AdamW section, again: a per-row/column
  running estimate of typical update size, used to rescale the orthogonalised update so that
  rows or columns which have recently needed bigger corrections do not permanently dominate.
- **Cautious weight decay** - `mask = (g * p) >= 0`. When the (already-Muon-processed) update
  `g` and the current weight `p` have the same sign, the step `-lr * g` is already moving that
  weight toward zero, and decay is applied on top. Where the update wants the weight to *grow*
  (opposite signs), decay is skipped, so it never fights the direction Muon just chose.

### The algorithm, end to end

Every piece above is one stage of a single pipeline, run once per weight matrix, per step:

    1. backward() computes G = dL/dW                         (lesson 10 - unchanged by Muon)
    2. Nesterov momentum smooths G into `direction`,          (this section)
       using this parameter's persistent momentum buffer
    3. normalise + 5 Newton-Schulz passes orthogonalise       (this section)
       `direction`: singular values -> ~1, directions kept
    4. NorMuon rescales the orthogonalised update              (previous section)
       per row/column
    5. cautious weight decay optionally shrinks W a little     (previous section)
       more, where the update agrees with W's own sign
    6. W -= lr * (the finished update from steps 2-5)          (the only line that touches W)

Step 6 is the only point in the whole pipeline where `W` itself is read or written. Every
earlier step operates purely on the gradient and its derived matrices (`direction`, the
orthogonalised `X`, the NorMuon-scaled version) - which is the precise sense in which "Muon
does not simply change the values of the weight matrix to 1": it never looks at `W`'s values
at all until the final subtraction, and what gets subtracted is a small, `lr`-scaled update
whose *singular values* are near 1, not a matrix of 1s.

## Who gets which optimizer, and at what learning rate

With the values `train.py` passes in:

    adamw: lm_head        lr = 0.004 * scale
    adamw: wte            lr = 0.6   * scale
    adamw: value_embeds   lr = 0.6   * scale
    adamw: resid_lambdas  lr = 0.005
    adamw: x0_lambdas     lr = 0.5            betas (0.96, 0.95)
    muon:  every 2-D matrix in transformer.h, grouped by shape, lr = 0.04

**Why embeddings get a big learning rate.** An embedding table is a lookup: a batch only
*uses* the rows for the tokens that appear in it, so each row gets a gradient only
occasionally, and can afford a much bigger step when it does (`0.6`). `lm_head` is the same
shape, but it is a real matmul that scores *every* token at *every* position, so every row gets
a gradient on every step. It gets `0.004` - **150 times smaller**. These are the kind of
numbers a research loop finds; do not change them casually.

**Why the scale factor.**

    dmodel_lr_scale = (model_dim / 768) ** -0.5

The AdamW learning rates were tuned for a model of width 768. Wider models need smaller steps,
and this scales them by `1/sqrt(width / 768)`:

    n_embd = 768   ->  x 1.00
    n_embd = 256   ->  x 1.73      (the real CPU training run)
    n_embd = 128   ->  x 2.45      (the course's toy model)

This is what lets hyperparameters survive a change of width - "try depth 6 instead of 4" is a
one-line experiment instead of a re-tuning project.

**Why Muon groups by shape.** `_step_muon` stacks every matrix of the same shape into one 3-D
tensor and orthogonalises them all in a single batched call. At the course's scale that is 8
matrices of `(128, 128)`, 2 of `(512, 128)`, 2 of `(128, 512)`, and 1 `ve_gate` of `(2, 32)` -
four groups, four calls.

## Do this

1. In the lab shell:

       bash lab/lab.sh shell

       import torch
       from lib.common import build_model, train_defs
       defs = train_defs(); model = build_model()
       opt = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.6,
                                   scalar_lr=0.5, matrix_lr=0.04, weight_decay=0.2)
       for g in opt.param_groups:
           print(g["kind"], len(g["params"]), round(g["lr"], 5), tuple(g["params"][0].shape))

   (Pass the values explicitly: `setup_optimizer()`'s own defaults are older numbers, not
   the ones `train.py` uses.)

   Reproduce the lesson's single-parameter AdamW trace exactly:

       p = torch.tensor(1.0); grad = torch.tensor(0.6)
       m, v = torch.tensor(0.0), torch.tensor(0.0); T = torch.tensor
       defs.adamw_step_fused(p, grad, m, v, T(1.), T(0.1), T(0.8), T(0.95), T(1e-10), T(0.1))
       p                                               # tensor(0.89)  - matches the lesson

   And Adam's first step on gradients 100,000 times apart:

       p = torch.tensor([1.0, 1.0]); g = torch.tensor([100.0, 0.001])
       m, v = torch.zeros(2), torch.zeros(2)
       defs.adamw_step_fused(p, g, m, v, T(1.), T(0.01), T(0.8), T(0.95), T(1e-10), T(0.))
       p                                               # tensor([0.99, 0.99]) - same step

   Reproduce the Muon pipeline example end to end:

       x1, y1 = torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])
       x2, y2 = torch.tensor([0.0, 1.0]), torch.tensor([0.0, 3.0])
       W0 = torch.eye(2)
       err1 = W0 @ x1 - y1; err2 = W0 @ x2 - y2
       G = torch.outer(err1, x1) + torch.outer(err2, x2); G     # [[-1, 0], [0, -2]]
       torch.linalg.svdvals(G)                                  # tensor([2., 1.])

       buf = torch.zeros(2, 2)
       buf.lerp_(G, 1 - 0.85)
       direction = torch.lerp(G, buf, 0.85); direction           # [[-0.2775, 0], [0, -0.555]]

       X = direction / (direction.norm() * 1.02 + 1e-6)
       for a, b, c in defs.polar_express_coeffs[:5]:
           A = X.mT @ X
           X = a * X + X @ (b * A + c * (A @ A))
       torch.linalg.svdvals(X)                                  # ~[1.12, 0.93]

       W0 - 0.1 * X                                             # the updated weight matrix

   What orthogonalisation does to a badly conditioned matrix (the illustrative 100-to-1 case):

       torch.manual_seed(0)
       G = torch.randn(64, 64) @ torch.diag(torch.linspace(1, 0.001, 64))
       torch.linalg.svdvals(G)                         # from ~10 down to ~0.0001
       X = G / (G.norm() * 1.02 + 1e-6)
       for a, b, c in defs.polar_express_coeffs[:5]:
           A = X.mT @ X
           X = a * X + X @ (b * A + c * (A @ A))
       torch.linalg.svdvals(X)                         # mostly ~0.75 to ~1.15

2. Fill in `lab/exercises/lesson_11.py`: `adamw_update(...)` and `nesterov_momentum(...)`.

3. Grade it:

       bash lab/lab.sh check 11

## Hints

- `adamw_update` changes `p`, `exp_avg` and `exp_avg_sq` **in place** (the `_`-suffixed
  methods like `mul_`, `lerp_`, `add_`), in the order the kernel does: decay `p` first, then
  update both averages, then the parameter step. Getting the order wrong changes the answer by
  one step's worth of decay, and the check will see it.
- `a.lerp_(b, w)` means `a = a + w * (b - a)`, which is `(1 - w) * a + w * b`. So
  `exp_avg.lerp_(grad, 1 - beta1)` is `beta1 * exp_avg + (1 - beta1) * grad` - Idea 1 exactly.
- `step` counts from 1, not 0. `bias1 = 1 - beta1**step`; at step 0 it would be 0 and you would
  divide by zero.
- `eps` is added **after** the square root, not inside it.
- `nesterov_momentum` must not modify `grads`. `momentum_buffer` is updated in place;
  return the direction to step along.
- The two lines: first update the buffer as ordinary momentum,
  `momentum_buffer.lerp_(grads, 1 - momentum)`. Then look ahead by blending the raw gradient
  toward that updated buffer, `torch.lerp(grads, momentum_buffer, momentum)` - the non-`_`
  version, which returns a new tensor instead of changing `grads`.

## Solution

    import torch

    def adamw_update(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
        p.mul_(1 - lr * wd)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        bias1 = 1 - beta1 ** step
        bias2 = 1 - beta2 ** step
        denom = (exp_avg_sq / bias2).sqrt() + eps
        p.add_(exp_avg / denom, alpha=-lr / bias1)
        return p

    def nesterov_momentum(momentum_buffer, grads, momentum):
        momentum_buffer.lerp_(grads, 1 - momentum)
        return torch.lerp(grads, momentum_buffer, momentum)

## Summary

Plain gradient descent fails because parameters need very different step sizes. AdamW fixes
that per number: smooth the gradient into `m` (momentum), divide by `sqrt(v)` so every
parameter steps about `lr` regardless of its gradient's typical size (normalisation), correct
both averages for their zero start (bias correction), and decay weights separately from the
gradient (decoupled weight decay) - tracing one parameter (`p = 1.0`, `grad = 0.6`) through all
four gave `p = 0.89`.

Muon fixes the analogous problem per matrix, not per number: it never touches `W` directly,
only the gradient `G` that `backward()` produces for it. Nesterov momentum smooths `G` into a
`direction`; five Newton-Schulz passes replace that `direction` with the nearest matrix whose
*singular values* are all close to 1 - preserving which directions matter (`U`, `V`) while
equalising how strongly each one is updated (`Sigma`) - using only matrix multiplies instead of
an SVD; NorMuon and cautious weight decay refine it further; and only then does `W -= lr *
(that finished update)` touch the actual weights. `setup_optimizer` decides who gets which
optimizer, at learning rates scaled for the model's width. Next: the schedules that move those
learning rates over a run, and how a big batch is faked on a small machine.
