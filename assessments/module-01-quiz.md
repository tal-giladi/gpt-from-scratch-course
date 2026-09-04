# Module 01 quiz - From text to tensors

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** A colleague proposes raising `VOCAB_SIZE` from 8,192 to 32,768 "so the model
understands more words." Name two concrete things that get *more expensive*, and one number
that will improve without the model being any better.

**2.** A row has capacity `T + 1 = 257`. The document buffer holds documents of length
`[300, 180, 90, 40, 12]`. Walk the packer through building one row: which documents are used,
in what order, and is anything cropped?

**3.** Why does `prepare.py` prepend BOS to every *document* rather than to every *row*?
What would break if it did the latter?

**4.** Two models are trained on the same text. Model A uses a 8k vocabulary and reports a
final loss of 3.1 nats/token. Model B uses a 64k vocabulary and reports 3.4 nats/token. Which
is the better model, and what would you need to decide?

**5.** You start a training run and the first logged loss is 6.2, not 9.01. Give two distinct
explanations, one that is a bug and one that is not.

**6.** The dataloader crops the *shortest* document in the buffer when nothing fits, and
throws the remainder away. Why the shortest? And name one thing you would measure before
changing this to carry the remainder over to the next row.

---

## Answers

**1.** More expensive: (a) the `lm_head` matrix grows from `8192 x n_embd` to
`32768 x n_embd`, and it is already about half the FLOPs of a small model - so every token
costs materially more; (b) `wte` and every value-embedding table grow by the same factor,
raising the model's parameter floor (lesson 15's binding constraint). Improving for free:
loss *per token* falls, because each token now covers more bytes - which is precisely why
`val_bpb` is measured per byte instead.

**2.** `remaining = 257`: the longest that fits is 180. `remaining = 77`: 300 and 180 are
gone or too big, so 40 fits (90 does not). `remaining = 37`: 12 fits. `remaining = 25`:
nothing fits (90 and 300 remain, both too long), so the **shortest** remaining document -
90 - is cropped to its first 25 tokens and the row is full. Order: 180, 40, 12, then 25
tokens of the 90. The 300 is untouched and stays in the buffer.

**3.** Because a row holds several documents. Prepending per document means every document
boundary inside a row is marked, not just the first one - which is the signal the model needs
("the previous text is over, start again"). Marking only the row would leave every internal
boundary invisible, and the model would spend capacity learning that text sometimes changes
topic mid-sentence for no reason.

**4.** You cannot tell from those numbers, and that is the point. Convert both to bits per
byte: `nats/token ÷ ln(2) ÷ bytes-per-token`. Model B's tokens are fatter, so its higher
per-token loss may still be a lower per-byte score. You need each model's average bytes per
token - or better, just have both report `val_bpb`.

**5.** Not a bug: your logged value is an EMA that has already been updated several times,
or the model was resumed from a checkpoint rather than initialised. A bug: the loss is being
computed over something smaller than the full vocabulary (a masking or reshaping error), or
the targets are not actually shifted - if `y` equals `x`, the model can score well by copying
its input, and the loss collapses immediately.

**6.** Cropping the shortest loses the fewest tokens - the shortest document's tail is the
smallest thing you can throw away. Before carrying remainders over, measure how many tokens
are actually lost: at `T = 2048` with a well-filled buffer it is a small fraction of a
percent, and the change adds state to a loop whose simplicity is the reason it is easy to
reason about. Measure the loss rate first; a fix for a 0.05% problem is not worth new state.
