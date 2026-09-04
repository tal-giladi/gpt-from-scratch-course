# Module 07 quiz - Research

Six questions. Answers and explanations at the bottom - try all six first.

---

**1.** `program.md` lets the agent change anything in `train.py` and nothing in
`prepare.py`. Give two changes an agent could make to `prepare.py` that would lower `val_bpb`
without improving the model at all.

**2.** Why does the loop redirect training output to a file and read it with `grep` rather
than letting it print? Answer in terms of a resource.

**3.** You run 60 experiments overnight, keeping each one that lowers `val_bpb`. Twelve are
kept. Explain why fewer than twelve are real, and describe the cheapest defence.

**4.** Two runs: A is depth 4 with `TIME_BUDGET=600` scoring 2.399; B is depth 6 with
`TIME_BUDGET=900` scoring 2.310. Write the one-sentence rejection, and say what B would need
to be re-run with.

**5.** Your measured noise floor is 0.005 bpb. A change comes in at 0.004 better. What do you
report, and what would you have to do to turn it into a result?

**6.** The `NEVER STOP` instruction tells the agent not to ask whether it should continue.
Give the reason it is there, and one failure mode it creates that a human should check for in
the morning.

---

## Answers

**1.** (a) Reduce `EVAL_TOKENS`, or point `VAL_SHARD` at a shard that is also in the training
set - the second is straightforward data leakage and would produce a spectacular, meaningless
improvement. (b) Change the tokenizer or `MAX_SEQ_LEN`: a larger vocabulary lowers loss per
token, and a different eval context length changes the score without changing the model. Both
are "improvements" to the ruler. This is why the file is fenced off rather than merely
discouraged.

**2.** Context. The agent's context window is a fixed budget, and a training run emits
hundreds of progress lines; letting them into the transcript would evict the reasoning about
what is being tested and why. `grep "^val_bpb:"` extracts one line. Treating context as a
managed resource - like GPU memory - is a real part of building agent loops.

**3.** Keeping a change if it improves the metric is a statistical test, and 60 of them were
run without any correction. With a noise floor of a few hundredths and improvements of a
similar size, several "wins" are noise that got promoted into the baseline - and because the
branch advances on each keep, those promotions compound. Cheapest defence: at the end, re-run
the surviving changes against the original baseline under the same protocol. A real effect
reproduces; a noise win does not.

**4.** "B trained 50% longer, so this is an experiment about time in which depth also
changed - it says nothing about depth." B needs to be re-run at `TIME_BUDGET=600`, identical
in every other setting, and ideally at a second seed if the result matters.

**5.** Report it as **inconclusive**: the difference is inside your own measurement noise, so
you cannot distinguish it from a coin flip. To turn it into a result you need to reduce the
noise or increase the signal - run both arms several times and compare distributions rather
than single numbers, or lengthen the runs so a single extra step matters less. Reporting an
honest "inconclusive" is a result; reporting 0.004 as a win is not.

**6.** It is there because the whole value proposition is unattended operation - a human
leaves it running overnight and wakes to a hundred experiments, and an agent that pauses for
approval every hour delivers one experiment. The failure mode: with nobody to stop it, the
agent can spend hours hill-climbing inside the noise floor, or accumulate a series of small
"wins" that are individually unverified and collectively unreadable. The morning check is
`results.tsv` plus a re-run of the survivors - and a read of the diff, because the simplicity
criterion is the one rule no metric enforces.
