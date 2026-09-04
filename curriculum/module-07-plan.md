# Module 07 - Research

**Lessons 18-20.** The loop the repo was built for, the discipline that makes its results
mean anything, and one experiment of your own.

## Learning objectives

After this module a learner can:

1. Explain `program.md` as an agent contract: what is frozen, what is editable, and why each
   boundary is where it is.
2. Explain why a fixed *time* budget - not a fixed step count - is what makes a speed change
   and a modelling change comparable.
3. Parse a run log into a result record, handling the completed, crashed and diverged cases,
   and explain why a crash is a result rather than an error.
4. Measure a noise floor by running the same code twice, and state the smallest improvement
   they are entitled to believe.
5. Judge whether two runs can be compared at all: one variable, equal budgets, the same
   ruler, a stated seed, a direction declared in advance.
6. Explain why 100 experiments kept-if-better is 100 uncorrected tests, and what the cheap
   defence is.
7. Run a controlled experiment end to end and write a conclusion their own grader will
   reject if the data does not support it.

## Dependencies

All previous modules. The capstone runs the real training script and reads the metric from
Module 05.

## Misconceptions this module is written to break

- *"The agent does research."* It hill-climbs one scalar over one file. That is a real and
  useful thing, and it is not the same thing.
- *"An improvement is an improvement."* Not below your noise floor, and not when a second
  variable moved.
- *"A longer run is a fairer comparison."* It is a different experiment. Equal budgets or no
  comparison.
- *"Being wrong is a failed experiment."* A refuted hypothesis with clean methodology is a
  result. An unfalsifiable claim is not.
- *"CPU results are just slower GPU results."* At three-minute runs with a noise floor of a
  few hundredths, most real effects are invisible. The mechanism is real; the conclusions
  are not transferable.

## Exercises

| # | Function | What it proves |
|---|---|---|
| 18 | `parse_summary` | you can turn any of the three run outcomes into a record, ignore unknown fields, and not be fooled by colons in progress lines |
| 19 | `compare_runs` | you can reject the seven ways a comparison goes wrong, and call a within-noise difference a tie |
| 20 | `capstone_report` | you ran three real trainings, measured your own noise floor, and wrote a conclusion consistent with the data |

## Time

About three hours, of which roughly ten minutes is the three capstone training runs.
