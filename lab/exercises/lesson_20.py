# Lesson 20 - Capstone: one honest experiment
# Read lessons/module-07/lesson-03.md, run the three experiments, then fill this in.
#
#   bash lab/lab.sh shell
#   python tools/run_experiment.py baseline
#   python tools/run_experiment.py baseline2
#   python tools/run_experiment.py variant AR_DEPTH=2


def capstone_report() -> dict:
    """Your experiment, in seven fields.

        hypothesis   what you tested and why, in a sentence you wrote BEFORE
                     looking at the variant's result
        changed      the single AR_* knob you overrode, e.g. "AR_DEPTH"
        expectation  "lower" | "higher" | "no change" - what you predicted
                     val_bpb would do
        baseline     name of the baseline run's log (without .log)
        repeat       name of the second, identical run - your noise floor
        variant      name of the changed run
        conclusion   "confirmed" | "refuted" | "inconclusive", judged against
                     your own measured noise floor
    """
    # TODO: replace every value with your own experiment's.
    raise NotImplementedError
