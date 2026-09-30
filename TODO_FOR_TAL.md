# TODO for Tal

## State (2026-09-15)

## Lesson 11 (AdamW + Muon) rewrite and general quiz (2026-09-23)

- Lesson 11 rewritten per your remarks: worked one-parameter AdamW trace, and a full
  x -> W -> prediction -> loss -> gradient -> Muon -> W-update pipeline for Muon, with an SVD
  vs "entries = 1" section and a Newton-Schulz line-by-line walkthrough. All numbers verified
  in the lab container. Committed and pushed (`d8e4ff3`).
- Module 04 quiz gets a 7th question on the "Muon sets W's singular values to 1" misconception,
  plus small additions to Q3/Q4.
- New file `assessments/general-gpt-quiz.csv`: 49 self-contained general GPT/transformer
  theory + arithmetic questions (not tied to this course's fork), sent to you and committed.

- Lessons 08-17 rewritten for clarity (08 revised again from your remarks), thread fix, quiz fixes:
  committed and pushed.
- Autoresearch fork: thread fix `04d613e` pushed to `fork/main` (a stray `master` branch created by
  mistake during the push was deleted).
- Lessons 18-20 done and pushed (`e180d3e`). Added `bash lab/lab.sh experiment NAME [AR_X=value]`:
  lesson 20 used to type `python tools/run_experiment.py` into the Python prompt `lab.sh shell` opens.
- `lab/exercises/lesson_20.py` header comment now shows the `lab.sh experiment` commands (`72bf6d3`).

## Done: CPU training ~8x slower than necessary (issue 1)

- PyTorch's default 16 threads -> pinned `OMP_NUM_THREADS: "8"` in both `docker-compose.yml` files.
- `AR_PEAK_GFLOPS` 30 -> 150 (compose + train.py default), re-measured at 8 threads.
- Two 2-minute capstone-protocol runs: 97 / 102 steps (was 16), val_bpb 2.2824 / 2.2756 (was
  2.6074). Logs in `assets/logs/`.
- Lessons 09, 12, 13, 14, 15, 16 and `README_CPU.md` re-quote the new numbers.
- Stale capstone logs from building the course (16 threads) deleted from `lab/capstone/runs/`.
- Cosmetic, not changed: `lab/checks/14.py` and `lab/checks/15.py` still say "~300 tok/s measured"
  in a comment/message (the checks themselves still pass).

## Left open by decision

2. **`sync()` in `autoresearch/train.py` calls itself forever on CUDA.** Left alone: no GPU. Tal will
   report it if he ever hits it.

## Done: quiz answers (issue 3)

- Modules 04-06 quiz answers aligned with the corrected lessons (commit after `0cc6530`).
