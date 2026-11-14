# Runbook: producing the BAR-ATH submission

The match data never goes to a cloud AI (PLAN.md, Hard rule 5). Run these commands **yourself, in
your own terminal**: not through a coding agent's shell, and not with an in-chat `!` command. When
you report back, share only what the last step of each block says you may.

## 0. One-time setup

```powershell
# Keep the data outside OneDrive or other synced folders.
$env:MTAH_DATA_DIR = "C:\mtah\data"      # holds tactical\BAR-ATH_tactical_atd.xml etc.
C:\mtah\venv\Scripts\activate             # or: python -m venv .venv; pip install -r requirements-solution.txt
cd <repo>
```

## 1. Make the submission

```powershell
python -m solution.pipeline.run_all --stages C:\mtah\stages --out evaluation\results\<team_name>.csv
python evaluation\validate_submission.py evaluation\results\<team_name>.csv
```

- **Takes:** a few minutes on CPU. S1 and S3 each run the model once.
- **May share:** the validator's pass/fail line, the S1 line listing the detected kick-off frames, and
  the S4 line (possession count and mean duration).

## 2. Correctness check on the public window (once per frozen candidate)

```powershell
python evaluation\run_evaluation.py
```

This checks alignment only: labels the right way round, and a score well above `baseline_smooth50`.
**Never choose between versions with it, never tune on it** (Hard rule 1). If it shows a bug (H/A
swapped, an off-by-one, halves wrong), fix the bug.

- **May share:** the board's `spellF1` and `macroF1` for our row.

## 3. Domain check: does the simulated noise look like the real ATD?

```powershell
$env:MTAH_CACHE = "C:\mtah\cache"   # where the external benchmark lives
python -m solution.bench.domain_check --stages C:\mtah\stages
```

This prints input statistics only (ball gaps, visible players, speeds), with no labels and no
per-frame values.

- **May share:** the table. It tells us whether the external benchmark is optimistic and how to fix
  the degradation.

## 4. Reproducibility (before submitting)

Run step 1 twice from a clean clone into two files, then check they are identical:

```powershell
certutil -hashfile a.csv SHA256
certutil -hashfile b.csv SHA256     # must match
```
