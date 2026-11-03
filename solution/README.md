**Submission pipeline**

This small scaffold works before the match data arrives. It writes 199,319 rows
with the exact header `frame,label`, then runs the official validator on the saved
file. The smoke test writes X (abstain) for every frame. It checks formatting only.

Run from the repository root using the existing environment:

```bash
.venv/bin/python -m solution.submission --smoke-test
```

The output is `outputs/smoke_test.csv`. Generated outputs are ignored by Git.
To regenerate that file, explicitly allow replacement:

```bash
.venv/bin/python -m solution.submission --smoke-test --overwrite
```

Use `--output outputs/another_name.csv` to choose a different destination.
The command exits with a nonzero status if writing or official validation fails.

On a fresh setup, create a Python environment and install the existing
`requirements.txt`; the pipeline adds no dependencies.

**Connecting a model later**

Supply exactly one predicted label for every tactical-video frame, in order:

```python
from solution.submission import write_submission, run_official_validator

# predicted_labels comes from our algorithm and has exactly 199,319 entries.
path = write_submission(predicted_labels, "outputs/team_submission.csv")
run_official_validator(path)
```

The writer uses the official validation function before opening the output,
rejects missing/extra rows and invalid or blank labels, and never pads predictions.
The CLI then runs the official validation script on the exact saved CSV.
Existing files are preserved unless `overwrite=True` is passed to the writer.

Run the contract checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Keep developing in `solution/` in this checkout. The official `evaluation/` code
and `games.py` remain the reference. A team-owned public fork can host the final
submission code later; no separate repository is needed for this scaffold.
