"""Guard for PLAN.md's Hard rules. Run before every merge to main:  pytest tests/test_rules.py

1. Nothing under solution/ may reference the BAR-ATH ground truth (public or private) or the
   scorer's --gt switch: ground truth never trains, tunes or selects anything.
2. git must not track data-like files except the organiser-committed ones and the allow-listed,
   externally trained model weights.
3. Every committed model checkpoint must declare that it was trained on external matches only.
"""

import fnmatch
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = [
    re.compile(r"evaluation[\\/]+gt", re.I),
    re.compile(r"possession_gt", re.I),
    re.compile(r"""["']--gt["']"""),
    re.compile(r"BAR-ATH_possession", re.I),
]
DATA_PATTERNS = ("*.npz", "*.npy", "*.pkl", "*.pickle", "*.parquet", "*.csv", "*.pt", "*.pth",
                 "*.h5", "*.mp4", "*.xml", "*.txt")
ALLOWED = (
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",   # committed by the organisers
    "evaluation/results/baseline_*.csv",                # organiser baselines, ATD only
    "solution/models/*.pt",                             # external-trained weights (check 3)
    "requirements*.txt",
    "CITATION.cff",
)


def _tracked():
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT).decode()
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_solution_never_references_ground_truth():
    offenders = []
    for folder, _, files in os.walk(os.path.join(ROOT, "solution")):
        for name in files:
            if not name.endswith((".py", ".json", ".md", ".txt", ".cfg", ".toml", ".yaml")):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    offenders.append("{}: {}".format(os.path.relpath(path, ROOT), pattern.pattern))
    assert not offenders, "solution/ references the ground truth:\n" + "\n".join(offenders)


def test_no_tracked_data_files():
    bad = [p for p in _tracked()
           if any(fnmatch.fnmatch(os.path.basename(p), pat) for pat in DATA_PATTERNS)
           and not any(fnmatch.fnmatch(p, allowed) for allowed in ALLOWED)]
    assert not bad, "data-like files are tracked by git:\n" + "\n".join(bad)


def test_committed_models_are_external_only():
    models = [p for p in _tracked() if fnmatch.fnmatch(p, "solution/models/*.pt")]
    if not models:
        return
    import torch
    for rel in models:
        ck = torch.load(os.path.join(ROOT, rel), map_location="cpu", weights_only=False)
        names = ck.get("train_matches") or []
        assert names, "{} does not record its training matches".format(rel)
        assert all(n.startswith(("DFL-", "MET-")) for n in names), \
            "{} was trained on something other than external matches: {}".format(rel, names)
        assert ck.get("data_sources"), "{} does not record its data sources".format(rel)


if __name__ == "__main__":
    test_solution_never_references_ground_truth()
    test_no_tracked_data_files()
    test_committed_models_are_external_only()
    print("rules guard: OK")
