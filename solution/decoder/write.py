"""Write a full-length submission and check it with the official validator before returning."""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])


def to_labels(n_frames, halves, codes_per_half):
    """Letters for every frame: decoded codes inside the halves, X everywhere else."""
    out = np.full(n_frames, "X", dtype="<U1")
    for half, codes in zip(halves, codes_per_half):
        out[half.start: half.start + len(codes)] = LETTERS[np.asarray(codes)]
    return out


def write_submission(labels, path, n_expected):
    import pandas as pd
    frame = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    reason = pm.validate_submission(frame, n_expected)
    if reason:
        raise ValueError("refusing to write an invalid submission: " + reason)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def run_official_validator(path):
    script = os.path.join(ROOT, "evaluation", "validate_submission.py")
    subprocess.run([sys.executable, script, path], check=True)
