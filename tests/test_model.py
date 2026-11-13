"""The learned-model path of the pipeline, on the synthetic match. No real data needed.

    pytest tests/test_model.py

Checks wiring and contracts only (the synthetic match is not football, so no accuracy bar):
the stages accept the model's evidence, the submission validates, X sits exactly outside the
halves, and two runs give byte-identical files (PLAN.md, Hard rule 7).
"""

import filecmp
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solution.pipeline import contracts as C  # noqa: E402
from solution.pipeline import fixtures, run_all  # noqa: E402
from solution.model import infer  # noqa: E402

MODEL = infer.DEFAULT_MODEL
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL), reason="no trained model committed")


def _run(tmp, tag):
    data_dir, stage_dir = os.path.join(tmp, "data"), os.path.join(tmp, "stages_" + tag)
    truth = fixtures.write_bundle(data_dir)
    out = run_all.run(stage_dir, os.path.join(tmp, tag + ".csv"), data_dir=data_dir,
                      n_frames=truth["n_frames"], blocks=truth["blocks"], model_path=MODEL)
    return truth, stage_dir, out


def test_model_pipeline_is_valid_and_reproducible():
    tmp = tempfile.mkdtemp(prefix="model_test_")
    try:
        truth, stage_dir, out_a = _run(tmp, "a")
        _, _, out_b = _run(tmp, "b")
        assert filecmp.cmp(out_a, out_b, shallow=False), "two runs differ"

        n = truth["n_frames"]
        s0 = C.load("s0_tracking", stage_dir, n=n)
        s1 = C.load("s1_inplay", stage_dir, n=n)
        s3 = C.load("s3_touch", stage_dir, n=n)
        s4 = C.load("s4_possession", stage_dir, n=n)
        assert s1["meta"]["params"].get("model") and s3["meta"]["params"].get("model")
        assert s4["meta"]["params"].get("model")
        label = s4["label"]
        assert np.array_equal(label == C.X, s0["half"] == 0)
        assert np.isin(label[s0["half"] > 0], (C.H, C.A, C.D)).all()
        assert np.array_equal(label == C.D, s1["alive"] == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
