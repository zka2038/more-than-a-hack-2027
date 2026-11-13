"""Contract and wiring checks for solution/pipeline, on a synthetic match. No real data needed.

    python tests/test_pipeline.py        # or: pytest tests/test_pipeline.py
"""

import atexit
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# These checks pin the stub rules' behaviour; the learned model has its own test below.
os.environ["MTAH_MODEL"] = "none"
from solution.pipeline import contracts as C  # noqa: E402
from solution.pipeline import fixtures, run_all  # noqa: E402

_CACHE = {}


def pipeline():
    """Run the stub pipeline once on a synthetic bundle; reuse the result across tests."""
    if not _CACHE:
        tmp = tempfile.mkdtemp(prefix="pipeline_test_")
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)
        data_dir, stage_dir = os.path.join(tmp, "data"), os.path.join(tmp, "stages")
        truth = fixtures.write_bundle(data_dir)
        out = run_all.run(stage_dir, os.path.join(tmp, "submission.csv"), data_dir=data_dir,
                          n_frames=truth["n_frames"], blocks=truth["blocks"])
        _CACHE.update(truth=truth, stage_dir=stage_dir, out=out)
    return _CACHE


def rejects(function, *args):
    try:
        function(*args)
    except C.ContractError:
        return True
    return False


def test_s0_reads_the_bundle():
    run = pipeline()
    truth, n = run["truth"], run["truth"]["n_frames"]
    s0 = C.load("s0_tracking", run["stage_dir"], n=n)

    # Positions come back in metres, in <Player> order on both sides of the column re-ordering.
    assert np.allclose(s0["player_xy"], truth["seen_players"], atol=1e-3, equal_nan=True)
    assert np.allclose(s0["ball_xy"], truth["seen_ball"], atol=1e-3, equal_nan=True)
    assert np.array_equal(s0["player_team"], truth["team"])
    assert np.nanmax(s0["player_speed"]) == 1.5

    for number, first, last in truth["blocks"]:
        assert (s0["half"][first:last + 1] == number).all()
    assert (s0["half"] > 0).sum() == sum(last - first + 1 for _, first, last in truth["blocks"])

    # The fixture's homography is a plain scaling, so the image centre is the centre spot.
    assert np.allclose(s0["cam_xy"][1], (C.PITCH[0] / 2, C.PITCH[1] / 2), atol=1e-3)
    assert np.isnan(s0["cam_xy"][0]).all() and np.isnan(s0["cam_xy"][500]).all()

    tags = C.load_table("s0_tags", run["stage_dir"], n=n)
    assert tags["team"].tolist() == [C.HOME, C.AWAY, C.UNKNOWN, C.HOME]   # "Atletic Club" -> A
    assert tags["half"].tolist() == [1, 1, 1, 2]
    assert tags["start_frame"].tolist() == sorted(tags["start_frame"].tolist())


def test_stage_files_honour_their_contracts():
    run = pipeline()
    n, stage_dir = run["truth"]["n_frames"], run["stage_dir"]
    s0 = C.load("s0_tracking", stage_dir, n=n)
    s1 = C.load("s1_inplay", stage_dir, n=n)
    s2 = C.load("s2_ball", stage_dir, n=n)
    s3 = C.load("s3_touch", stage_dir, n=n)
    s4 = C.load("s4_possession", stage_dir, n=n)
    stoppages = C.load_table("s1_stoppages", stage_dir, n=n)
    touches = C.load_table("s3_touches", stage_dir, n=n)

    C.check_s1_against_s0(s1, s0)
    C.check_stoppages_against_s1(stoppages, s1)
    C.check_s2_against_s1(s2, s1)
    C.check_s4_against_s0(s4, s0)
    assert s1["meta"]["stage"] == "s1_inplay" and s1["meta"]["params"]["stub"] is True
    assert len(stoppages) >= 2 and len(touches) >= 10
    assert (s2["ball_src"] == C.SRC_INTERPOLATED).any()
    # A last touch never leaks across a stoppage: unknown again at the start of each segment
    # unless a player is on the ball on that very frame.
    for start, _ in C.alive_segments(s1["alive"]):
        assert s3["frames_since_touch"][start] in (-1, 0)


def test_submission_is_valid_and_not_backwards():
    run = pipeline()
    truth = run["truth"]
    table = run_all.pm.read_label_csv(run["out"])
    assert run_all.pm.validate_submission(table, truth["n_frames"]) is None
    pred = run_all.pm.label_codes(table)
    gt = truth["label"]

    assert np.array_equal(pred == C.X, gt == C.X)       # X exactly outside the halves
    assert {C.H, C.A, C.D} <= set(pred.tolist())
    both_alive = (gt < C.D) & (pred < C.D)
    # The fixture is easy: a stub with home and away swapped, or dead and alive swapped, fails.
    assert (pred[both_alive] == gt[both_alive]).mean() > 0.8
    assert ((pred == C.D) & (gt == C.D)).sum() > 0.5 * (gt == C.D).sum()
    assert run_all.pm.evaluate(gt, pred)["macro_f1"] > 0.6


def test_contracts_reject_bad_files():
    run = pipeline()
    n, stage_dir = run["truth"]["n_frames"], run["stage_dir"]
    s0 = C.load("s0_tracking", stage_dir)
    s1 = C.load("s1_inplay", stage_dir)
    s2 = C.load("s2_ball", stage_dir)
    s4 = C.load("s4_possession", stage_dir)
    for payload in (s0, s1, s2, s4):
        payload.pop("meta")

    assert rejects(C.validate, "s1_inplay", s1, n + 1)                           # wrong length
    assert rejects(C.validate, "s1_inplay", dict(s1, alive=s1["alive"].astype(np.int32)))
    assert rejects(C.validate, "s1_inplay", {"alive": s1["alive"]})              # missing array
    assert rejects(C.validate, "s1_inplay", dict(s1, extra=s1["alive"]))         # unknown array
    assert rejects(C.validate, "s2_ball", dict(s2, ball_src=np.zeros_like(s2["ball_src"])))

    # An estimated ball inside a stoppage breaks the S2 guarantee.
    dead = int(np.flatnonzero(s1["alive"] == 0)[0])
    src = s2["ball_src"].copy()
    src[dead] = C.SRC_INTERPOLATED
    assert rejects(C.check_s2_against_s1, dict(s2, ball_src=src), s1)

    # Abstaining on a graded frame is never allowed out of S4.
    label = s4["label"].copy()
    label[int(np.flatnonzero(s0["half"] > 0)[0])] = C.X
    assert rejects(C.check_s4_against_s0, dict(s4, label=label), s0)

    stoppages = C.load_table("s1_stoppages", stage_dir)
    overlap = pd.concat([stoppages.iloc[:1], stoppages.iloc[:1].assign(
        start_frame=stoppages["start_frame"].iloc[0] + 1)])
    assert rejects(C.validate_table, "s1_stoppages", overlap)
    assert rejects(C.validate_table, "s1_stoppages", stoppages.assign(cause="handball"))
    assert rejects(C.validate_table, "s1_stoppages", stoppages.drop(columns=["restart_team"]))


def test_runs_helper():
    assert C.runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert C.runs(np.zeros(4, dtype=bool)) == []
    assert C.alive_segments(np.array([-1, 1, 1, 0, 1, -1], dtype=np.int8)) == [(1, 3), (4, 5)]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  {}".format(test.__name__))
    print("{} passed".format(len(tests)))
