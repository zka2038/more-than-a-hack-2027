"""Score a possession method on external matches with ATD-like degradation.

    python -m solution.bench.evaluate --method stub
    python -m solution.bench.evaluate --method model --model solution/models/possession.pt

Per match: spellF1 (the ranking metric), macroF1, predicted vs true possession count, using the
official scorer (evaluation/possession_metrics.py). The two HOLDOUT matches are never used for
training or tuning and are only scored with --holdout, once, at the end (PLAN.md section 6).
"""

import argparse
import os
import sys
import time
import zlib
from multiprocessing import get_context

import numpy as np

from solution.pipeline import contracts as C
from solution.bench import external
from solution.bench.degrade import DEFAULTS, degrade, missing_rate, sample_params

sys.path.insert(0, os.path.join(C.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

HOLDOUT = ("DFL-J03WR9", "DFL-J03WQQ")
BENCH = os.path.join(external.CACHE, "bench")
DEGRADE_VERSION = 1
S0_KEYS = tuple(C.NPZ_SCHEMAS["s0_tracking"])


def dev_matches():
    return [m for m in external.match_names() if m not in HOLDOUT]


# Possession convention of the reference labels. "controlled" folds DFL's first-touch spells
# shorter than DFL_CONTROL_FRAMES back into the surrounding possession; 25 frames makes DFL's
# possession counts match the Metrica games' (an external-to-external calibration).
CONVENTION = os.environ.get("MTAH_CONVENTION", "controlled")
DFL_CONTROL_FRAMES = 25


def reference_labels(name, gt):
    if CONVENTION == "controlled" and name.startswith("DFL-"):
        return external.controlled(gt, DFL_CONTROL_FRAMES)
    return gt


def degraded(name, seed=0, randomize=False):
    """(s0, gt) for one external match under one degradation draw, cached on disk."""
    os.makedirs(BENCH, exist_ok=True)
    tag = "r" if randomize else "d"
    path = os.path.join(BENCH, "{}_{}{}_v{}.npz".format(name, tag, seed, DEGRADE_VERSION))
    if os.path.exists(path):
        with np.load(path) as data:
            return {k: data[k] for k in S0_KEYS}, data["gt"]
    s0, gt, info = external.load(name)
    mix = seed * 1000 + zlib.crc32(name.encode()) % 1000
    params = sample_params(np.random.default_rng(mix)) if randomize else dict(DEFAULTS)
    d = degrade(s0, gt, info, seed=mix, params=params)
    np.savez_compressed(path, gt=gt, **d)
    return d, gt


def clean(name):
    s0, gt, _ = external.load(name)
    return s0, gt


# -- methods -------------------------------------------------------------------------------------

def predict_stub(s0):
    """The four stub stages, in memory: the published baseline re-expressed (PLAN.md section 1)."""
    from solution.pipeline.s1_inplay import run as s1m
    from solution.pipeline.s2_ball import run as s2m
    from solution.pipeline.s3_touch import run as s3m
    from solution.pipeline.s4_possession import run as s4m
    p_alive = s1m.decide(s0)
    alive = s1m.to_intervals(p_alive, s0["half"])
    s1 = {"p_alive": p_alive, "alive": alive}
    s2 = s2m.track(s0, s1)
    s3, _ = s3m.detect(s0, s1, s2)
    label, _ = s4m.decode(s0["half"], s1, s3)
    return label


def predict_pipeline(s0, model_path):
    """The real stages S1-S4 with the learned model, in memory (what run_all does on BAR-ATH)."""
    from solution.pipeline.s1_inplay import run as s1m
    from solution.pipeline.s2_ball import run as s2m
    from solution.pipeline.s3_touch import run as s3m
    from solution.pipeline.s4_possession import run as s4m
    p_alive, alive, _, _ = s1m.decide_model(s0, model_path)
    s1 = {"p_alive": p_alive, "alive": alive}
    s2 = s2m.track(s0, s1)
    s3, _ = s3m.detect(s0, s1, s2)
    s3["touch_ll"], s3["control"] = s3m.model_evidence(s0, model_path)
    label, _ = s4m.decode_model(s0["half"], s1, s3, model_path=model_path)
    return label


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "alive_dead": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"]}


def _job(args):
    name, method, kwargs = args
    s0, gt = clean(name) if kwargs.get("clean") else degraded(name, kwargs.get("seed", 0))
    t = time.time()
    if method == "stub":
        pred = predict_stub(s0)
    elif method == "model":
        from solution.model import infer
        pred = infer.predict(s0, kwargs["model"], params=kwargs.get("decoder"))
    elif method == "pipeline":
        pred = predict_pipeline(s0, kwargs["model"])
    else:
        raise KeyError(method)
    out = score(reference_labels(name, gt), pred)
    out.update(name=name, seconds=time.time() - t,
               missing=missing_rate(s0, gt))
    return out


def run(names, method, workers=4, **kwargs):
    jobs = [(n, method, kwargs) for n in names]
    if workers <= 1:
        results = [_job(j) for j in jobs]
    else:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            results = pool.map(_job, jobs)
    return results


def report(results, title=""):
    print("\n{}".format(title))
    print("{:12s} {:>7s} {:>7s} {:>6s} {:>6s} {:>7s} {:>7s} {:>6s}".format(
        "match", "spellF1", "macroF1", "n_pred", "n_gt", "aliveF1", "teamAcc", "miss"))
    for r in results:
        print("{:12s} {:7.4f} {:7.4f} {:6d} {:6d} {:7.4f} {:7.4f} {:5.1f}%".format(
            r["name"], r["spell"], r["macro"], r["n_pred"], r["n_gt"], r["alive_dead"],
            r["team_acc"], 100 * r["missing"]))
    mean = {k: float(np.mean([r[k] for r in results])) for k in ("spell", "macro", "alive_dead",
                                                                  "team_acc")}
    print("{:12s} {:7.4f} {:7.4f} {:>6s} {:>6s} {:7.4f} {:7.4f}".format(
        "MEAN", mean["spell"], mean["macro"], "", "", mean["alive_dead"], mean["team_acc"]))
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="stub", choices=("stub", "model", "pipeline"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--holdout", action="store_true", help="score the held-out matches")
    parser.add_argument("--clean", action="store_true", help="no degradation (upper bound)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    names = args.matches or (list(HOLDOUT) if args.holdout else dev_matches())
    results = run(names, args.method, workers=args.workers, model=args.model, clean=args.clean)
    report(results, "{} on {} matches ({})".format(args.method, len(names),
                                                  "clean" if args.clean else "degraded"))


if __name__ == "__main__":
    main()
