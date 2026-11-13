"""Tune the decoder on out-of-fold evidence from external matches, on spellF1 directly.

    python -m solution.model.tune --run cv1 [--weights MET=2]

The objective is the mean spellF1 over the dev matches (optionally weighting a source), computed
with the official scorer's own functions. Never run on BAR-ATH (PLAN.md, Hard rule 1).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from solution.pipeline import contracts as C
from solution.model import decode as Dc
from solution.model.train import OOF_DIR

sys.path.insert(0, os.path.join(C.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

SEARCH = {
    "temp": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "c_sw": [1, 2, 4, 6, 8, 12, 16, 24],
    "c_in": [1, 2, 4, 6, 8, 12, 16, 24],
    "c_out": [1, 2, 4, 6, 8, 12, 16, 24],
    "beta": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "prior": [0.0, 0.5, 1.0],
}


def load_oof(run, convention=None):
    """Out-of-fold evidence; the reference labels follow the current convention (see
    bench.evaluate.reference_labels), so runs trained under different conventions compare fairly."""
    from solution.bench import evaluate as EV
    data = []
    for path in sorted(glob.glob(os.path.join(OOF_DIR, run, "*.npz"))):
        name = os.path.basename(path)[:-4]
        with np.load(path) as d:
            gt = d["gt"]
            if convention or EV.CONVENTION:
                raw = EV.degraded(name, 0)[1]          # labels as the source defines them
                gt = EV.reference_labels(name, raw)
            data.append({"name": name, "logp": d["logp"], "change": d["change"], "gt": gt,
                         "segments": d["segments"], "log_prior": d["log_prior"]})
    return data


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    return pm.spell_scores(gt, filled, runs)["mean_f1"]


def decode_match(m, params):
    pred = np.full(len(m["gt"]), C.X, dtype=np.int8)
    for s, e in m["segments"]:
        pred[s:e] = Dc.decode_segment(m["logp"][s:e], m["change"][s:e], params, m["log_prior"])
    return pred


def objective(data, params, weights=None):
    scores = {m["name"]: spell_f1(m["gt"], decode_match(m, params)) for m in data}
    w = {n: (weights or {}).get(n.split("-")[0], 1.0) for n in scores}
    return sum(scores[n] * w[n] for n in scores) / sum(w.values()), scores


def coordinate_search(data, start=None, rounds=3, weights=None, verbose=True):
    best = dict(Dc.DEFAULTS, **(start or {}))
    best_score, _ = objective(data, best, weights)
    if verbose:
        print("start {:.4f} {}".format(best_score, best), flush=True)
    for r in range(rounds):
        improved = False
        for key, grid in SEARCH.items():
            for value in grid:
                if value == best[key]:
                    continue
                trial = dict(best, **{key: value})
                score, _ = objective(data, trial, weights)
                if score > best_score + 1e-5:
                    best, best_score, improved = trial, score, True
            if verbose:
                print("round {} {:6s} -> {:.4f} {}".format(r, key, best_score, best), flush=True)
        if not improved:
            break
    return best, best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="cv1")
    parser.add_argument("--weights", nargs="*", default=[], help="SOURCE=weight, e.g. MET=2")
    parser.add_argument("--out", default=None, help="write the tuned params as JSON")
    args = parser.parse_args()
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights)}
    data = load_oof(args.run)
    print("matches:", [m["name"] for m in data])
    best, score = coordinate_search(data, weights=weights)
    _, per = objective(data, best)
    print("\nbest {:.4f}  {}".format(score, json.dumps(best)))
    for name, s in per.items():
        print("  {:12s} {:.4f}".format(name, s))
    for src in ("DFL", "MET"):
        vals = [s for n, s in per.items() if n.startswith(src)]
        if vals:
            print("  mean {} {:.4f}".format(src, np.mean(vals)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(best, f, indent=1)


if __name__ == "__main__":
    main()
