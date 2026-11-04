"""Score the decoder on other games with the official metric code.

    .venv/bin/python -m solution.decoder.evaluate            # dev matches, default weights
    .venv/bin/python -m solution.decoder.evaluate --baseline # the starter baseline, same data

Degraded tracking (degrade.py) stands in for the challenge's automatic tracking. Reference labels
are DFL's own per-frame ball status and possession.
"""

import argparse
import os
import pickle
import sys
import zlib

import numpy as np

from . import dfl
from .degrade import degrade, missing_rate
from .evidence import features
from .pipeline import DEFAULTS, predict

ROOT = dfl.ROOT
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Five matches to iterate on, two held back and scored once at the very end.
HOLDOUT = ("DFL-MAT-J03WR9", "DFL-MAT-J03WQQ")
FEAT_CACHE = os.path.join(dfl.CACHE_DIR, "features")


def dev_ids():
    return [m for m in dfl.match_ids() if m not in HOLDOUT]


def prepared(match_id, seed=0):
    """(degraded match, features) for one match, cached on disk."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    path = os.path.join(FEAT_CACHE, "{}_s{}.pkl".format(match_id, seed))
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    clean = dfl.load(match_id)
    match = degrade(clean, seed=seed + zlib.crc32(match_id.encode()) % 1000)
    feats = [features(h, match.pitch) for h in match.halves]
    item = (match, feats)
    with open(path, "wb") as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return item


def concat(match, codes_per_half):
    """Halves on one timeline with an X frame between them, as the scorer expects."""
    parts = []
    for i, codes in enumerate(codes_per_half):
        if i:
            parts.append(np.array([pm.X], dtype=np.int8))
        parts.append(np.asarray(codes, dtype=np.int8))
    return np.concatenate(parts)


def score(match, pred_halves):
    gt = concat(match, [h.labels for h in match.halves])
    pred = concat(match, pred_halves)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)
    cm = pm.scored_confusion(pm.confusion(gt, pred))
    return dict(spellF1=spell["mean_f1"], macroF1=pm.macro_f1(cm),
                n_gt=spell["n_gt"], n_pred=spell["n_pred"])


def baseline(match, feats, window=50):
    """The starter's nearest-player + majority vote, on the same degraded data."""
    out = []
    for f in feats:
        near = np.where(np.isnan(f["dH"]) | np.isnan(f["dA"]), -1,
                        np.where(f["dH"] <= f["dA"], 0, 1))
        near[~f["seen"]] = -1
        h = np.concatenate(([0], np.cumsum(near == 0)))
        a = np.concatenate(([0], np.cumsum(near == 1)))
        i = np.arange(len(near))
        lo, hi = np.maximum(0, i - window), np.minimum(len(near), i + window + 1)
        nh, na = h[hi] - h[lo], a[hi] - a[lo]
        lab = np.full(len(near), 2, dtype=np.int8)
        s = nh + na > 0
        lab[s] = np.where(nh[s] >= na[s], 0, 1)
        out.append(lab)
    return out


def evaluate(ids, params=None, method="decoder", verbose=True):
    rows = []
    for mid in ids:
        match, feats = prepared(mid)
        pred = baseline(match, feats) if method == "baseline" else predict(match, params, feats)
        r = score(match, pred)
        r["match"] = mid
        r["miss"] = missing_rate(match)
        rows.append(r)
        if verbose:
            print("  {match}  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  "
                  "spells {n_pred}/{n_gt}  ball missing {miss:.0%}".format(**r))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spellF1", "macroF1")}
    if verbose:
        print("  MEAN  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}".format(**mean))
    return mean, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="score the held-back matches")
    args = ap.parse_args()
    ids = list(HOLDOUT) if args.holdout else dev_ids()
    evaluate(ids, method="baseline" if args.baseline else "decoder")
