"""Coordinate search over the decoder weights, scored on other games only.

    .venv/bin/python -m solution.decoder.tune            # leave-one-match-out on the dev matches
    .venv/bin/python -m solution.decoder.tune --fit-all  # fit on all dev matches, print weights

Leave-one-match-out: for each dev match, weights are tuned on the other dev matches and the
untouched match is scored with them. The mean of those scores is the iteration metric, so no
match is ever scored with weights that saw it.
"""

import argparse
import json

import numpy as np

from . import evaluate as ev
from .pipeline import DEFAULTS, predict

# Parameters searched and how: multiplicative steps for positive scales, additive for biases.
SEARCH = {
    "w_team": "mul", "r_ctrl": "mul", "s_diff": "mul", "v0": "mul",
    "w_out": "mul", "w_slow": "mul", "s_slow": "mul", "w_still": "mul",
    "w_gap": "mul", "gap0": "mul", "w_edge": "mul", "bias_d": "add",
    "w_touch": "mul", "w_carry": "mul", "r_touch": "mul", "dv0": "mul", "v_drib": "mul",
    "p_switch": "mul", "p_dead": "mul",
}


def objective(params, data):
    return float(np.mean([ev.score(m, predict(m, params, f))["spellF1"] for m, f in data]))


def coordinate_search(data, start=None, rounds=3, keys=None, verbose=False):
    p = dict(DEFAULTS, **(start or {}))
    best = objective(p, data)
    keys = keys or list(SEARCH)
    for r in range(rounds):
        step = (2.0, 1.4, 1.15)[min(r, 2)]
        improved = False
        for k in keys:
            if k not in p:
                continue
            for cand in ((p[k] * step, p[k] / step) if SEARCH.get(k, "mul") == "mul"
                         else (p[k] + 0.5 / (r + 1), p[k] - 0.5 / (r + 1))):
                q = dict(p, **{k: cand})
                s = objective(q, data)
                if s > best + 1e-4:
                    p, best, improved = q, s, True
        if verbose:
            print("    round {} step {} -> {:.4f}".format(r, step, best), flush=True)
        if not improved and step == 1.15:
            break
    return p, best


def lomo(ids, start=None, rounds=3, verbose=True):
    data = {i: ev.prepared(i) for i in ids}
    rows = []
    for held in ids:
        train = [data[i] for i in ids if i != held]
        p, fit = coordinate_search(train, start, rounds)
        m, f = data[held]
        r = ev.score(m, predict(m, p, f))
        r.update(match=held, fit=fit)
        rows.append(r)
        if verbose:
            print("  {match}  test spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}  "
                  "spells {n_pred}/{n_gt}  (train {fit:.4f})".format(**r), flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spellF1", "macroF1")}
    if verbose:
        print("  LOMO MEAN  spellF1 {spellF1:.4f}  macroF1 {macroF1:.4f}".format(**mean))
    return mean, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-all", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    ids = ev.dev_ids()
    if args.fit_all:
        p, s = coordinate_search([ev.prepared(i) for i in ids], rounds=args.rounds, verbose=True)
        print(json.dumps(p, indent=1))
        print("train spellF1 {:.4f}".format(s))
    else:
        lomo(ids, rounds=args.rounds)
