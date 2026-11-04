"""Features -> per-frame state scores -> Viterbi -> labels. All weights live in DEFAULTS."""

import numpy as np

from .evidence import features
from .viterbi import viterbi

DEFAULTS = dict(
    # team evidence
    w_team=1.0,      # weight of one frame of clear control evidence
    r_ctrl=2.0,      # metres: confidence decays with distance from ball to nearest player
    s_diff=1.5,      # metres: distance gap between teams that counts as decisive
    v0=12.0,         # m/s: ball speed above which control evidence is discounted (in flight)
    # dead-ball evidence
    w_out=3.0,       # ball beyond the lines
    out_margin=0.5,  # metres
    w_slow=0.6,      # low mean player speed
    s_slow=1.6,      # m/s: mean speed below which play looks stopped
    s_scale=0.8,     # m/s
    w_miss=0.0,      # ball unseen
    bias_d=-0.3,
    # decoder
    p_switch=25.0,   # cost of an H<->A hand-over
    p_dead=25.0,     # cost of entering or leaving D
)


def emissions(f, p):
    T = f["T"]
    E = np.zeros((T, 3))
    dH, dA, v = f["dH"], f["dA"], f["v"]
    ok = f["seen"] & ~np.isnan(dH) & ~np.isnan(dA)
    m = np.fmin(dH, dA)
    conf = np.exp(-np.nan_to_num(m, nan=99.0) / p["r_ctrl"])
    gate = 1.0 / (1.0 + (np.nan_to_num(v, nan=0.0) / p["v0"]) ** 2)
    diff = np.clip(np.nan_to_num(dA - dH) / p["s_diff"], -1, 1)
    lam = np.where(ok, p["w_team"] * diff * conf * gate, 0.0)
    E[:, 0] = lam / 2
    E[:, 1] = -lam / 2

    out = np.nan_to_num(f["out"], nan=-1.0) > p["out_margin"]
    slow = np.clip((p["s_slow"] - np.nan_to_num(f["mean_speed"], nan=p["s_slow"])) / p["s_scale"],
                   -1, 1)
    E[:, 2] = p["bias_d"] + p["w_out"] * out + p["w_slow"] * slow + p["w_miss"] * ~f["seen"]
    return E


def switch_matrix(p):
    s, d = p["p_switch"], p["p_dead"]
    return np.array([[0, s, d], [s, 0, d], [d, d, 0]], dtype=float)


def decode_half(f, p):
    return viterbi(emissions(f, p), switch_matrix(p))


def predict(match, params=None, feats=None):
    p = dict(DEFAULTS, **(params or {}))
    feats = feats or [features(h, match.pitch) for h in match.halves]
    return [decode_half(f, p) for f in feats]
