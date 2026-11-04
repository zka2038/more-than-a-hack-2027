"""Features -> per-frame state scores -> Viterbi -> labels. All weights live in DEFAULTS."""

import numpy as np

from .evidence import features
from .viterbi import viterbi

DEFAULTS = dict(
    # team evidence
    w_team=0.5,      # weight of one frame of clear control evidence
    r_ctrl=4.6,      # metres: confidence decays with distance from ball to nearest player
    s_diff=0.54,     # metres: distance gap between teams that counts as decisive
    v0=12.0,         # m/s: ball speed above which control evidence is discounted (in flight)
    # touch evidence
    w_touch=2.0,     # one detected touch
    w_carry=0.2,     # per frame, for the team that touched last
    r_touch=1.0,     # metres
    dv0=4.0,         # m/s velocity change that marks a kick or deflection
    v_drib=4.0,      # m/s: a ball this slow next to a player is on that player's foot
    # dead-ball evidence
    w_out=8.4,       # ball beyond the lines
    out_margin=0.5,  # metres
    w_slow=0.35,     # low mean player speed
    s_slow=3.2,      # m/s: mean speed below which play looks stopped
    s_scale=0.8,     # m/s
    w_miss=0.0,      # ball unseen
    w_still=0.5,     # ball at rest (a restart is being prepared)
    w_gap=0.3,       # long ball gap
    gap0=100.0,      # frames: gaps shorter than this say nothing
    w_edge=0.58,     # gap that began with the ball at or beyond a line
    bias_d=-0.3,
    # decoder
    p_switch=6.0,    # cost of an H<->A hand-over
    p_dead=10.0,     # cost of entering or leaving D
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

    # Touches: ball within r_touch of a player and either deflected (velocity change) or slow
    # enough to be on that player's foot. Each touch votes for the toucher's team, and the last
    # touch is carried forward so a hand-over lands on the new team's first touch.
    touch = (f["seen"] & (np.nan_to_num(f["near_d"], nan=99) < p["r_touch"])
             & ((np.nan_to_num(f["dv"], nan=0) > p["dv0"])
                | (np.nan_to_num(f["v"], nan=99) < p["v_drib"])))
    sign = np.where(f["near_team"] == 0, 1.0, -1.0)
    lam = lam + p["w_touch"] * np.where(touch, sign, 0.0)
    idx = np.maximum.accumulate(np.where(touch, np.arange(T), -1))
    carried = np.where(idx >= 0, sign[np.maximum(idx, 0)], 0.0)
    lam = lam + p["w_carry"] * carried
    E[:, 0] = lam / 2
    E[:, 1] = -lam / 2

    out = np.nan_to_num(f["out"], nan=-1.0) > p["out_margin"]
    slow = np.clip((p["s_slow"] - np.nan_to_num(f["mean_speed"], nan=p["s_slow"])) / p["s_scale"],
                   -1, 1)
    gap = np.clip((f["gap_len"] - p["gap0"]) / (3 * p["gap0"]), 0, 1)
    E[:, 2] = (p["bias_d"] + p["w_out"] * out + p["w_slow"] * slow + p["w_miss"] * ~f["seen"]
               + p["w_still"] * f["still"] + p["w_gap"] * gap + p["w_edge"] * f["gap_edge"])
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
