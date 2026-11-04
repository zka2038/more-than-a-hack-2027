"""Per-frame features from team-level tracking. No learning: geometry and kinematics only.

Everything here is computed once per half and does not depend on the decoder's weights, so a
parameter sweep only re-runs the cheap part (emissions + Viterbi).
"""

import numpy as np

FPS = 25


def _nanmin_team(d, mask):
    x = np.where(mask[None, :], d, np.nan)
    with np.errstate(all="ignore"):
        out = np.nanmin(np.where(np.isnan(x), np.inf, x), axis=1)
    out[np.isinf(out)] = np.nan
    return out


def moving_mean(x, w):
    """Centred NaN-aware moving mean over +/- w frames."""
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    c = np.concatenate(([0.0], np.cumsum(v)))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def features(half, pitch):
    ball, players, team = half.ball, half.players, half.team
    T = len(ball)
    seen = ~np.isnan(ball[:, 0])
    d = np.linalg.norm(players - ball[:, None, :], axis=2)          # (T, P)
    dH = _nanmin_team(d, team == 0)
    dA = _nanmin_team(d, team == 1)

    # Ball speed from a centred difference over 2 frames each side, only where both ends exist.
    k = 2
    v = np.full(T, np.nan)
    a, b = ball[: T - 2 * k], ball[2 * k:]
    v[k: T - k] = np.linalg.norm(b - a, axis=1) * FPS / (2 * k)

    # Ball outside the lines (positive metres beyond the nearer line).
    L, W = pitch
    out = np.maximum(np.abs(ball[:, 0]) - L / 2, np.abs(ball[:, 1]) - W / 2)

    # Mean player speed over the field, 1 s smoothing. Low for long stretches = stoppage.
    pv = np.full((T, players.shape[1]), np.nan)
    pv[k: T - k] = np.linalg.norm(players[2 * k:] - players[: T - 2 * k], axis=2) * FPS / (2 * k)
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(np.where(pv > 12, np.nan, pv), axis=1)
    mean_speed = moving_mean(mean_speed, FPS)

    return dict(seen=seen, dH=dH, dA=dA, v=v, out=out, mean_speed=mean_speed, T=T)
