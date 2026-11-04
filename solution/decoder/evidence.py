"""Per-frame features from team-level tracking. No learning: geometry and kinematics only.

Everything here is computed once per half and does not depend on the decoder's weights, so a
parameter sweep only re-runs the cheap part (emissions + Viterbi).
"""

import numpy as np

FPS = 25


def _nanmin_team(d, mask):
    x = np.where(mask[None, :], d, np.inf)
    x = np.where(np.isnan(x), np.inf, x)
    out = x.min(axis=1)
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


def runs_of(mask):
    """[(start, stop)] of True runs."""
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def features(half, pitch):
    ball, players, team = half.ball, half.players, half.team
    T = len(ball)
    L, W = pitch
    seen = ~np.isnan(ball[:, 0])
    d = np.linalg.norm(players - ball[:, None, :], axis=2)          # (T, P)
    dH = _nanmin_team(d, team == 0)
    dA = _nanmin_team(d, team == 1)

    # Ball speed from a centred difference over 2 frames each side, only where both ends exist.
    k = 2
    v = np.full(T, np.nan)
    v[k: T - k] = np.linalg.norm(ball[2 * k:] - ball[: T - 2 * k], axis=1) * FPS / (2 * k)

    # Signed distance beyond the nearer line (positive = outside the pitch).
    out = np.maximum(np.abs(ball[:, 0]) - L / 2, np.abs(ball[:, 1]) - W / 2)

    # Mean player speed over the field, 1 s smoothing. Low for long stretches = stoppage.
    pv = np.full((T, players.shape[1]), np.nan)
    pv[k: T - k] = np.linalg.norm(players[2 * k:] - players[: T - 2 * k], axis=2) * FPS / (2 * k)
    pv = np.where(pv > 12, np.nan, pv)
    cnt = (~np.isnan(pv)).sum(axis=1)
    mean_speed = np.where(cnt > 0, np.nansum(pv, axis=1) / np.maximum(cnt, 1), np.nan)
    mean_speed = moving_mean(mean_speed, FPS)

    # Ball at rest: smoothed speed over 0.5 s below 0.6 m/s, with the ball seen.
    v_s = moving_mean(v, FPS // 2)
    still = seen & (np.nan_to_num(v_s, nan=9.0) < 0.6)

    # Ball-gap features: how long is the gap this frame sits in, and was the ball last seen
    # near (or beyond) a line, which is how most stoppages begin.
    gap_len = np.zeros(T)
    gap_edge = np.zeros(T, dtype=bool)
    for s, e in runs_of(~seen):
        gap_len[s:e] = e - s
        if s > 0:
            gap_edge[s:e] = out[s - 1] > -2.0
    # Touch candidates: the nearest player and its team, plus how sharply the ball's velocity
    # changes there (a kick, trap or deflection). k=4 frames each side averages out jitter.
    kk = 4
    dv = np.full(T, np.nan)
    before = (ball[kk: T - kk] - ball[: T - 2 * kk]) / kk
    after = (ball[2 * kk:] - ball[kk: T - kk]) / kk
    dv[kk: T - kk] = np.linalg.norm(after - before, axis=1) * FPS
    near_d = np.fmin(dH, dA)
    near_team = np.where(np.nan_to_num(dH, nan=99) <= np.nan_to_num(dA, nan=99), 0, 1)
    return dict(seen=seen, dH=dH, dA=dA, v=v, out=out, mean_speed=mean_speed, still=still,
                gap_len=gap_len, gap_edge=gap_edge, dv=dv, near_d=near_d,
                near_team=near_team, T=T)
