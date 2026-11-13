"""Make clean external tracking look like the challenge's automatic tracking (ATD).

Targets come from what DATA.md publishes about the BAR-ATH ATD, never from its ground truth:
  - the ball is missing on ~41% of graded frames, disproportionately in crowds, in the air and out
    of play, in bursts;
  - about 20.5 of 22 players are tracked on average, with fragmented tracklets;
  - tracklets occasionally carry the wrong team (ghosts);
  - positions come from a ground-plane homography, so a ball in the air is projected away from
    the camera.

Every draw comes from a seeded generator, so a (match, seed) pair always gives the same data.
`sample_params(rng)` randomises the knobs around those targets so a model trained on several seeds
does not overfit one noise setting (domain randomisation); evaluation uses DEFAULTS.
"""

import numpy as np

from solution.pipeline import contracts as C
from solution.bench.external import player_speed

DEFAULTS = dict(
    ball_missing=0.41,      # target share of graded frames with no ball (DATA.md)
    gap_mean=40.0,          # frames; mean length of a ball gap
    crowd_gain=0.8,         # hazard multiplier per extra player within 3 m of the ball
    air_gain=2.0,           # extra hazard when the ball is above 1.8 m
    out_gain=4.0,           # extra hazard when the ball is off the pitch
    ball_jitter=0.25,       # metres, temporally correlated
    ball_white=0.10,        # metres, per-frame detection noise
    false_ball=0.002,       # share of frames with a spurious ball detection
    player_jitter=0.25,     # metres, temporally correlated
    player_drop=0.07,       # share of player-frames lost, in bursts
    player_burst=50.0,      # frames; mean dropout burst
    ghosts=2,               # extra slots that copy a player under the wrong team
    ghost_rate=1 / 3000.0,  # per-frame chance a ghost appears
    camera=(52.65, -40.0, 20.0),   # tactical camera position (x, y, height) for the air effect
)


def sample_params(rng):
    p = dict(DEFAULTS)
    p["ball_missing"] = float(rng.uniform(0.33, 0.50))
    p["gap_mean"] = float(rng.uniform(25, 70))
    p["crowd_gain"] = float(rng.uniform(0.3, 1.2))
    p["ball_jitter"] = float(rng.uniform(0.1, 0.45))
    p["player_jitter"] = float(rng.uniform(0.1, 0.4))
    p["player_drop"] = float(rng.uniform(0.03, 0.12))
    p["ghosts"] = int(rng.integers(0, 4))
    return p


def bursts(rng, T, share, mean_len):
    """Boolean mask with about `share` of frames True, in runs of mean length `mean_len`."""
    if share <= 0:
        return np.zeros(T, dtype=bool)
    mean_off = mean_len * (1 - share) / share
    n = int(T / (mean_len + mean_off)) * 2 + 10
    on = rng.geometric(1.0 / mean_len, n)
    off = rng.geometric(1.0 / max(mean_off, 1.0), n)
    lengths = np.empty(2 * n, dtype=np.int64)
    lengths[0::2], lengths[1::2] = off, on
    edges = np.cumsum(lengths)
    state = np.zeros(T + 1, dtype=np.int8)
    starts, ends = edges[0::2], edges[1::2]
    keep = starts < T
    np.add.at(state, starts[keep], 1)
    np.add.at(state, np.minimum(ends[keep], T), -1)
    return np.cumsum(state[:T]) > 0


def smooth_noise(rng, shape, sigma, width=12):
    """Gaussian noise smoothed along axis 0 by a moving average, rescaled to std sigma."""
    if sigma <= 0:
        return np.zeros(shape, dtype=np.float32)
    white = rng.normal(0.0, 1.0, shape)
    kernel = np.ones(width) / width
    c = np.cumsum(np.concatenate([np.zeros((1,) + shape[1:]), white]), axis=0)
    sm = (c[width:] - c[:-width]) / width
    pad = np.repeat(sm[-1:], shape[0] - len(sm), axis=0)
    sm = np.concatenate([sm, pad])[: shape[0]]
    return (sm / max(sm.std(), 1e-9) * sigma).astype(np.float32)


def _ball_gaps(rng, hazard, gap_mean):
    """Markov-like gaps: a gap starts where u < hazard (outside a gap), lasts ~Geom(1/gap_mean)."""
    T = len(hazard)
    candidates = np.flatnonzero(rng.random(T) < hazard)
    mask = np.zeros(T, dtype=bool)
    t_free = 0
    i = 0
    while i < len(candidates):
        t = candidates[i]
        if t < t_free:
            i = np.searchsorted(candidates, t_free)
            continue
        length = int(rng.geometric(1.0 / gap_mean))
        mask[t:t + length] = True
        t_free = t + length + 1
        i += 1
    return mask


def _air_shift(ball, z, camera):
    """Ground-plane projection of a ball at height z seen from a camera at `camera`."""
    cx, cy, ch = camera
    z = np.clip(np.nan_to_num(z), 0.0, ch * 0.8)
    scale = ch / (ch - z)
    out = ball.copy()
    out[:, 0] = cx + (ball[:, 0] - cx) * scale
    out[:, 1] = cy + (ball[:, 1] - cy) * scale
    return out


def degrade(s0, gt, info, seed=0, params=None):
    """A new s0 dict (contract-shaped) with ATD-like noise. The inputs are not modified."""
    rng = np.random.default_rng(seed)
    p = dict(DEFAULTS if params is None else params)
    T, P = s0["player_xy"].shape[:2]
    clean_ball, clean_players = s0["ball_xy"], s0["player_xy"]
    graded = gt != C.X

    # Ball: air projection, jitter, then gaps whose hazard rises in crowds / air / off the pitch.
    ball = clean_ball.copy()
    z = info.get("ball_z")
    if z is not None:
        ball = _air_shift(ball, z, p["camera"])
    ball = ball + smooth_noise(rng, ball.shape, p["ball_jitter"]) \
        + rng.normal(0, p["ball_white"], ball.shape).astype(np.float32)

    d = np.linalg.norm(clean_players - clean_ball[:, None, :], axis=2)
    crowd = np.nansum(d < 3.0, axis=1)
    high = np.nan_to_num(z) > 1.8 if z is not None else np.zeros(T, dtype=bool)
    L, W = C.PITCH
    off = ((clean_ball[:, 0] < 0) | (clean_ball[:, 0] > L) | (clean_ball[:, 1] < 0)
           | (clean_ball[:, 1] > W))
    shape = (1 + p["crowd_gain"] * np.clip(crowd - 1, 0, 4)) * (1 + p["air_gain"] * high) \
        * (1 + p["out_gain"] * off)
    # Scale the hazard so the missing share on graded frames lands near the target.
    f, L_gap = p["ball_missing"], p["gap_mean"]
    scale = f / (L_gap * (1 - f)) / max(shape[graded].mean(), 1e-9)
    for _ in range(3):
        mask = _ball_gaps(np.random.default_rng(seed + 7), np.clip(scale * shape, 0, 1), L_gap)
        got = (mask | np.isnan(clean_ball[:, 0]))[graded].mean()
        scale *= np.clip((f / max(1 - f, 1e-6)) / max(got / max(1 - got, 1e-6), 1e-6), 0.25, 4)
    mask = _ball_gaps(rng, np.clip(scale * shape, 0, 1), L_gap)
    ball[mask] = np.nan

    # Rare spurious detections: the "ball" jumps onto a random visible player's feet.
    false = rng.random(T) < p["false_ball"]
    for t in np.flatnonzero(false):
        seen = np.flatnonzero(np.isfinite(clean_players[t, :, 0]))
        if len(seen):
            ball[t] = clean_players[t, rng.choice(seen)] + rng.normal(0, 0.3, 2)

    # Players: correlated jitter, dropout bursts, and ghost slots with the opposite team.
    players = clean_players + smooth_noise(rng, clean_players.shape, p["player_jitter"])
    for j in range(P):
        players[bursts(rng, T, p["player_drop"], p["player_burst"]), j] = np.nan
    team = list(s0["player_team"])
    ghosts = []
    for _ in range(int(p["ghosts"])):
        j = int(rng.integers(P))
        on = bursts(rng, T, p["ghost_rate"] * 60.0, 60.0)
        g = np.full((T, 2), np.nan, dtype=np.float32)
        g[on] = clean_players[on, j]
        ghosts.append(g[:, None, :])
        team.append(1 - int(s0["player_team"][j]))
    if ghosts:
        players = np.concatenate([players] + ghosts, axis=1)

    out = {
        "ball_xy": ball.astype(np.float32),
        "player_xy": players.astype(np.float32),
        "player_speed": player_speed(players),
        "player_team": np.asarray(team, dtype=np.int8),
        "half": s0["half"].copy(),
        "cam_xy": s0["cam_xy"].copy(),
        "reproj_err": s0["reproj_err"].copy(),
    }
    C.validate("s0_tracking", out)
    return out


def missing_rate(s0, gt):
    graded = gt != C.X
    return float(np.isnan(s0["ball_xy"][graded, 0]).mean())
