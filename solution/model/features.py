"""Per-frame features from team-level tracking, identical for BAR-ATH and external matches.

Design rules:
  - Only inputs every source has: ball and player positions, slot teams, half. No camera, no
    homography error, no Smart Tagging (external matches have none, so a model could not learn them).
  - No slot identity over time: tracklets swap and fragment, so everything is a per-frame statistic
    of each team (nearest players to the ball, centroid, spread, speeds).
  - Invariant to pitch flips by construction: x is expressed in each team's attacking direction
    (found per half from where the teams stand, never from labels) and y as distance from the
    centre line.
  - Team-symmetric layout: [ball block | team 0 block | team 1 block | global block]. Swapping the
    two team blocks is exactly the same frame with H and A exchanged, which training uses as
    augmentation so the model cannot learn a home-team bias.

    X, names = frame_features(s0)          # float32 (N, F); NaN-free (missing -> sentinel + flag)
"""

import numpy as np

from solution.pipeline import contracts as C

VERSION = 4
K_RAW = 4            # nearest players per team whose raw relative kinematics are given
FPS = C.FPS
INTERP_MAX = 150     # frames; straight-line ball fill through gaps up to 6 s inside a half
SENTINEL_D = 40.0    # metres, stands in for "no player / no ball"
K_NEAR = 3

BALL_NAMES = ["b_seen", "b_have", "b_gap", "b_since", "b_until", "b_y", "b_speed", "b_speed_s",
              "b_dv", "b_out", "b_line_x", "b_line_y", "b_out_last", "b_out_next", "b_still"]
TEAM_NAMES_V3 = (["d{}".format(i + 1) for i in range(K_NEAR)]
                 + ["n2", "n5", "n10", "near_speed", "near_rel", "near_approach", "goal_dist",
                    "cen_rel", "cen_vx", "spread_x", "spread_y", "mean_speed", "walk", "visible",
                    "last_line", "ball_vx_dir", "behind_share", "press", "d1_trend"])
TEAM_NAMES = (TEAM_NAMES_V3 + ["same_near"]
              + ["p{}_{}".format(i + 1, q) for i in range(K_RAW)
                 for q in ("dx", "dy", "rvx", "rvy")])
GLOBAL_NAMES = ["all_speed", "all_q90", "all_visible", "nearest_team_gap"]
SUPPORTED = (3, 4)   # older versions are exact column subsets of the newest computation


def team_names(version=VERSION):
    return TEAM_NAMES_V3 if version == 3 else TEAM_NAMES


def names(version=VERSION):
    t = team_names(version)
    return BALL_NAMES + ["t0_" + n for n in t] + ["t1_" + n for n in t] + GLOBAL_NAMES


def team_slices(version=VERSION):
    """Column slices of the two team blocks, for the swap augmentation."""
    b, t = len(BALL_NAMES), len(team_names(version))
    return slice(b, b + t), slice(b + t, b + 2 * t)


def swap_teams(X, version=None):
    """Same frames with the two teams exchanged (and the signed global feature negated)."""
    if version is None:
        version = 3 if X.shape[-1] == len(names(3)) else VERSION
    s0, s1 = team_slices(version)
    Y = X.copy()
    Y[..., s0], Y[..., s1] = X[..., s1], X[..., s0]
    Y[..., -1] = -X[..., -1]
    return Y


def _runs(mask):
    return C.runs(mask)


def interpolate(ball, max_gap=INTERP_MAX):
    """Straight-line fill of gaps up to max_gap frames with both ends observed."""
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    for s, e in _runs(~seen):
        if s == 0 or e == len(ball) or e - s > max_gap:
            continue
        w = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
        out[s:e] = ball[s - 1] + w[:, None] * (ball[e] - ball[s - 1])
    return out


def moving_mean(x, w):
    """Centred NaN-aware moving mean over +/- w frames along axis 0."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(v, axis=0)])
    n = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(ok, axis=0)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def velocity(xy, k=2):
    """Centred difference over +/- k frames, m/s, along axis 0. NaN where an end is missing."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    n = len(xy)
    if n > 2 * k:
        v[k:n - k] = (xy[2 * k:] - xy[:n - 2 * k]) * (FPS / (2.0 * k))
    return v


def _since_until(seen):
    """Frames since the last and until the next observed frame (0 on an observed frame)."""
    n = len(seen)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10 ** 9)[::-1])[::-1]
    since = np.minimum(idx - last, 10 ** 6)
    until = np.minimum(nxt - idx, 10 ** 6)
    return since, until


def _directions(players, team):
    """+1 if the team attacks toward +x in this stretch, judged by where the teams stand."""
    with np.errstate(all="ignore"):
        x0 = np.nanmean(players[:, team == 0, 0])
        x1 = np.nanmean(players[:, team == 1, 0])
    d0 = 1.0 if (np.isnan(x0) or np.isnan(x1) or x0 <= x1) else -1.0
    return d0, -d0


def _out_distance(xy):
    L, W = C.PITCH
    return np.maximum(np.maximum(-xy[:, 0], xy[:, 0] - L), np.maximum(-xy[:, 1], xy[:, 1] - W))


def _segment_features(ball_raw, players, team):
    """Features for one contiguous stretch (a half). Returns float32 (T, F)."""
    T, P = players.shape[:2]
    L, W = C.PITCH
    seen = np.isfinite(ball_raw[:, 0])
    ball = interpolate(ball_raw)
    have = np.isfinite(ball[:, 0])

    # -- ball block
    gap = np.zeros(T)
    for s, e in _runs(~seen):
        gap[s:e] = e - s
    since, until = _since_until(seen)
    bv = velocity(ball)
    speed = np.linalg.norm(bv, axis=1)
    speed_s = moving_mean(speed, FPS // 2)
    kk = 4
    dv = np.full(T, np.nan)
    if T > 2 * kk:
        before = (ball[kk:T - kk] - ball[:T - 2 * kk]) / kk
        after = (ball[2 * kk:] - ball[kk:T - kk]) / kk
        dv[kk:T - kk] = np.linalg.norm(after - before, axis=1) * FPS
    out = _out_distance(ball)
    raw_out = np.where(seen, _out_distance(np.where(seen[:, None], ball_raw, 0.0)), np.nan)
    idx = np.arange(T)
    last_i = np.maximum.accumulate(np.where(seen, idx, 0))
    next_i = np.minimum.accumulate(np.where(seen, idx, T - 1)[::-1])[::-1]
    out_last = np.where(np.isfinite(raw_out[last_i]), raw_out[last_i], -SENTINEL_D)
    out_next = np.where(np.isfinite(raw_out[next_i]), raw_out[next_i], -SENTINEL_D)
    line_x = np.minimum(ball[:, 0], L - ball[:, 0])
    line_y = np.minimum(ball[:, 1], W - ball[:, 1])
    still = moving_mean((np.nan_to_num(speed, nan=9.0) < 0.8).astype(float), FPS)

    ball_block = np.stack([
        seen, have, np.log1p(gap), np.log1p(since), np.log1p(until),
        np.abs(ball[:, 1] - W / 2) / (W / 2),
        np.minimum(speed, 40.0), np.minimum(speed_s, 40.0), np.minimum(dv, 80.0),
        np.clip(out, -SENTINEL_D, 10.0), line_x, line_y, np.clip(out_last, -SENTINEL_D, 10.0),
        np.clip(out_next, -SENTINEL_D, 10.0), still,
    ], axis=1)

    # -- player kinematics
    pv = velocity(players)                                   # (T, P, 2)
    ps = np.linalg.norm(pv, axis=2)
    ps = np.where(ps > 12.0, np.nan, ps)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)   # (T, P)
    d = np.where(np.isfinite(d), d, np.inf)
    dirs = _directions(players, team)

    blocks = []
    nearest_d = []
    for k in (0, 1):
        m = team == k
        dk = d[:, m] if m.any() else np.full((T, 1), np.inf)
        part = np.sort(np.partition(dk, min(K_NEAR, dk.shape[1]) - 1, axis=1)[:, :K_NEAR], axis=1) \
            if dk.shape[1] >= K_NEAR else np.sort(dk, axis=1)
        if part.shape[1] < K_NEAR:
            part = np.concatenate([part, np.full((T, K_NEAR - part.shape[1]), np.inf)], axis=1)
        near_d = np.where(np.isfinite(part), part, SENTINEL_D)
        nearest_d.append(near_d[:, 0])
        counts = [(dk < r).sum(axis=1) for r in (2.0, 5.0, 10.0)]
        j = np.argmin(dk, axis=1)
        rows = np.arange(T)
        vk = pv[:, m][rows, j] if m.any() else np.full((T, 2), np.nan)
        near_speed = np.linalg.norm(vk, axis=1)
        rel = np.linalg.norm(vk - bv, axis=1)
        # Is the nearest player closing on the ball (negative = approaching)?
        pk = players[:, m][rows, j] if m.any() else np.full((T, 2), np.nan)
        to_ball = ball - pk
        dist = np.linalg.norm(to_ball, axis=1)
        with np.errstate(all="ignore"):
            approach = -np.sum((vk - bv) * to_ball, axis=1) / np.maximum(dist, 0.5)
        found = np.isfinite(part[:, 0])
        direction = dirs[k]
        # Distance from the ball to the goal this team attacks.
        goal_x = L if direction > 0 else 0.0
        goal_dist = np.abs(goal_x - ball[:, 0]) / L
        with np.errstate(all="ignore"):
            cx = np.nanmean(players[:, m, 0], axis=1)
            cvx = np.nanmean(pv[:, m, 0], axis=1) * direction
            sx = np.nanstd(players[:, m, 0], axis=1)
            sy = np.nanstd(players[:, m, 1], axis=1)
            ms = np.nanmean(ps[:, m], axis=1)
            walk = np.nanmean((ps[:, m] < 1.5).astype(float) + np.where(np.isnan(ps[:, m]), np.nan,
                                                                          0.0), axis=1)
        vis = np.isfinite(players[:, m, 0]).sum(axis=1)
        cen_rel = (cx - ball[:, 0]) * direction
        # Last line: deepest outfield position relative to own goal (defensive line height).
        with np.errstate(all="ignore"):
            own_goal_x = 0.0 if direction > 0 else L
            depth = np.abs(players[:, m, 0] - own_goal_x)
            last_line = np.nanpercentile(depth, 15, axis=1) if m.any() else np.full(T, np.nan)
            # Share of the team goal-side of the ball: nearly all of the defending team.
            ball_depth = np.abs(ball[:, 0] - own_goal_x)
            behind = np.nansum(depth < ball_depth[:, None], axis=1) / np.maximum(vis, 1)
            # Opponents within 3 m of this team's player nearest the ball.
            opp = players[:, team == 1 - k]
            press = np.nansum(np.linalg.norm(opp - pk[:, None, :], axis=2) < 3.0, axis=1)
        # Ball velocity along this team's attacking direction (passes mostly go forward).
        ball_vx_dir = np.clip(np.nan_to_num(bv[:, 0]) * direction, -30, 30)
        # Is the team's nearest player getting closer to the ball over the last 0.4 s?
        d1 = np.where(np.isfinite(part[:, 0]), part[:, 0], np.nan)
        trend = np.zeros(T)
        trend[10:] = d1[10:] - d1[:-10]
        # Same tracklet nearest the ball as 0.4 s ago (a carry), only where both are known.
        slot_ids = np.flatnonzero(m)[j] if m.any() else np.full(T, -1)
        same = np.zeros(T)
        same[10:] = (slot_ids[10:] == slot_ids[:-10]) & found[10:] & found[:-10]
        # Raw kinematics of the K_RAW nearest players, relative to the ball, oriented so that
        # +dx is toward the goal this team attacks and +dy toward the touchline nearest the ball.
        side = np.where(ball[:, 1] >= W / 2, 1.0, -1.0)
        order = np.argsort(dk, axis=1)[:, :K_RAW] if m.any() else np.zeros((T, 0), dtype=int)
        raw = []
        team_players, team_vel = players[:, m], pv[:, m]
        for i in range(K_RAW):
            if i < order.shape[1]:
                idx = order[:, i]
                pos = team_players[rows, idx]
                vel = team_vel[rows, idx]
                ok = np.isfinite(dk[rows, idx]) & np.isfinite(pos[:, 0])
                dx = (pos[:, 0] - ball[:, 0]) * direction
                dy = (pos[:, 1] - ball[:, 1]) * side
                rvx = (vel[:, 0] - bv[:, 0]) * direction
                rvy = (vel[:, 1] - bv[:, 1]) * side
                for q, lim in ((dx, SENTINEL_D), (dy, SENTINEL_D), (rvx, 20.0), (rvy, 20.0)):
                    raw.append(np.where(ok, np.clip(np.nan_to_num(q), -lim, lim), 0.0))
            else:
                raw.extend([np.zeros(T)] * 4)
        blocks.append(np.stack([
            near_d[:, 0], near_d[:, 1], near_d[:, 2],
            counts[0], counts[1], counts[2],
            np.where(found, near_speed, 0.0), np.where(found, np.minimum(rel, 40.0), 0.0),
            np.where(found, np.clip(approach, -20, 20), 0.0),
            goal_dist, cen_rel / L, cvx, sx / L, sy / W, ms, walk, vis, last_line / L,
            ball_vx_dir, np.where(have, behind, 0.0), np.where(found, press, 0.0),
            np.clip(np.nan_to_num(trend), -10, 10), same,
        ] + raw, axis=1))

    with np.errstate(all="ignore"):
        all_speed = np.nanmean(ps, axis=1)
        all_q90 = np.nanpercentile(ps, 90, axis=1)
    all_vis = np.isfinite(players[:, :, 0]).sum(axis=1)
    nearest_gap = np.clip(nearest_d[1] - nearest_d[0], -SENTINEL_D, SENTINEL_D)  # >0: team 0 nearer
    glob = np.stack([moving_mean(all_speed, FPS), moving_mean(all_q90, FPS), all_vis, nearest_gap],
                    axis=1)

    X = np.concatenate([ball_block] + blocks + [glob], axis=1).astype(np.float32)
    # Missing -> 0, after the flags above have recorded what was missing.
    X[~np.isfinite(X)] = 0.0
    return X


def frame_features(s0, ball_xy=None):
    """float32 (N, F) for every frame inside a half; rows on half == 0 frames are zero.

    ball_xy: optional replacement ball track (e.g. S2's); defaults to s0's raw ball.
    """
    return frame_features_version(s0, VERSION, ball_xy)


def frame_features_version(s0, version, ball_xy=None):
    """Features of an older SUPPORTED version: the matching columns of the newest computation."""
    if version not in SUPPORTED:
        raise ValueError("feature version {} not supported (have {})".format(version, SUPPORTED))
    ball_raw = s0["ball_xy"] if ball_xy is None else ball_xy
    N = len(s0["half"])
    X = np.zeros((N, len(names())), dtype=np.float32)
    for number in (1, 2):
        for start, end in C.runs(s0["half"] == number):
            X[start:end] = _segment_features(ball_raw[start:end], s0["player_xy"][start:end],
                                             s0["player_team"])
    if version != VERSION:
        full = {n: i for i, n in enumerate(names())}
        X = X[:, [full[n] for n in names(version)]]
    return X


def segments(s0):
    """[(start, end)] of the play stretches the features are computed over (one per half)."""
    out = []
    for number in (1, 2):
        out.extend(C.runs(s0["half"] == number))
    return sorted(out)
