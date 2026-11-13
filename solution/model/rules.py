"""Physical rules applied on top of the learned evidence.

Kick-off detection: BAR-ATH's second half is graded from before its kick-off, and those frames are
all D (FAQ.md). The training matches start at kick-off, so the model never saw a pre-kick-off
stretch. The kick-off is found from tracking alone: the ball close to the centre spot while each
team stands in its own half, and then the ball leaves the spot. No frame number is hard-coded; the
published anchor is only a check (PLAN.md, Hard rule 3).
"""

import numpy as np

from solution.pipeline import contracts as C
from solution.model import features as FT

SPOT_RADIUS = 2.5       # metres from the centre spot
OWN_HALF_SHARE = 0.85   # share of each team's visible players in their own half
LEAVE_DIST = 4.0        # metres from the spot the ball must reach ...
LEAVE_FRAMES = 75       # ... within this many frames (3 s)
SEARCH_FRAMES = 25 * 600   # look for the kick-off in the first 10 minutes of a half


def kickoff_frame(s0, start, end):
    """First kick-off frame in [start, end), or None if no frame matches."""
    L, W = C.PITCH
    stop = min(end, start + SEARCH_FRAMES)
    ball = s0["ball_xy"][start:stop]
    players = s0["player_xy"][start:stop]
    team = s0["player_team"]
    if not len(ball):
        return None
    d_spot = np.hypot(ball[:, 0] - L / 2, ball[:, 1] - W / 2)
    d0, d1 = FT._directions(s0["player_xy"][start:end], team)
    own = []
    for k, direction in ((0, d0), (1, d1)):
        x = players[:, team == k, 0]
        vis = np.isfinite(x)
        # A team attacking +x defends the low-x half.
        inside = (x <= L / 2 + 1.0) if direction > 0 else (x >= L / 2 - 1.0)
        with np.errstate(invalid="ignore"):
            own.append((inside & vis).sum(axis=1) / np.maximum(vis.sum(axis=1), 1))
    lined_up = (own[0] >= OWN_HALF_SHARE) & (own[1] >= OWN_HALF_SHARE)
    on_spot = np.nan_to_num(d_spot, nan=99.0) <= SPOT_RADIUS
    for t in np.flatnonzero(on_spot & lined_up):
        ahead = d_spot[t + 1:t + 1 + LEAVE_FRAMES]
        if np.any(np.nan_to_num(ahead, nan=0.0) >= LEAVE_DIST):
            # The kick-off is the last frame the ball is still at its resting point.
            leave = t + 1 + int(np.argmax(np.nan_to_num(ahead, nan=0.0) >= LEAVE_DIST))
            rest = ball[t]
            moved = np.hypot(ball[t:leave, 0] - rest[0], ball[t:leave, 1] - rest[1])
            still = np.flatnonzero(np.nan_to_num(moved, nan=99.0) < 0.75)
            return start + t + int(still[-1]) if len(still) else start + t
    return None


def apply_kickoffs(s0, codes, p_dead=None, min_dead_share=0.5):
    """D on every frame of a half before its detected kick-off. Returns (codes, [kick-off frames]).

    A detection is accepted only if the model already calls most frames before it dead
    (p_dead >= 0.5 on at least `min_dead_share` of them): the rule sharpens the boundary, it
    cannot by itself declare minutes of play dead because it missed the real kick-off.
    """
    codes = codes.copy()
    found = []
    for s, e in FT.segments(s0):
        k = kickoff_frame(s0, s, e)
        if k is not None and k > s and p_dead is not None:
            if np.mean(np.nan_to_num(p_dead[s:k]) >= 0.5) < min_dead_share:
                k = None
        found.append(k)
        if k is not None and k > s:
            codes[s:k] = C.D
    return codes, found
