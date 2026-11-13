"""S2: ball track inside in-play segments. STUB: replace track(), keep the contract.

    python -m solution.pipeline.s2_ball.run --stages outputs/stages

In:  s0_tracking.npz, s1_inplay.npz
Out: s2_ball.npz (ball_xy, ball_vel, ball_src, ball_conf, in_air)      -- PLAN.md section 3

Stub rule: the raw ATD ball, plus straight-line interpolation through gaps of at most MAX_FILL
frames whose two ends lie in the same alive segment. Nothing is filled across or inside a
stoppage. No outlier removal, no carried ball, no in_air estimate.
"""

import argparse
import os

import numpy as np

from solution.pipeline import contracts as C

# Params
MAX_FILL = 150      # frames (6 s): longest gap bridged by a straight line
FILL_CONF = 0.5     # confidence given to an interpolated position


def track(s0, s1):
    """The S2 arrays from S0 and S1. This is the function a real S2 replaces."""
    raw = s0["ball_xy"]
    n = len(raw)
    xy = raw.copy()
    seen = np.isfinite(raw[:, 0])
    src = np.where(seen, C.SRC_OBSERVED, C.SRC_NONE).astype(np.uint8)
    conf = np.where(seen, 1.0, np.nan).astype(np.float32)

    for start, end in C.alive_segments(s1["alive"]):
        inside = np.flatnonzero(seen[start:end]) + start
        for left, right in zip(inside[:-1], inside[1:]):
            gap = right - left - 1
            if 0 < gap <= MAX_FILL:
                weight = (np.arange(1, gap + 1) / float(gap + 1))[:, None]
                xy[left + 1:right] = raw[left] * (1.0 - weight) + raw[right] * weight
                src[left + 1:right] = C.SRC_INTERPOLATED
                conf[left + 1:right] = FILL_CONF

    # Central difference where both neighbours exist; NaN otherwise.
    vel = np.full((n, 2), np.nan, dtype=np.float32)
    vel[1:-1] = (xy[2:] - xy[:-2]) * (C.FPS / 2.0)
    return {
        "ball_xy": xy.astype(np.float32),
        "ball_vel": vel,
        "ball_src": src,
        "ball_conf": conf,
        "in_air": np.full(n, np.nan, dtype=np.float32),
    }


def run(stage_dir=None):
    s0 = C.load("s0_tracking", stage_dir)
    s1 = C.load("s1_inplay", stage_dir)
    s2 = track(s0, s1)
    C.check_s2_against_s1(s2, s1)
    out = [C.save("s2_ball", stage_dir, s2, n=len(s0["half"]),
                  params={"stub": True, "max_fill": MAX_FILL, "fill_conf": FILL_CONF})]
    alive = s1["alive"] == 1
    if alive.any():
        print("S2: ball known on {:.1f}% of alive frames ({:.1f}% observed)".format(
            100.0 * (s2["ball_src"][alive] != C.SRC_NONE).mean(),
            100.0 * (s2["ball_src"][alive] == C.SRC_OBSERVED).mean()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR)
    args = parser.parse_args()
    for path in run(args.stages):
        print("S2: wrote {}".format(os.path.relpath(path, C.ROOT)))


if __name__ == "__main__":
    main()
