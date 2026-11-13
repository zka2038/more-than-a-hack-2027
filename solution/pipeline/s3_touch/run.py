"""S3: who touched the ball last. STUB: replace detect(), keep the contract.

    python -m solution.pipeline.s3_touch.run --stages outputs/stages

In:  s0_tracking.npz, s1_inplay.npz, s2_ball.npz
Out: s3_touch.npz (last_touch_team, frames_since_touch, touch_ll, control), s3_touches.csv
                                                                        -- PLAN.md section 4

Stub rule: on an alive frame, the nearest tracked player within REACH metres of the S2 ball is
touching it. A run of frames with the same slot nearest is one touch (kind "carry" once it lasts
CARRY_FRAMES, else "unknown"). No change-point detection, no duels, no restart attribution.
"""

import argparse
import os

import numpy as np
import pandas as pd

from solution.pipeline import contracts as C

# Params
REACH = 1.5          # metres from ball to player for a touch
CARRY_FRAMES = 25    # frames the same slot must stay on the ball to count as control
TOUCH_LL = 2.0       # log-evidence against the other team on a touch frame
CHUNK = 20000        # frames per distance batch, to bound memory


def nearest_slot(ball_xy, player_xy):
    """(slot (N,), distance (N,)) of the closest tracked player; -1 / inf where there is none."""
    n = len(ball_xy)
    slot = np.full(n, -1, dtype=np.int32)
    distance = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, CHUNK):
        end = min(n, start + CHUNK)
        delta = player_xy[start:end] - ball_xy[start:end, None, :]
        d = np.sqrt((delta ** 2).sum(axis=2))
        d[~np.isfinite(d)] = np.inf
        best = d.argmin(axis=1)
        best_d = d[np.arange(end - start), best]
        found = np.isfinite(best_d)
        slot[start:end][found] = best[found]
        distance[start:end][found] = best_d[found]
    return slot, distance


def detect(s0, s1, s2):
    """(arrays, touch table). This is the function a real S3 replaces."""
    n = len(s0["half"])
    team_of = s0["player_team"]
    slot, distance = nearest_slot(s2["ball_xy"], s0["player_xy"])
    on_ball = (s1["alive"] == 1) & (distance <= REACH)
    slot = np.where(on_ball, slot, -1)

    touch_ll = np.zeros((n, 2), dtype=np.float32)
    control = np.zeros((n, 2), dtype=np.float32)
    last_team = np.full(n, C.UNKNOWN, dtype=np.int8)
    since = np.full(n, -1, dtype=np.int32)
    rows = []

    for seg_start, seg_end in C.alive_segments(s1["alive"]):
        # Touches: maximal runs of one slot on the ball, inside this alive segment.
        seg = slot[seg_start:seg_end]
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        for a, b in zip(starts + seg_start, ends + seg_start):
            who = slot[a]
            if who < 0:
                continue
            team = int(team_of[who])
            touch_ll[a:b, 1 - team] = -TOUCH_LL
            carried = b - a >= CARRY_FRAMES
            if carried:
                control[a + CARRY_FRAMES - 1:b, team] = 1.0
            rows.append({
                "touch_id": len(rows), "frame": int(a), "team": team, "slot": int(who),
                "x": float(s2["ball_xy"][a, 0]), "y": float(s2["ball_xy"][a, 1]),
                "kind": "carry" if carried else "unknown",
                "conf": float(s2["ball_conf"][a]) * 0.8,
                "p_home": 1.0 if team == C.HOME else 0.0,
            })

        # Last touch carried forward, reset at the start of every alive segment.
        idx = np.arange(seg_start, seg_end)
        touched = seg >= 0
        latest = np.maximum.accumulate(np.where(touched, idx, -1))
        known = latest >= 0
        last_team[idx[known]] = team_of[slot[latest[known]]]
        since[idx[known]] = idx[known] - latest[known]

    table = pd.DataFrame(rows, columns=C.TABLE_SCHEMAS["s3_touches"])
    arrays = {"last_touch_team": last_team, "frames_since_touch": since,
              "touch_ll": touch_ll, "control": control}
    return arrays, table


def model_evidence(s0, model_path=None):
    """(touch_ll (N,2), control (N,2)) from the learned model.

    touch_ll = log P(H), log P(A) per frame (0 = no evidence, outside the halves).
    control  = P(possession changes here) split by P(team | alive), so control.sum(axis=1) is the
               change probability S4 uses to make a real hand-over cheaper to decode.
    """
    from solution.model import infer
    logp, change = infer.evidence(s0, model_path)
    inside = np.isfinite(logp).all(axis=1)
    touch_ll = np.where(inside[:, None], logp[:, :2], 0.0).astype(np.float32)
    p = np.exp(np.where(inside[:, None], logp[:, :2], 0.0))
    share = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)
    p_change = 1.0 / (1.0 + np.exp(-change.astype(np.float64)))
    control = np.where(inside[:, None], share * p_change[:, None], 0.0).astype(np.float32)
    return touch_ll, control


def run(stage_dir=None, model_path=None):
    s0 = C.load("s0_tracking", stage_dir)
    s1 = C.load("s1_inplay", stage_dir)
    s2 = C.load("s2_ball", stage_dir)
    n = len(s0["half"])
    arrays, table = detect(s0, s1, s2)
    params = {"stub": True, "reach": REACH, "carry_frames": CARRY_FRAMES, "touch_ll": TOUCH_LL}
    from solution.pipeline.s1_inplay.run import use_model
    if use_model(model_path):
        arrays["touch_ll"], arrays["control"] = model_evidence(s0, model_path)
        params = {"model": True, "reach": REACH, "carry_frames": CARRY_FRAMES}
    out = [C.save("s3_touch", stage_dir, arrays, params=params, n=n),
           C.save_table("s3_touches", stage_dir, table, n=n)]
    alive = s1["alive"] == 1
    print("S3: {} touches, last-touch team known on {:.1f}% of alive frames".format(
        len(table), 100.0 * (arrays["last_touch_team"][alive] >= 0).mean() if alive.any() else 0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR)
    args = parser.parse_args()
    for path in run(args.stages):
        print("S3: wrote {}".format(os.path.relpath(path, C.ROOT)))


if __name__ == "__main__":
    main()
