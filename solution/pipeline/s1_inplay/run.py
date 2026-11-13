"""S1: ball in play vs out of play. STUB: replace the body of decide(), keep the contract.

    python -m solution.pipeline.s1_inplay.run --stages outputs/stages

In:  s0_tracking.npz
Out: s1_inplay.npz (p_alive, alive), s1_stoppages.csv      -- PLAN.md section 2

Stub rule: the ball is dead where the raw track is outside the pitch, or missing for longer than
MAX_GAP frames. Dead runs shorter than MIN_DEAD are dropped and alive runs shorter than MIN_ALIVE
are absorbed, so the output is intervals rather than flicker. No restart type or team is inferred.
"""

import argparse
import os

import numpy as np
import pandas as pd

from solution.pipeline import contracts as C

# Params
MAX_GAP = 75        # frames; a ball unseen for longer than this is called dead
OUT_MARGIN = 0.5    # metres beyond a line before the ball counts as out
MIN_DEAD = 50       # frames; shorter stoppages are not real (PLAN.md section 0)
MIN_ALIVE = 25      # frames; shorter stretches of play between two stoppages are absorbed


def decide(s0):
    """p_alive (N,) for graded frames from S0. This is the function a real S1 replaces."""
    ball = s0["ball_xy"]
    seen = np.isfinite(ball[:, 0])
    length, width = C.PITCH
    with np.errstate(invalid="ignore"):
        out = seen & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN)
                      | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))
    dead = out.copy()
    for start, end in C.runs(~seen):
        if end - start > MAX_GAP:
            dead[start:end] = True
    return np.where(dead, 0.1, 0.9).astype(np.float32)


def to_intervals(p_alive, half):
    """Hard alive/dead per half with minimum durations. -1 on ungraded frames."""
    alive = np.full(len(half), -1, dtype=np.int8)
    for number in (1, 2):
        for first, last in C.runs(half == number):
            state = p_alive[first:last] >= 0.5
            for start, end in C.runs(~state):
                if end - start < MIN_DEAD:
                    state[start:end] = True
            for start, end in C.runs(state):
                # An alive sliver strictly between two stoppages; the edges of a half are kept.
                if end - start < MIN_ALIVE and start > 0 and end < len(state):
                    state[start:end] = False
            alive[first:last] = state
    return alive


def stoppage_table(alive, s0, team=None):
    """One row per dead interval. restart_team is the decoded team on the first alive frame."""
    rows = []
    ball = s0["ball_xy"]
    for stop_id, (start, end) in enumerate(C.runs(alive == 0)):
        restart = ball[end] if end < len(ball) else (np.nan, np.nan)
        restart_team = C.UNKNOWN
        if team is not None and end < len(alive) and alive[end] == 1:
            restart_team = int(team[end])
        rows.append({
            "stop_id": stop_id, "start_frame": start, "end_frame": end,
            "cause": "unknown", "restart_type": "unknown", "restart_team": restart_team,
            "restart_x": float(restart[0]), "restart_y": float(restart[1]), "conf": 0.5,
        })
    return pd.DataFrame(rows, columns=C.TABLE_SCHEMAS["s1_stoppages"])


def decide_model(s0, model_path=None):
    """(p_alive, alive, restart team per frame, kick-off frames) from the learned evidence.

    alive is the D/not-D part of the joint H/A/D Viterbi decode (solution/model/decode.py), with
    every frame of a half before its detected kick-off set dead (solution/model/rules.py).
    """
    from solution.model import infer, rules
    logp, change = infer.evidence(s0, model_path)
    codes = infer.decode(s0, logp, change, model_path=model_path)
    p_dead = np.exp(logp[:, C.D])
    codes, kickoffs = rules.apply_kickoffs(s0, codes, p_dead)
    alive = np.where(s0["half"] == 0, -1, codes != C.D).astype(np.int8)
    p_alive = np.where(s0["half"] == 0, np.nan, 1.0 - p_dead).astype(np.float32)
    team = np.where(codes <= C.A, codes, C.UNKNOWN).astype(np.int8)
    return p_alive, alive, team, kickoffs


def use_model(model_path=None):
    """True when a trained model is available (MTAH_MODEL=none forces the stub rules)."""
    from solution.model import infer
    return infer.available(model_path)


def run(stage_dir=None, model_path=None):
    s0 = C.load("s0_tracking", stage_dir)
    n = len(s0["half"])
    team = None
    if use_model(model_path):
        p_alive, alive, team, kickoffs = decide_model(s0, model_path)
        print("S1: learned model; kick-offs detected at {}".format(kickoffs))
    else:
        print("S1: no trained model found, using the stub rule")
        p_alive = decide(s0)
        alive = to_intervals(p_alive, s0["half"])
    p_alive[alive < 0] = np.nan
    # Keep the soft score on the same side as the hard decision after the duration rules.
    p_alive = np.where(alive == 1, np.maximum(p_alive, 0.5),
                       np.where(alive == 0, np.minimum(p_alive, 0.49), p_alive)).astype(np.float32)

    s1 = {"p_alive": p_alive, "alive": alive}
    C.check_s1_against_s0(s1, s0)
    table = stoppage_table(alive, s0, team)
    C.check_stoppages_against_s1(table, s1)
    params = ({"model": True} if team is not None else
              {"stub": True, "max_gap": MAX_GAP, "out_margin": OUT_MARGIN,
               "min_dead": MIN_DEAD, "min_alive": MIN_ALIVE})
    out = [C.save("s1_inplay", stage_dir, s1, params=params, n=n),
           C.save_table("s1_stoppages", stage_dir, table, n=n)]
    graded = alive >= 0
    print("S1: {} stoppages, {:.1f}% of graded frames dead".format(
        len(table), 100.0 * (alive[graded] == 0).mean() if graded.any() else 0.0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR)
    args = parser.parse_args()
    for path in run(args.stages):
        print("S1: wrote {}".format(os.path.relpath(path, C.ROOT)))


if __name__ == "__main__":
    main()
