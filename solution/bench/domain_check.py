"""Compare the input statistics of BAR-ATH's tracking with the degraded external matches.

    python -m solution.bench.domain_check                 # external matches only
    python -m solution.bench.domain_check --stages DIR    # + BAR-ATH, from s0_tracking.npz

Only aggregate statistics of the *input* tracking are printed: no labels, no ground truth, no
per-frame values. It answers one question: does the simulated degradation look like the real
automatic tracking? If it does not, benchmark scores are optimistic and the degradation should be
revisited. Run the BAR-ATH part yourself (PLAN.md, Hard rule 5); only this table goes back to an
agent.
"""

import argparse

import numpy as np

from solution.pipeline import contracts as C
from solution.bench.evaluate import degraded, dev_matches


def stats(s0):
    graded = s0["half"] > 0
    ball = s0["ball_xy"][graded]
    players = s0["player_xy"][graded]
    team = s0["player_team"]
    seen = np.isfinite(ball[:, 0])
    gaps = np.array([e - s for s, e in C.runs(~seen)])
    L, W = C.PITCH
    off = seen & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))
    vis = np.isfinite(players[:, :, 0])
    speed = s0["player_speed"][graded]
    sp = speed[np.isfinite(speed)]
    # Crowding around the last seen ball: how often is the ball missing when players are close?
    d = np.linalg.norm(players - np.where(seen[:, None], ball, np.nan)[:, None, :], axis=2)
    near = np.nansum(d < 3.0, axis=1)
    return {
        "frames": int(graded.sum()),
        "slots": int(players.shape[1]),
        "ball_missing_%": 100 * (1 - seen.mean()),
        "gaps_per_min": len(gaps) / (graded.sum() / 1500.0),
        "gap_p50_frames": float(np.percentile(gaps, 50)) if len(gaps) else 0.0,
        "gap_p90_frames": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
        "gap_over_6s_%": 100 * float((gaps > 150).sum() / max(len(gaps), 1)),
        "ball_off_pitch_%": 100 * off.mean(),
        "visible_team0": float(vis[:, team == 0].sum(axis=1).mean()),
        "visible_team1": float(vis[:, team == 1].sum(axis=1).mean()),
        "visible_total": float(vis.sum(axis=1).mean()),
        "speed_p50": float(np.percentile(sp, 50)) if len(sp) else float("nan"),
        "speed_p90": float(np.percentile(sp, 90)) if len(sp) else float("nan"),
        "players_within_3m_of_ball": float(near[seen].mean()) if seen.any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=None, help="folder with BAR-ATH's s0_tracking.npz")
    args = parser.parse_args()
    columns = {}
    for name in dev_matches():
        columns[name] = stats(degraded(name, 0)[0])
    if args.stages:
        s0 = C.load("s0_tracking", args.stages)
        columns["BAR-ATH"] = stats(s0)
    keys = list(next(iter(columns.values())))
    print("{:28s}".format("statistic") + "".join("{:>12s}".format(n[-10:]) for n in columns))
    for k in keys:
        print("{:28s}".format(k) + "".join("{:12.2f}".format(columns[n][k]) for n in columns))


if __name__ == "__main__":
    main()
