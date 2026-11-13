"""S4: from the touch sequence to possession. STUB: replace decode(), keep the contract.

    python -m solution.pipeline.s4_possession.run --stages outputs/stages --out my_team.csv

In:  s0_tracking.npz (half only), s1_inplay.npz, s3_touch.npz
Out: s4_possession.npz (prob, label) and the submission CSV          -- PLAN.md section 5

Stub rule: D where S1 says dead, X on ungraded frames. On alive frames, a majority vote of the
last-touch team over +/- WINDOW frames inside the alive segment; frames with no vote take the
nearest decided label in the same segment, and a segment with no touch at all is written D. No
control gate, no backdating, no switch cost: those are the real S4.
"""

import argparse
import os

import numpy as np

from solution.pipeline import contracts as C

# Params
WINDOW = 50   # frames either side in the majority vote (the published baseline's +/- 2 s)
DEFAULT_OUT = os.path.join(C.ROOT, "evaluation", "results", "pipeline_stub.csv")


def decode(half, s1, s3):
    """(label (N,), prob (N,3)). This is the function a real S4 replaces."""
    n = len(half)
    label = np.full(n, C.X, dtype=np.uint8)
    label[s1["alive"] == 0] = C.D
    prob = np.full((n, 3), np.nan, dtype=np.float32)
    last = s3["last_touch_team"]

    for start, end in C.alive_segments(s1["alive"]):
        seg = last[start:end]
        home = np.concatenate(([0], np.cumsum(seg == C.HOME)))
        away = np.concatenate(([0], np.cumsum(seg == C.AWAY)))
        index = np.arange(end - start)
        low = np.maximum(0, index - WINDOW)
        high = np.minimum(end - start, index + WINDOW + 1)
        votes_home = home[high] - home[low]
        votes_away = away[high] - away[low]
        total = votes_home + votes_away
        if not total.any():
            label[start:end] = C.D      # nobody ever on the ball: treat as part of the stoppage
            continue

        decided = np.flatnonzero(total > 0)
        # Nearest decided frame for the undecided ones: forward-fill, then back-fill the head.
        carry = np.maximum.accumulate(np.where(total > 0, index, -1))
        carry[carry < 0] = decided[0]
        share = (votes_home[carry] / total[carry].astype(np.float64))
        team = np.where(share >= 0.5, C.H, C.A)     # ties to home, as the baseline does
        label[start:end] = team
        prob[start:end, C.H] = share
        prob[start:end, C.A] = 1.0 - share
        prob[start:end, C.D] = 0.0

    dead = label == C.D
    prob[dead] = (0.0, 0.0, 1.0)
    # Blend with S1's soft score so prob is a distribution over all three states.
    p_alive = np.nan_to_num(s1["p_alive"], nan=0.0)
    alive = (label == C.H) | (label == C.A)
    prob[alive, :2] *= p_alive[alive, None]
    prob[alive, C.D] = 1.0 - p_alive[alive]
    return label, prob


def decode_model(half, s1, s3, params=None, model_path=None):
    """(label, prob) from S1's dead intervals and the model evidence S3 carries.

    Inside every alive segment, H/A is the Viterbi path over S3's log P(H), log P(A) with the
    tuned switch cost, lowered where the change probability is high. With D pinned to S1 (which
    came from the same joint decode) this reproduces the joint H/A/D path exactly.
    """
    from solution.model import decode as Dc, infer
    _, ck = infer.load(model_path)
    p = dict(ck.get("decoder") or {})
    p.update(params or {})
    n = len(half)
    label = np.full(n, C.X, dtype=np.uint8)
    label[s1["alive"] == 0] = C.D
    p_change = np.clip(s3["control"].sum(axis=1).astype(np.float64), 1e-6, 1 - 1e-6)
    change_logit = np.log(p_change / (1 - p_change))
    for start, end in C.alive_segments(s1["alive"]):
        E = np.column_stack([s3["touch_ll"][start:end, 0], s3["touch_ll"][start:end, 1],
                             np.full(end - start, -1e6)])
        label[start:end] = Dc.decode_segment(E, change_logit[start:end], p, ck.get("log_prior"))
    prob = np.full((n, 3), np.nan, dtype=np.float32)
    graded = label != C.X
    ph = np.exp(s3["touch_ll"][:, 0].astype(np.float64))
    pa = np.exp(s3["touch_ll"][:, 1].astype(np.float64))
    share = ph / np.maximum(ph + pa, 1e-12)
    p_alive = np.clip(np.nan_to_num(s1["p_alive"], nan=0.0), 0.0, 1.0)
    prob[graded, C.H] = (p_alive * share)[graded]
    prob[graded, C.A] = (p_alive * (1.0 - share))[graded]
    prob[graded, C.D] = (1.0 - p_alive)[graded]
    return label, prob


def run(stage_dir=None, out_path=None, model_path=None):
    s0 = C.load("s0_tracking", stage_dir)
    s1 = C.load("s1_inplay", stage_dir)
    s3 = C.load("s3_touch", stage_dir)
    n = len(s0["half"])
    if s3["meta"].get("params", {}).get("model"):
        label, prob = decode_model(s0["half"], s1, s3, model_path=model_path)
        params = {"model": True}
    else:
        label, prob = decode(s0["half"], s1, s3)
        params = {"stub": True, "window": WINDOW}
    s4 = {"prob": prob, "label": label}
    C.check_s4_against_s0(s4, s0)
    out = [C.save("s4_possession", stage_dir, s4, n=n, params=params),
           C.write_submission(out_path or DEFAULT_OUT, label)]

    spells = sum(len(C.runs(label == team)) for team in (C.H, C.A))
    alive = int(((label == C.H) | (label == C.A)).sum())
    print("S4: {} possessions, mean {:.1f} s, H {} A {} D {} X {}".format(
        spells, alive / float(C.FPS * spells) if spells else 0.0,
        *(int((label == code).sum()) for code in (C.H, C.A, C.D, C.X))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT, help="submission CSV to write")
    args = parser.parse_args()
    for path in run(args.stages, args.out):
        print("S4: wrote {}".format(os.path.relpath(path, C.ROOT)))


if __name__ == "__main__":
    main()
