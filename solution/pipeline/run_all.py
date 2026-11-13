"""Run the whole pipeline: provided data in, validated submission CSV out.

    python -m solution.pipeline.run_all --out evaluation/results/<team_name>.csv
    python -m solution.pipeline.run_all --from 3        # reuse the S0-S2 files already on disk

This is the reproduction command for the submission. main must always be able to run it.
"""

import argparse
import os
import sys

from solution.pipeline import contracts as C
from solution.pipeline.s0_tracking import run as s0
from solution.pipeline.s1_inplay import run as s1
from solution.pipeline.s2_ball import run as s2
from solution.pipeline.s3_touch import run as s3
from solution.pipeline.s4_possession import run as s4

sys.path.insert(0, os.path.join(C.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def run(stage_dir=None, out_path=None, data_dir=None, n_frames=None, blocks=None, start=0):
    """Run stages start..4 and return the submission path. Raises if the file is not valid."""
    out_path = out_path or s4.DEFAULT_OUT
    if start <= 0:
        s0.run(stage_dir, data_dir=data_dir, n_frames=n_frames, blocks=blocks)
    if start <= 1:
        s1.run(stage_dir)
    if start <= 2:
        s2.run(stage_dir)
    if start <= 3:
        s3.run(stage_dir)
    s4.run(stage_dir, out_path)

    problem = pm.validate_submission(pm.read_label_csv(out_path), n_frames or C.N_FRAMES)
    if problem:
        raise C.ContractError("{} is not a valid submission: {}".format(out_path, problem))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR, help="stage file folder")
    parser.add_argument("--out", default=s4.DEFAULT_OUT, help="submission CSV to write")
    parser.add_argument("--data-dir", default=None, help="match bundle (default: games.DATA_DIR)")
    parser.add_argument("--from", dest="start", type=int, default=0, choices=range(5),
                        help="first stage to run; earlier stage files are reused from --stages")
    args = parser.parse_args()
    path = run(args.stages, args.out, data_dir=args.data_dir, start=args.start)
    print("\nvalid submission: {}".format(os.path.relpath(path, C.ROOT)))
    print("score it:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
