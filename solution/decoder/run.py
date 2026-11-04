"""Produce the challenge submission from the provided tactical tracking.

    .venv/bin/python -m solution.decoder.run [--out outputs/decoder_submission.csv]

Weights come from params.json, which is fitted on the DFL open matches only (tune.py --fit-all)
and never on the challenge match. The output is deterministic: same data in, same file out.
"""

import argparse
import json
import os

from . import atd
from .pipeline import DEFAULTS, predict
from .write import run_official_validator, to_labels, write_submission

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "params.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(atd.ROOT, "outputs", "decoder_submission.csv"))
    ap.add_argument("--params", default=PARAMS)
    args = ap.parse_args()

    params = dict(DEFAULTS)
    if os.path.exists(args.params):
        with open(args.params) as handle:
            params.update(json.load(handle))
    match, n_frames = atd.load()
    codes = predict(match, params)
    labels = to_labels(n_frames, match.halves, codes)
    path = write_submission(labels, args.out, n_frames)
    run_official_validator(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
