"""Merge single-member checkpoints into one ensemble checkpoint.

Training the members as separate processes (``train --final --members 1 --set seed=K``) lets
them run in parallel on machines without a GPU. The members must come from the same training
set, so their normalisation statistics, features, decoder and class prior are identical; this
is checked before merging. At inference the members' probabilities are averaged, exactly as
for an ensemble trained in one process.

    python -m solution.model.merge --out solution/models/possession.pt a.pt b.pt c.pt
"""
import argparse

import numpy as np
import torch

SAME = ("net", "feature_names", "feature_version", "decoder", "train_matches", "train_seeds",
        "degradation_profile", "data_sources")
SAME_ARRAYS = ("mean", "std", "log_prior")


def merge(paths):
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    base = dict(ckpts[0])
    for p, c in zip(paths[1:], ckpts[1:]):
        for k in SAME:
            if c.get(k) != base.get(k):
                raise ValueError("{}: '{}' differs from {}".format(p, k, paths[0]))
        for k in SAME_ARRAYS:
            if not np.array_equal(np.asarray(c[k]), np.asarray(base[k])):
                raise ValueError("{}: '{}' differs from {}".format(p, k, paths[0]))
    base["members"] = [m for c in ckpts for m in c["members"]]
    base["member_seeds"] = [int(c["train_config"]["seed"]) for c in ckpts
                            for _ in c["members"]]
    if len(set(base["member_seeds"])) != len(base["member_seeds"]):
        raise ValueError("duplicate member seeds: {}".format(base["member_seeds"]))
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    merged = merge(args.inputs)
    torch.save(merged, args.out)
    print("merged {} members (seeds {}) into {}".format(
        len(merged["members"]), merged["member_seeds"], args.out))


if __name__ == "__main__":
    main()
