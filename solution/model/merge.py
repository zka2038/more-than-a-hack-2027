"""Merge checkpoints into one ensemble checkpoint.

Training members as separate processes (``train --final --members 1 --set seed=K``) lets them run
in parallel on machines without a GPU. Members from the same training run share their
normalisation statistics and the result has the usual single-run format. Members from different
runs (other training matches or degradation profile) keep their own normalisation
(``member_mean``/``member_std``, used by ``infer.evidence``) and are described per member in
``member_info``. In both cases the features, network shape and decoder must agree, and at
inference the members' probabilities are averaged with equal weight.

    python -m solution.model.merge --out solution/models/possession.pt a.pt b.pt c.pt
"""
import argparse

import numpy as np
import torch

SAME = ("net", "feature_names", "feature_version", "decoder")
RUN = ("train_matches", "train_seeds", "degradation_profile")


def _members(ck):
    """(state_dict, mean, std, info) per member, whatever format ck is in."""
    n = len(ck["members"])
    means = ck.get("member_mean") or [ck["mean"]] * n
    stds = ck.get("member_std") or [ck["std"]] * n
    infos = ck.get("member_info") or [
        {"seed": int(ck["train_config"]["seed"]) - (n - 1 - i),   # train_members: seed + i
         "steps": int(ck["train_config"]["steps"]),
         "train_matches": list(ck["train_matches"]), "train_seeds": list(ck["train_seeds"]),
         "degradation_profile": int(ck.get("degradation_profile", 1))} for i in range(n)]
    return list(zip(ck["members"], means, stds, infos))


def merge(paths):
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    base = dict(ckpts[0])
    for p, c in zip(paths[1:], ckpts[1:]):
        for k in SAME:
            if c.get(k) != base.get(k):
                raise ValueError("{}: '{}' differs from {}".format(p, k, paths[0]))
    members = [m for c in ckpts for m in _members(c)]
    keys = [(i["seed"], i["degradation_profile"], tuple(i["train_matches"])) for *_, i in members]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate members (same seed, profile and training matches)")
    same_run = (all(np.array_equal(m[1], members[0][1]) and np.array_equal(m[2], members[0][2])
                    for m in members)
                and all(c.get(k) == base.get(k) for c in ckpts for k in RUN))
    if not same_run and any(not np.array_equal(c["log_prior"], base["log_prior"]) for c in ckpts) \
            and float((base.get("decoder") or {}).get("prior", 0.0)) != 0.0:
        raise ValueError("runs differ in class prior and the decoder uses it (prior != 0)")
    base["members"] = [m[0] for m in members]
    base["member_info"] = [m[3] for m in members]
    for k in ("member_mean", "member_std"):
        base.pop(k, None)
    if not same_run:
        base["member_mean"] = [np.asarray(m[1], dtype=np.float32) for m in members]
        base["member_std"] = [np.asarray(m[2], dtype=np.float32) for m in members]
        base["train_matches"] = sorted({n for *_, i in members for n in i["train_matches"]})
        base["degradation_profile"] = sorted({i["degradation_profile"] for *_, i in members})
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    merged = merge(args.inputs)
    torch.save(merged, args.out)
    print("merged {} members into {} ({}): {}".format(
        len(merged["members"]), args.out,
        "per-member normalisation" if "member_mean" in merged else "one training run",
        ", ".join("seed {} profile {} on {} matches".format(
            i["seed"], i["degradation_profile"], len(i["train_matches"]))
            for i in merged["member_info"])))


if __name__ == "__main__":
    main()
