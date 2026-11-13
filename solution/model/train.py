"""Train the possession model on external matches only, with grouped cross-validation.

    python -m solution.model.train --cv  --run cv1        # 4 folds -> out-of-fold evidence
    python -m solution.model.train --final --out solution/models/possession.pt

Training data: every dev match (solution.bench.evaluate.dev_matches) under several randomised
degradations (domain randomisation). Validation: the held-out fold under the fixed benchmark
degradation (seed 0), exactly what bench.evaluate scores. No BAR-ATH data is read here.
"""

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

from solution.pipeline import contracts as C
from solution.bench import external
from solution.bench.evaluate import degraded, dev_matches, reference_labels
from solution.model import features as FT
from solution.model.net import PossessionNet

FEAT_DIR = os.path.join(external.CACHE, "feat")
OOF_DIR = os.path.join(external.CACHE, "oof")
FOLDS = [("DFL-J03WMX", "MET-1"), ("DFL-J03WN1", "MET-2"), ("DFL-J03WOH", "MET-3"),
         ("DFL-J03WOY", "DFL-J03WPY")]
TRAIN_SEEDS = (1, 2, 3, 4)
CHANGE_RADIUS = 5
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CONFIG = dict(hidden=128, layers=2, dropout=0.1, kernel=9, steps=3000, batch=32, chunk=1024,
              lr=2e-3, weight_decay=1e-4, change_weight=0.5, swap_p=0.5, seed=0)


# -- data ----------------------------------------------------------------------------------------

def feature_path(name, seed, randomize):
    tag = "r" if randomize else "d"
    return os.path.join(FEAT_DIR, "{}_{}{}_f{}.npz".format(name, tag, seed, FT.VERSION))


def build_features(job):
    name, seed, randomize = job
    path = feature_path(name, seed, randomize)
    if not os.path.exists(path):
        os.makedirs(FEAT_DIR, exist_ok=True)
        s0, gt = degraded(name, seed, randomize)
        X = FT.frame_features(s0)
        np.savez_compressed(path, X=X.astype(np.float16), gt=gt,
                            segments=np.array(FT.segments(s0), dtype=np.int64))
    return path


def ensure_features(jobs, workers=6):
    todo = [j for j in jobs if not os.path.exists(feature_path(*j))]
    if todo:
        print("building {} feature files ...".format(len(todo)), flush=True)
        with get_context("spawn").Pool(min(workers, len(todo))) as pool:
            pool.map(build_features, todo)


def load_segments(name, seed, randomize):
    with np.load(feature_path(name, seed, randomize)) as d:
        X, gt, segs = d["X"].astype(np.float32), reference_labels(name, d["gt"]), d["segments"]
    return [(X[s:e], gt[s:e].astype(np.int64)) for s, e in segs]


def change_target(y):
    c = np.zeros(len(y), dtype=np.float32)
    idx = np.flatnonzero(y[1:] != y[:-1]) + 1
    for i in idx:
        c[max(0, i - CHANGE_RADIUS):i + CHANGE_RADIUS + 1] = 1.0
    return c


# -- training ------------------------------------------------------------------------------------

def _norm_stats(segments):
    X = np.concatenate([x[::7] for x, _ in segments])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train(segments, cfg=None, device="cuda", log_every=500):
    cfg = dict(CONFIG, **(cfg or {}))
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    mean, std = _norm_stats(segments)
    changes = [change_target(y) for _, y in segments]
    lengths = np.array([len(y) for _, y in segments], dtype=np.float64)
    weights = lengths / lengths.sum()
    n_feat = segments[0][0].shape[1]
    model = PossessionNet(n_feat, cfg["hidden"], cfg["layers"], cfg["dropout"],
                          cfg["kernel"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"],
                                                pct_start=0.1)
    L, B = cfg["chunk"], cfg["batch"]
    mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    s0_sl, s1_sl = FT.team_slices()
    t0 = time.time()
    for step in range(cfg["steps"]):
        xb = np.zeros((B, L, n_feat), dtype=np.float32)
        yb = np.full((B, L), -100, dtype=np.int64)
        cb = np.zeros((B, L), dtype=np.float32)
        mb = np.zeros((B, L), dtype=np.float32)
        for b in range(B):
            k = rng.choice(len(segments), p=weights)
            x, y = segments[k]
            T = len(y)
            s = int(rng.integers(0, max(1, T - L)))
            e = min(T, s + L)
            xs, ys = x[s:e], y[s:e].copy()
            if rng.random() < cfg["swap_p"]:
                xs = FT.swap_teams(xs)
                ys = np.where(ys == C.H, C.A, np.where(ys == C.A, C.H, ys))
            n = e - s
            xb[b, :n], cb[b, :n], mb[b, :n] = xs, changes[k][s:e], 1.0
            yb[b, :n] = np.where(ys <= C.D, ys, -100)
        x_t = (torch.from_numpy(xb).to(device) - mean_t) / std_t
        y_t, c_t, m_t = (torch.from_numpy(a).to(device) for a in (yb, cb, mb))
        logits, change = model(x_t)
        loss_state = F.cross_entropy(logits.float().reshape(-1, 3), y_t.reshape(-1),
                                     ignore_index=-100)
        loss_change = (F.binary_cross_entropy_with_logits(change.float(), c_t, reduction="none",
                                                          pos_weight=torch.tensor(4.0,
                                                                                  device=device))
                       * m_t).sum() / m_t.sum()
        loss = loss_state + cfg["change_weight"] * loss_change
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            print("  step {:5d}  loss {:.4f} (state {:.4f}, change {:.4f})  {:.0f}s".format(
                step + 1, loss.item(), loss_state.item(), loss_change.item(), time.time() - t0),
                flush=True)
    model.eval()
    return model, mean, std, cfg


WINDOW, CONTEXT = 8192, 1024   # inference windows; cuDNN rejects very long single sequences


@torch.no_grad()
def predict_segment(model, mean, std, X, device="cuda"):
    """(log-probs (T,3) float32, change logit (T,) float32) for one contiguous stretch.

    Run in overlapping windows (each output frame sees >= CONTEXT frames either side where they
    exist), so the result does not depend on how long the stretch is.
    """
    T = len(X)
    x_all = (torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
             - torch.tensor(mean, device=device)) / torch.tensor(std, device=device)
    logp = np.empty((T, 3), dtype=np.float32)
    change = np.empty(T, dtype=np.float32)
    for start in range(0, T, WINDOW):
        end = min(T, start + WINDOW)
        a, b = max(0, start - CONTEXT), min(T, end + CONTEXT)
        logits, ch = model(x_all[a:b][None].contiguous())
        lp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        logp[start:end] = lp[start - a:end - a]
        change[start:end] = ch[0].float().cpu().numpy()[start - a:end - a]
    return logp, change


def save_checkpoint(path, models, mean, std, cfg, decoder, train_names, log_prior):
    """One file holding every ensemble member (their probabilities are averaged at inference)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    models = models if isinstance(models, (list, tuple)) else [models]
    torch.save({
        "members": [{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
        "net": models[0].config, "mean": mean, "std": std, "train_config": cfg,
        "feature_names": FT.names(), "feature_version": FT.VERSION, "decoder": decoder,
        "log_prior": log_prior, "train_matches": list(train_names),
        "train_seeds": list(TRAIN_SEEDS), "git_sha": C.git_sha(),
        "data_sources": ["DFL open data, figshare 28196177, CC BY 4.0",
                         "Metrica Sports sample-data, github.com/metrica-sports/sample-data"],
    }, path)


def class_log_prior(segments):
    y = np.concatenate([s[1] for s in segments])
    y = y[y <= C.D]
    counts = np.bincount(y, minlength=3).astype(np.float64) + 1
    return np.log(counts / counts.sum()).astype(np.float32)


# -- cross-validation ----------------------------------------------------------------------------

def predict_members(models, mean, std, X, device="cuda", swap_tta=False):
    """Ensemble: log of the mean probability, mean change logit.

    swap_tta: also predict on the team-swapped frames and swap H/A back (the model should be
    team-symmetric; averaging both views removes any residual asymmetry).
    """
    outs = [predict_segment(m, mean, std, X, device) for m in models]
    if swap_tta:
        Xs = FT.swap_teams(X)
        for m in models:
            lp, ch = predict_segment(m, mean, std, Xs, device)
            outs.append((lp[:, [1, 0, 2]], ch))
    if len(outs) == 1:
        return outs[0]
    p = np.mean([np.exp(lp) for lp, _ in outs], axis=0)
    return np.log(np.maximum(p, 1e-12)).astype(np.float32), \
        np.mean([ch for _, ch in outs], axis=0).astype(np.float32)


def train_members(segs, cfg, device, members=1):
    models = []
    for i in range(members):
        c = dict(cfg or {})
        c["seed"] = int(c.get("seed", CONFIG["seed"])) + i
        model, mean, std, used = train(segs, c, device)
        models.append(model)
    return models, mean, std, used


def run_cv(run, cfg=None, device="cuda", members=1):
    names = dev_matches()
    folds = [[n for n in f if n in names] for f in FOLDS]
    ensure_features([(n, s, True) for n in names for s in TRAIN_SEEDS]
                    + [(n, 0, False) for n in names])
    out_dir = os.path.join(OOF_DIR, run)
    os.makedirs(out_dir, exist_ok=True)
    for k, held in enumerate(folds):
        train_names = [n for n in names if n not in held]
        print("fold {}: train {} | held {}".format(k, train_names, held), flush=True)
        segs = [seg for n in train_names for s in TRAIN_SEEDS for seg in load_segments(n, s, True)]
        models, mean, std, used = train_members(segs, cfg, device, members)
        save_checkpoint(os.path.join(out_dir, "fold{}.pt".format(k)), models, mean, std, used, {},
                        train_names, class_log_prior(segs))
        for name in held:
            with np.load(feature_path(name, 0, False)) as d:
                X, bounds = d["X"].astype(np.float32), d["segments"]
                gt = reference_labels(name, d["gt"])
            logp = np.full((len(gt), 3), np.nan, dtype=np.float32)
            change = np.zeros(len(gt), dtype=np.float32)
            for s, e in bounds:
                logp[s:e], change[s:e] = predict_members(models, mean, std, X[s:e], device)
            np.savez_compressed(os.path.join(out_dir, name + ".npz"), logp=logp, change=change,
                                gt=gt, segments=bounds, log_prior=class_log_prior(segs))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        from solution.bench import evaluate as EV
        json.dump(dict(CONFIG, **(cfg or {}), convention=EV.CONVENTION), f, indent=1)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--run", default="cv")
    parser.add_argument("--out", default=os.path.join(C.ROOT, "solution", "models",
                                                      "possession.pt"))
    parser.add_argument("--decoder", default=None, help="JSON file of tuned decoder params")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--members", type=int, default=1, help="ensemble size (seeds)")
    parser.add_argument("--with-holdout", action="store_true",
                        help="final model only: also train on the two held-out matches")
    args = parser.parse_args()
    cfg = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(CONFIG[k])(v)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cv:
        print("out-of-fold evidence in", run_cv(args.run, cfg, device, args.members))
    if args.final:
        from solution.bench.evaluate import HOLDOUT
        names = dev_matches() + (list(HOLDOUT) if args.with_holdout else [])
        ensure_features([(n, s, True) for n in names for s in TRAIN_SEEDS])
        segs = [seg for n in names for s in TRAIN_SEEDS for seg in load_segments(n, s, True)]
        models, mean, std, used = train_members(segs, cfg, device, args.members)
        decoder = json.load(open(args.decoder)) if args.decoder else {}
        save_checkpoint(args.out, models, mean, std, used, decoder, names, class_log_prior(segs))
        print("saved", args.out)


if __name__ == "__main__":
    main()
