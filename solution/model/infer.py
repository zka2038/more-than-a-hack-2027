"""Run a trained possession model on any s0-shaped match (BAR-ATH or external).

    logp, change = evidence(s0, model_path)       # (N,3) log P(H/A/D), (N,) change logit
    codes = predict(s0, model_path)               # (N,) H/A/D codes, X outside the halves

Reproducibility (PLAN.md, Hard rule 7): inference runs on CPU with deterministic algorithms and a
fixed thread count, and the log-probabilities are rounded before decoding, so float noise between
machines cannot move a decoded boundary.
"""

import os

import numpy as np
import torch

from solution.pipeline import contracts as C
from solution.model import decode as Dc
from solution.model import features as FT
from solution.model.net import PossessionNet
from solution.model.train import predict_segment

DEFAULT_MODEL = os.path.join(C.ROOT, "solution", "models", "possession.pt")
ROUND = 3            # decimals kept on log-probabilities before decoding
_LOADED = {}


def load(path=None):
    path = os.path.abspath(path or DEFAULT_MODEL)
    if path not in _LOADED:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["feature_version"] != FT.VERSION or ck["feature_names"] != FT.names():
            raise ValueError("{} was trained on feature version {}, code has {}".format(
                path, ck["feature_version"], FT.VERSION))
        model = PossessionNet(**ck["net"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _LOADED[path] = (model, ck)
    return _LOADED[path]


def evidence(s0, model_path=None, device="cpu", ball_xy=None):
    """(logp (N,3) float32 with NaN outside the halves, change logit (N,) float32)."""
    model, ck = load(model_path)
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
    model = model.to(device)
    X = FT.frame_features(s0, ball_xy)
    N = len(s0["half"])
    logp = np.full((N, 3), np.nan, dtype=np.float32)
    change = np.zeros(N, dtype=np.float32)
    for s, e in FT.segments(s0):
        lp, ch = predict_segment(model, ck["mean"], ck["std"], X[s:e], device)
        logp[s:e] = np.round(lp, ROUND)
        change[s:e] = np.round(ch, ROUND)
    model.to("cpu")
    return logp, change


def decode(s0, logp, change, params=None, model_path=None):
    _, ck = load(model_path)
    p = dict(ck.get("decoder") or {})
    p.update(params or {})
    codes = np.full(len(s0["half"]), C.X, dtype=np.int8)
    for s, e in FT.segments(s0):
        codes[s:e] = Dc.decode_segment(logp[s:e], change[s:e], p, ck.get("log_prior"))
    return codes


def predict(s0, model_path=None, params=None, device="cpu"):
    logp, change = evidence(s0, model_path, device)
    return decode(s0, logp, change, params, model_path)
