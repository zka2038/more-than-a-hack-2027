"""Turn per-frame state log-probabilities into one H/A/D sequence with few, well-placed changes.

Exact Viterbi over three states. A change of state pays a cost, so the path switches only when the
summed evidence for the new state outweighs it: spellF1 scores whole possessions, and every
flicker splits one. The H<->A cost can be lowered where the model's change head fires, so real
hand-overs are cheap and noise is expensive.
"""

import numpy as np

H, A, D = 0, 1, 2

DEFAULTS = dict(
    temp=1.0,        # emission temperature (log-probs are divided by it)
    c_sw=6.0,        # H <-> A
    c_in=8.0,        # alive -> D
    c_out=8.0,       # D -> alive
    beta=0.0,        # how much a change-head logit lowers every switch cost
    prior=0.0,       # 1 = divide posteriors by the training class prior (likelihood emissions)
)


try:
    from numba import njit
except ImportError:          # pure Python fallback: same result, ~100x slower
    def njit(*args, **kwargs):
        return (lambda f: f) if not (len(args) == 1 and callable(args[0])) else args[0]


def viterbi(E, cost_sw, cost_in, cost_out):
    """Most likely path. E (T,3) scores; costs are scalars or (T,) arrays applied at frame t."""
    T = len(E)
    cs = np.ascontiguousarray(np.broadcast_to(np.asarray(cost_sw, dtype=np.float64), (T,)))
    ci = np.ascontiguousarray(np.broadcast_to(np.asarray(cost_in, dtype=np.float64), (T,)))
    co = np.ascontiguousarray(np.broadcast_to(np.asarray(cost_out, dtype=np.float64), (T,)))
    return _viterbi(np.ascontiguousarray(E, dtype=np.float64), cs, ci, co)


@njit(cache=True)
def _viterbi(El, cs, ci, co):
    T = El.shape[0]
    back = np.zeros((T, 3), dtype=np.int8)
    h, a, d = El[0, 0], El[0, 1], El[0, 2]
    for t in range(1, T):
        sw, ii, oo = cs[t], ci[t], co[t]
        eh, ea, ed = El[t, 0], El[t, 1], El[t, 2]
        # into H: from H, A (switch), D (out)
        bh, ah = h, 0
        x = a - sw
        if x > bh:
            bh, ah = x, 1
        x = d - oo
        if x > bh:
            bh, ah = x, 2
        ba, aa = a, 1
        x = h - sw
        if x > ba:
            ba, aa = x, 0
        x = d - oo
        if x > ba:
            ba, aa = x, 2
        bd, ad = d, 2
        x = h - ii
        if x > bd:
            bd, ad = x, 0
        x = a - ii
        if x > bd:
            bd, ad = x, 1
        back[t, 0], back[t, 1], back[t, 2] = ah, aa, ad
        h, a, d = bh + eh, ba + ea, bd + ed
    path = np.empty(T, dtype=np.int8)
    last = 0
    if a > h and a >= d:
        last = 1
    elif d > h and d > a:
        last = 2
    path[T - 1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_segment(logp, change_logit=None, params=None, log_prior=None):
    """H/A/D codes for one contiguous stretch. logp (T,3) natural-log posteriors."""
    p = dict(DEFAULTS)
    p.update(params or {})
    E = np.asarray(logp, dtype=np.float64)
    if p["prior"] and log_prior is not None:
        E = E - p["prior"] * np.asarray(log_prior)[None, :]
    E = E / max(p["temp"], 1e-6)
    sw, ci, co = p["c_sw"], p["c_in"], p["c_out"]
    if p["beta"] and change_logit is not None:
        bonus = p["beta"] * np.clip(change_logit, -8, 8)
        sw = np.maximum(sw - bonus, 0.0)
        ci = np.maximum(ci - bonus, 0.0)
        co = np.maximum(co - bonus, 0.0)
    return viterbi(E, sw, ci, co)
