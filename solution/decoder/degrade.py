"""Make clean (corrected) tracking look like the challenge's automatic tracking.

DATA.md: the ball is missing on ~41% of graded frames, disproportionately when it matters --
crowded areas, balls in the air, out of play. About 20.5 of 22 players are tracked on average, with
fragmented tracks. This module imitates those properties with a fixed seed so every evaluation run
sees the same degraded data.
"""

import numpy as np

from .match import Half, Match


def _bursts(rng, T, p_on, p_off, start_on=False):
    """Boolean mask from a two-state Markov chain; True = dropped. Mean burst = 1/p_off frames."""
    u = rng.random(T)
    out = np.empty(T, dtype=bool)
    state = start_on
    for t in range(T):
        state = (u[t] >= p_off) if state else (u[t] < p_on)
        out[t] = state
    return out


def _ball_dropout(rng, half, pitch):
    """Ball missing mask with a hazard that rises when crowded, airborne or off the pitch."""
    T = half.n
    ball, players = half.ball, half.players
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    crowd = np.nansum(d < 3.0, axis=1)
    z = getattr(half, "z", np.zeros(T))
    z = np.nan_to_num(z)
    out = (np.abs(ball[:, 0]) > pitch[0] / 2) | (np.abs(ball[:, 1]) > pitch[1] / 2)
    hazard = 0.0085 * (1 + 0.8 * np.clip(crowd - 1, 0, 4)) * (1 + 2.0 * (z > 1.8)) * (1 + 4 * out)
    u = rng.random(T)
    recover = 1 / 40.0  # mean gap ~1.6 s
    mask = np.empty(T, dtype=bool)
    state = False
    for t in range(T):
        state = (u[t] >= recover) if state else (u[t] < hazard[t])
        mask[t] = state
    return mask


def degrade(match, seed=0, ball_jitter=0.3, player_jitter=0.25):
    rng = np.random.default_rng(seed)
    halves = []
    for half in match.halves:
        T, P = half.n, half.players.shape[1]
        ball = half.ball + rng.normal(0, ball_jitter, half.ball.shape)
        ball[_ball_dropout(rng, half, match.pitch)] = np.nan
        players = half.players + rng.normal(0, player_jitter, half.players.shape)
        for j in range(P):
            players[_bursts(rng, T, 1 / 600.0, 1 / 50.0), j] = np.nan   # ~7% lost in ~2 s bursts
        # Identity switches: a slot briefly carries the other team's label. Modelled by adding
        # a ghost copy of a player on the wrong team for a short burst.
        ghost = np.full((T, 2, 2), np.nan)
        ghost_team = []
        for g in range(2):
            j = int(rng.integers(P))
            on = _bursts(rng, T, 1 / 3000.0, 1 / 60.0)
            ghost[on, g] = half.players[on, j]
            ghost_team.append(1 - half.team[j])
        players = np.concatenate([players, ghost], axis=1)
        team = np.concatenate([half.team, ghost_team]).astype(np.int8)
        out = Half(ball=ball, players=players, team=team, start=half.start, labels=half.labels)
        halves.append(out)
    return Match(name=match.name, pitch=match.pitch, halves=halves)


def missing_rate(match):
    graded = np.concatenate([h.labels != 3 for h in match.halves])
    miss = np.concatenate([np.isnan(h.ball[:, 0]) for h in match.halves])
    return float(miss[graded].mean())
