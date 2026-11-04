"""Behavioural checks for the no-training decoder on small synthetic sequences.

    .venv/bin/python tests/test_decoder.py          (or: python -m pytest tests/test_decoder.py)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from solution.decoder.match import Half, Match  # noqa: E402
from solution.decoder.pipeline import predict  # noqa: E402
from solution.decoder.viterbi import viterbi  # noqa: E402
from solution.decoder.write import to_labels, write_submission  # noqa: E402

PITCH = (105.0, 68.0)
FPS = 25


def scene(segments, n_players=11):
    """Build a half from [(owner team or 'dead', seconds)]: the ball sits on one player of the
    owning team, who walks across the pitch; the other team stands 6 m away."""
    T = sum(int(s * FPS) for _, s in segments)
    ball = np.zeros((T, 2))
    players = np.zeros((T, 2 * n_players, 2))
    team = np.array([0] * n_players + [1] * n_players, dtype=np.int8)
    rng = np.random.default_rng(0)
    base = rng.uniform([-40, -25], [40, 25], size=(2 * n_players, 2))
    t = 0
    for owner, seconds in segments:
        n = int(seconds * FPS)
        steps = np.arange(n)[:, None] * np.array([0.2, 0.0])  # 5 m/s
        players[t:t + n] = base[None] + rng.normal(0, 0.05, (n, 2 * n_players, 2))
        players[t:t + n] += steps[:, None, :] * 0.4           # everyone moves while alive
        if owner == "dead":
            players[t:t + n] = base[None]                     # standing still
            ball[t:t + n] = (PITCH[0] / 2 + 3, 0)             # out of play
        else:
            j = 0 if owner == 0 else n_players
            ball[t:t + n] = players[t:t + n, j] + 0.3
            other = n_players if owner == 0 else 0
            players[t:t + n, other] = ball[t:t + n] + 6.0
        t += n
    return Half(ball=ball, players=players, team=team)


def run(half):
    return predict(Match(name="synthetic", pitch=PITCH, halves=[half]))[0]


def n_spells(codes):
    alive = codes < 2
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    starts = np.concatenate(([0], change))
    return int(sum(alive[s] for s in starts))


def test_viterbi_prefers_fewer_switches():
    E = np.zeros((100, 3))
    E[:, 0] = 1.0
    E[48:52, 1] = 3.0  # a short blip of contrary evidence
    path = viterbi(E, np.array([[0, 20, 20], [20, 0, 20], [20, 20, 0]], dtype=float))
    assert (path == 0).all()


def test_missing_ball_does_not_split_a_possession():
    half = scene([(0, 20)])
    half.ball[150:300] = np.nan
    codes = run(half)
    assert n_spells(codes) == 1
    assert (codes[: len(codes) - 5] == 0).mean() > 0.95


def test_turnover_is_found_near_the_right_frame():
    half = scene([(0, 10), (1, 10)])
    codes = run(half)
    assert n_spells(codes) == 2
    change = np.flatnonzero(np.diff(codes) != 0)[0] + 1
    assert abs(change - 10 * FPS) <= FPS


def test_same_team_restart_gives_two_possessions():
    half = scene([(0, 10), ("dead", 15), (0, 10)])
    codes = run(half)
    assert n_spells(codes) == 2
    assert (codes[12 * FPS: 23 * FPS] == 2).mean() > 0.8


def test_full_length_submission(tmp_path=None):
    import pathlib, tempfile
    tmp_path = pathlib.Path(tmp_path or tempfile.mkdtemp())
    half = Half(ball=np.zeros((10, 2)), players=np.zeros((10, 2, 2)),
                team=np.array([0, 1], dtype=np.int8), start=5)
    labels = to_labels(30, [half], [np.zeros(10, dtype=np.int8)])
    assert len(labels) == 30 and set(labels) == {"H", "X"}
    path = write_submission(labels, str(tmp_path / "s.csv"), 30)
    assert open(path).readline().strip() == "frame,label"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as exc:
                failed += 1
                print("FAIL", name, exc)
    sys.exit(1 if failed else 0)
