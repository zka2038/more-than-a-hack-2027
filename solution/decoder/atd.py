"""Reader for the challenge's automatic tracking (Metrica FIFA/EPTS: metadata XML + raw text).

Raw line: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed" normalised 0-1, "NaN" for
a missing observation. frame_id counts from 1; submission rows count from 0. Team per slot comes
from the starter's own reader, so both agree on which side is home.
"""

import os
import sys

import numpy as np

from .match import Half, Match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402


def read_raw(raw_path, n_frames, n_slots):
    """(ball (N,2), players (N,S,2)) in normalised units, NaN where missing, indexed by video frame."""
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1
            if not 0 <= t < n_frames:
                continue
            b = parts[2].split(",")
            if b[0][:1] != "N":
                ball[t] = (float(b[0]), float(b[1]))
            for s, group in enumerate(parts[1].split(";")[:n_slots]):
                if group[:1] == "N":
                    continue
                g = group.split(",")
                players[t, s] = (float(g[0]), float(g[1]))
    return ball, players


def load(game=GAME, data_dir=None):
    """A Match over the graded halves of the tactical recording."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)
    ball, players = read_raw(raw_path, n_frames, len(team))
    L, W = GAMES[game]["pitch"]
    scale = np.array([L, W])
    ball = (ball - 0.5) * scale
    players = (players - 0.5) * scale
    halves = []
    for _, first, last in half_video_blocks(game):
        sl = slice(first, last + 1)
        halves.append(Half(ball=ball[sl], players=players[sl], team=team, start=first))
    return Match(name=game, pitch=(L, W), halves=halves), n_frames
