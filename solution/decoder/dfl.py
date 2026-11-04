"""Reader for the DFL open dataset (Bassek et al. 2025, figshare 28196177, CC BY 4.0).

Seven Bundesliga matches at 25 fps. The positions file is corrected tracking and carries, on every
ball frame, BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away). That is the same
mechanical derivation the challenge ground truth uses, so it gives reference H/A/D labels on
other games -- which RULES.md allows.

The XML is ~400 MB per match, so it is read line by line with regexes and cached as .npz.
"""

import glob
import os
import re

import numpy as np

from .match import AWAY, HOME, Half, Match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
CACHE_DIR = os.path.join(ROOT, "outputs", "cache", "dfl")

ATTR = re.compile(r'(\w+)="([^"]*)"')
SECTIONS = ("firstHalf", "secondHalf")


def match_ids():
    paths = sorted(glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*.xml")))
    return [re.search(r"(DFL-MAT-\w+)\.xml$", p).group(1) for p in paths]


def _path(kind, match_id):
    hits = glob.glob(os.path.join(DFL_DIR, "DFL_{}_*{}.xml".format(kind, match_id)))
    if len(hits) != 1:
        raise FileNotFoundError("{} for {}: {}".format(kind, match_id, hits))
    return hits[0]


def _teams(match_id):
    """{team_id: HOME/AWAY} and pitch size from the match information file."""
    with open(_path("02_01_matchinformation", match_id), encoding="utf-8") as handle:
        text = handle.read()
    general = dict(ATTR.findall(re.search(r"<General [^>]*>", text).group(0)))
    env = dict(ATTR.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    teams = {general["HomeTeamId"]: HOME, general["GuestTeamId"]: AWAY}
    return teams, (float(env["PitchX"]), float(env["PitchY"]))


def _parse(match_id):
    """Raw arrays per half, straight from the XML."""
    teams, pitch = _teams(match_id)
    sections = {s: {"players": {}, "ball": None} for s in SECTIONS}
    current, rows = None, None
    with open(_path("04_03_positions_raw_observed", match_id), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if rows is not None:
                    rows.append(line)
                continue
            if line.startswith("<FrameSet "):
                attrs = dict(ATTR.findall(line))
                section = attrs.get("GameSection")
                team = attrs.get("TeamId")
                if section not in sections or (team != "BALL" and team not in teams):
                    current, rows = None, None  # referees and extra time are skipped
                    continue
                rows = []
                current = (section, team, attrs.get("PersonId"))
            elif line.startswith("</FrameSet>") and current is not None:
                section, team, person = current
                if team == "BALL":
                    sections[section]["ball"] = _ball_rows(rows)
                else:
                    sections[section]["players"][person] = (teams[team], _player_rows(rows))
                current, rows = None, None
    return sections, pitch


NUM = re.compile(r' (N|X|Y|Z|S|BallPossession|BallStatus)="([^"]*)"')


def _ball_rows(rows):
    out = np.empty((len(rows), 6))
    for i, line in enumerate(rows):
        a = dict(NUM.findall(line))
        out[i] = (float(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("Z", "nan")),
                  float(a.get("BallStatus", "nan")), float(a.get("BallPossession", "nan")))
    return out


def _player_rows(rows):
    out = np.empty((len(rows), 3))
    for i, line in enumerate(rows):
        a = dict(NUM.findall(line))
        out[i] = (float(a["N"]), float(a["X"]), float(a["Y"]))
    return out


def _to_arrays(sections):
    arrays = {}
    for s, data in sections.items():
        ball = data["ball"]
        n0 = int(ball[:, 0].min())
        T = int(ball[:, 0].max()) - n0 + 1
        b = np.full((T, 2), np.nan)
        z = np.full(T, np.nan)
        status = np.zeros(T)
        poss = np.zeros(T)
        idx = ball[:, 0].astype(int) - n0
        b[idx] = ball[:, 1:3]
        z[idx] = ball[:, 3]
        status[idx] = np.nan_to_num(ball[:, 4])
        poss[idx] = np.nan_to_num(ball[:, 5])
        persons = sorted(data["players"])
        P = len(persons)
        players = np.full((T, P, 2), np.nan)
        team = np.zeros(P, dtype=np.int8)
        for j, person in enumerate(persons):
            t, rows = data["players"][person]
            team[j] = t
            k = rows[:, 0].astype(int) - n0
            keep = (k >= 0) & (k < T)
            players[k[keep], j] = rows[keep, 1:3]
        arrays[s] = dict(ball=b, z=z, status=status, poss=poss, players=players, team=team)
    return arrays


def load(match_id, verbose=True):
    """A Match with reference labels: D where BallStatus is 0, else the possessing side."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, match_id + ".npz")
    if not os.path.exists(cache):
        if verbose:
            print("parsing", match_id, "...", flush=True)
        sections, pitch = _parse(match_id)
        arrays = _to_arrays(sections)
        flat = {"pitch": np.array(pitch)}
        for s, a in arrays.items():
            for k, v in a.items():
                flat[s + "/" + k] = v
        np.savez_compressed(cache, **flat)
    data = np.load(cache)
    halves, start = [], 0
    for s in SECTIONS:
        status, poss = data[s + "/status"], data[s + "/poss"]
        labels = np.where(status < 0.5, 2, np.where(poss == 2, AWAY, HOME)).astype(np.int8)
        half = Half(ball=data[s + "/ball"], players=data[s + "/players"],
                    team=data[s + "/team"], start=start, labels=labels)
        half.z = data[s + "/z"]
        halves.append(half)
        start += half.n + 1  # one X frame between halves keeps them separate for the scorer
    return Match(name=match_id, pitch=tuple(data["pitch"]), halves=halves)
