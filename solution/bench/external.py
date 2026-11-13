"""Load external matches into the stage-0 contract shape, with reference H/A/D labels.

Sources (PLAN.md section 6; none of them is BAR-ATH):
  - DFL open data: Bassek et al. 2025, figshare article 28196177, CC BY 4.0. Seven Bundesliga
    matches at 25 fps. Corrected tracking carrying BallStatus (1 alive, 0 dead) and BallPossession
    (1 home, 2 away) on every frame: the same mechanical derivation the challenge ground truth uses.
  - Metrica Sports sample games 1-3 (github.com/metrica-sports/sample-data): see metrica.py.

A match is returned as (s0, gt, info):
  s0    dict with exactly the s0_tracking arrays: metres on BAR-ATH's 105.3 x 68 pitch, corner
        origin, HOME = 0, half 1/2 on play frames and 0 on the padding around them
  gt    int8 (N,) codes H, A, D, X (X exactly where half == 0)
  info  {"name", "source", "ball_z" (N,) or None}

Raw downloads and parsed caches live under MTAH_CACHE (default outputs/cache, gitignored).

    python -m solution.bench.external --download      # fetch the DFL files from figshare
    python -m solution.bench.external --parse         # parse + cache every match, print summary
"""

import argparse
import glob
import json
import os
import re
import urllib.request
from multiprocessing import get_context

import numpy as np

from solution.pipeline import contracts as C

CACHE = os.environ.get("MTAH_CACHE") or os.path.join(C.ROOT, "outputs", "cache")
DFL_RAW = os.path.join(CACHE, "dfl", "raw")
DFL_PARSED = os.path.join(CACHE, "dfl", "parsed")
FIGSHARE_ARTICLE = 28196177
PAD = 250   # ungraded frames before, between and after the halves (like BAR-ATH's X frames)

DFL_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


# -- shared helpers ------------------------------------------------------------------------------

def player_speed(player_xy, k=2, cap=12.0):
    """m/s from a centred difference over +/- k frames; NaN where either end is missing."""
    n = len(player_xy)
    speed = np.full(player_xy.shape[:2], np.nan, dtype=np.float32)
    if n > 2 * k:
        d = np.linalg.norm(player_xy[2 * k:] - player_xy[:n - 2 * k], axis=2) * C.FPS / (2 * k)
        speed[k:n - k] = np.where(d > cap, np.nan, d)
    return speed


def assemble(halves, name, source):
    """Concatenate halves with PAD ungraded frames around them.

    halves: list of dicts with ball (T,2), players (T,P,2) in BAR-ATH metres (corner origin),
    team (P,), labels (T,) codes H/A/D, and optionally z (T,). All halves share the slot axis.
    """
    total = PAD + sum(len(h["ball"]) + PAD for h in halves)
    P = halves[0]["players"].shape[1]
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    players = np.full((total, P, 2), np.nan, dtype=np.float32)
    half = np.zeros(total, dtype=np.int8)
    gt = np.full(total, C.X, dtype=np.int8)
    z = np.full(total, np.nan, dtype=np.float32)
    at = PAD
    for number, h in enumerate(halves, start=1):
        T = len(h["ball"])
        ball[at:at + T] = h["ball"]
        players[at:at + T] = h["players"]
        half[at:at + T] = number
        gt[at:at + T] = h["labels"]
        if h.get("z") is not None:
            z[at:at + T] = h["z"]
        at += T + PAD
    s0 = {
        "ball_xy": ball,
        "player_xy": players,
        "player_speed": player_speed(players),
        "player_team": np.asarray(halves[0]["team"], dtype=np.int8),
        "half": half,
        "cam_xy": np.full((total, 2), np.nan, dtype=np.float32),
        "reproj_err": np.full(total, np.nan, dtype=np.float32),
    }
    C.validate("s0_tracking", s0)
    has_z = any(h.get("z") is not None for h in halves)
    return s0, gt, {"name": name, "source": source, "ball_z": z if has_z else None}


def forward_fill_team(team):
    """Fill -1 entries of an int array with the previous valid value (then the next one)."""
    idx = np.where(team >= 0, np.arange(len(team)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, team[np.maximum(idx, 0)], -1)
    if (out < 0).any() and (out >= 0).any():
        out[out < 0] = out[out >= 0][0]
    return out


# -- DFL -----------------------------------------------------------------------------------------

ATTR = re.compile(r'(\w+)="([^"]*)"')
NUM = re.compile(r' (N|X|Y|Z|BallPossession|BallStatus)="([^"]*)"')
SECTIONS = ("firstHalf", "secondHalf")


def download_dfl():
    """Fetch the 21 DFL files (about 2.6 GB) into DFL_RAW. Skips files already present."""
    os.makedirs(DFL_RAW, exist_ok=True)
    url = "https://api.figshare.com/v2/articles/{}/files?page_size=100".format(FIGSHARE_ARTICLE)
    with urllib.request.urlopen(url) as response:
        files = json.load(response)
    for f in files:
        target = os.path.join(DFL_RAW, f["name"])
        if os.path.exists(target) and os.path.getsize(target) == f["size"]:
            continue
        print("downloading", f["name"], flush=True)
        urllib.request.urlretrieve(f["download_url"], target)


def _dfl_path(kind, match_id):
    hits = glob.glob(os.path.join(DFL_RAW, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
    if len(hits) != 1:
        raise FileNotFoundError("DFL {} for {}: {} (run --download)".format(kind, match_id, hits))
    return hits[0]


def _dfl_teams(match_id):
    with open(_dfl_path("02_01_matchinformation", match_id), encoding="utf-8") as handle:
        text = handle.read()
    general = dict(ATTR.findall(re.search(r"<General [^>]*>", text).group(0)))
    env = dict(ATTR.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    teams = {general["HomeTeamId"]: C.HOME, general["GuestTeamId"]: C.AWAY}
    return teams, (float(env["PitchX"]), float(env["PitchY"]))


def _rows(lines, keys):
    out = np.full((len(lines), len(keys)), np.nan)
    for i, line in enumerate(lines):
        a = dict(NUM.findall(line))
        for j, key in enumerate(keys):
            if key in a:
                out[i, j] = float(a[key])
    return out


def _parse_dfl(match_id):
    """Per section: ball rows (N,X,Y,Z,status,poss) and {person: (team, rows (N,X,Y))}."""
    teams, pitch = _dfl_teams(match_id)
    sections = {s: {"players": {}, "ball": None} for s in SECTIONS}
    current, rows = None, None
    with open(_dfl_path("04_03_positions_raw_observed", match_id), encoding="utf-8") as handle:
        for line in handle:
            line = line.lstrip()
            if line.startswith("<Frame "):
                if rows is not None:
                    rows.append(line)
                continue
            if line.startswith("<FrameSet "):
                attrs = dict(ATTR.findall(line))
                section, team = attrs.get("GameSection"), attrs.get("TeamId")
                if section not in sections or (team != "BALL" and team not in teams):
                    current, rows = None, None      # referees and extra sections are skipped
                    continue
                rows, current = [], (section, team, attrs.get("PersonId"))
            elif line.startswith("</FrameSet>") and current is not None:
                section, team, person = current
                if team == "BALL":
                    sections[section]["ball"] = _rows(rows, ("N", "X", "Y", "Z", "BallStatus",
                                                             "BallPossession"))
                else:
                    sections[section]["players"][person] = (teams[team],
                                                            _rows(rows, ("N", "X", "Y")))
                current, rows = None, None
    return sections, pitch


def parse_dfl(match_id):
    """Parse one match into a cached .npz of BAR-ATH-metre arrays per half. Returns the path."""
    os.makedirs(DFL_PARSED, exist_ok=True)
    path = os.path.join(DFL_PARSED, match_id + ".npz")
    if os.path.exists(path):
        return path
    sections, (length, width) = _parse_dfl(match_id)
    sx, sy = C.PITCH[0] / length, C.PITCH[1] / width
    persons = sorted(set().union(*(s["players"].keys() for s in sections.values())))
    team = np.zeros(len(persons), dtype=np.int8)
    out = {}
    for name, data in sections.items():
        ball = data["ball"]
        n0 = int(np.nanmin(ball[:, 0]))
        T = int(np.nanmax(ball[:, 0])) - n0 + 1
        idx = ball[:, 0].astype(int) - n0
        b = np.full((T, 2), np.nan, dtype=np.float32)
        b[idx, 0] = (ball[:, 1] + length / 2) * sx
        b[idx, 1] = (ball[:, 2] + width / 2) * sy
        z = np.full(T, np.nan, dtype=np.float32)
        z[idx] = ball[:, 3]
        status = np.zeros(T, dtype=np.int8)
        status[idx] = np.nan_to_num(ball[:, 4]).astype(np.int8)
        poss = np.full(T, -1, dtype=np.int8)
        poss[idx] = np.where(ball[:, 5] == 1, C.HOME, np.where(ball[:, 5] == 2, C.AWAY, -1))
        players = np.full((T, len(persons), 2), np.nan, dtype=np.float32)
        for j, person in enumerate(persons):
            if person not in data["players"]:
                continue
            t, rows = data["players"][person]
            team[j] = t
            k = rows[:, 0].astype(int) - n0
            keep = (k >= 0) & (k < T)
            players[k[keep], j, 0] = (rows[keep, 1] + length / 2) * sx
            players[k[keep], j, 1] = (rows[keep, 2] + width / 2) * sy
        poss = forward_fill_team(poss)
        labels = np.where(status == 1, poss, C.D).astype(np.int8)
        out.update({name + "/ball": b, name + "/z": z, name + "/players": players,
                    name + "/labels": labels})
    out["team"] = team
    np.savez_compressed(path, **out)
    return path


def load_dfl(match_id):
    with np.load(parse_dfl(match_id)) as data:
        halves = [{"ball": data[s + "/ball"], "players": data[s + "/players"],
                   "team": data["team"], "labels": data[s + "/labels"], "z": data[s + "/z"]}
                  for s in SECTIONS]
    return assemble(halves, "DFL-" + match_id, "dfl")


# -- registry ------------------------------------------------------------------------------------

def match_names():
    from solution.bench import metrica
    return ["DFL-" + m for m in DFL_IDS] + ["MET-{}".format(g) for g in metrica.GAMES]


def load(name):
    """(s0, gt, info) for 'DFL-<id>' or 'MET-<game>'."""
    if name.startswith("DFL-"):
        return load_dfl(name[4:])
    if name.startswith("MET-"):
        from solution.bench import metrica
        return metrica.load(int(name[4:]))
    raise KeyError(name)


def summary(name):
    s0, gt, info = load(name)
    graded = gt != C.X
    n_spells = sum(len(C.runs(gt == team)) for team in (C.H, C.A))
    alive = int(((gt == C.H) | (gt == C.A)).sum())
    miss = np.isnan(s0["ball_xy"][graded, 0]).mean()
    return ("{:12s} frames {:7d} graded {:7d}  D {:4.1f}%  possessions {:4d}  mean {:4.1f} s  "
            "slots {:2d}  ball missing {:4.1f}%".format(
                name, len(gt), int(graded.sum()), 100.0 * (gt[graded] == C.D).mean(), n_spells,
                alive / (C.FPS * max(n_spells, 1)), s0["player_xy"].shape[1], 100.0 * miss))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.download:
        download_dfl()
    if args.parse:
        with get_context("spawn").Pool(args.workers) as pool:
            pool.map(parse_dfl, DFL_IDS)
        for name in match_names():
            print(summary(name), flush=True)


if __name__ == "__main__":
    main()
