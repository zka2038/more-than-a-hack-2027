"""Metrica Sports sample games 1-3 (github.com/metrica-sports/sample-data), as external matches.

Same vendor as the challenge data, so the closest possession convention we can get. The source asks
users to acknowledge it; no other licence terms are stated. Games 1-2 are CSV, game 3 is FIFA EPTS
(the same format as the challenge ATD, read with the S0 reader) with JSON events.

Reference labels are derived from the human event data of these games (allowed: other games):
  - possession = team of the latest possession event (PASS, RECOVERY, CARRY, SET PIECE, SHOT,
    BALL LOST, BALL OUT) at or before the frame. CHALLENGE never flips possession: a duel or a touch
    without control does not count (CHALLENGE.md).
  - dead ball = from the event that stopped play (BALL OUT, a foul, a goal, an offside, or the end
    of the previous event) to the next SET PIECE, and from the start of a period to its kick-off.

    git clone https://github.com/metrica-sports/sample-data <MTAH_CACHE>/metrica
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from solution.pipeline import contracts as C
from solution.bench.external import CACHE, assemble, forward_fill_team

GAMES = (1, 2, 3)
DATA = os.path.join(CACHE, "metrica", "data")

POSSESSION_TYPES = {"PASS", "RECOVERY", "CARRY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT"}


def _game_dir(game):
    return os.path.join(DATA, "Sample_Game_{}".format(game))


# -- tracking ------------------------------------------------------------------------------------

def _csv_side(game, side):
    path = os.path.join(_game_dir(game), "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game,
                                                                                          side))
    table = pd.read_csv(path, skiprows=2, low_memory=False)
    values = table.values
    period = values[:, 0].astype(int)
    frame = values[:, 1].astype(int)
    xy = values[:, 3:].astype(np.float64)
    xy = xy[:, : (xy.shape[1] // 2) * 2].reshape(len(xy), -1, 2)
    return period, frame, xy[:, :-1], xy[:, -1]          # players, ball (last pair)


def _tracking_csv(game):
    period, frame, home, ball = _csv_side(game, "Home")
    period_a, frame_a, away, _ = _csv_side(game, "Away")
    if not np.array_equal(frame, frame_a):
        raise ValueError("game {}: home and away frames differ".format(game))
    players = np.concatenate([home, away], axis=1)
    team = np.array([C.HOME] * home.shape[1] + [C.AWAY] * away.shape[1], dtype=np.int8)
    return period, frame, players, ball, team


def _tracking_epts(game):
    from solution.pipeline.s0_tracking.run import read_meta, read_raw
    folder = _game_dir(game)
    meta = os.path.join(folder, "Sample_Game_{}_metadata.xml".format(game))
    raw = os.path.join(folder, "Sample_Game_{}_tracking.txt".format(game))
    team, specs, _ = read_meta(meta)
    params = {}
    for p in ET.parse(meta).getroot().iter():
        if p.tag.rsplit("}", 1)[-1] == "ProviderParameter":
            kids = {c.tag.rsplit("}", 1)[-1]: (c.text or "") for c in p}
            params[kids.get("Name")] = kids.get("Value")
    last = int(params["second_half_end"])
    ball, players, _ = read_raw(raw, last, len(team), specs)
    frame = np.arange(1, last + 1)
    period = np.where(frame <= int(params["first_half_end"]), 1, 2)
    return period, frame, players.astype(np.float64), ball.astype(np.float64), team


def _home_team_id(game):
    meta = os.path.join(_game_dir(game), "Sample_Game_{}_metadata.xml".format(game))
    teams = [t for t in ET.parse(meta).getroot().iter() if t.tag.rsplit("}", 1)[-1] == "Team"]
    return teams[0].get("id")


# -- events --------------------------------------------------------------------------------------

def events(game):
    """DataFrame: team (0/1/-1), type, subtype (upper, '-'-joined), period, start, end (frames)."""
    if game in (1, 2):
        e = pd.read_csv(os.path.join(_game_dir(game), "Sample_Game_{}_RawEventsData.csv".format(
            game)))
        out = pd.DataFrame({
            "team": e["Team"].map({"Home": C.HOME, "Away": C.AWAY}).fillna(-1).astype(int),
            "type": e["Type"].str.upper(),
            "subtype": e["Subtype"].fillna("").str.upper(),
            "period": e["Period"].astype(int),
            "start": e["Start Frame"].astype(int),
            "end": e["End Frame"].astype(int),
        })
    else:
        home = _home_team_id(game)
        with open(os.path.join(_game_dir(game), "Sample_Game_{}_events.json".format(game))) as f:
            data = json.load(f)["data"]
        rows = []
        for x in data:
            sub = x.get("subtypes")
            if isinstance(sub, list):
                sub = "-".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            team_id = (x.get("team") or {}).get("id")
            rows.append({"team": C.HOME if team_id == home else (C.AWAY if team_id else -1),
                         "type": x["type"]["name"].upper(), "subtype": (sub or "").upper(),
                         "period": int(x["period"]), "start": int(x["start"]["frame"]),
                         "end": int(x["end"]["frame"] or x["start"]["frame"])})
        out = pd.DataFrame(rows)
    out["end"] = np.maximum(out["end"], out["start"])
    return out.sort_values(["start", "end"], kind="stable").reset_index(drop=True)


def _stops_play(row):
    t, s = row.type, row.subtype
    if t in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if t == "SHOT" and ("GOAL" in s.split("-") or "OUT" in s.split("-")):
        return True
    if "OFFSIDE" in s or "END HALF" in s:
        return True
    return False


def labels_for(events_table, period, frame):
    """H/A/D codes (len(frame),) derived from the events, for one game's tracking frames."""
    n = len(frame)
    index_of = {f: i for i, f in enumerate(frame)}
    first = frame[0]

    def at(f):
        return int(np.clip(f - first, 0, n - 1)) if f not in index_of else index_of[f]

    team = np.full(n, -1, dtype=np.int64)
    for row in events_table.itertuples():
        if row.type in POSSESSION_TYPES and row.team >= 0:
            team[at(row.start)] = row.team
    # A later event at the same frame wins; carry the latest team forward.
    possession = forward_fill_team(team)

    dead = np.zeros(n, dtype=bool)
    ev = events_table
    for p in np.unique(period):
        frames_p = np.flatnonzero(period == p)
        p0, p1 = frames_p[0], frames_p[-1] + 1
        ev_p = ev[ev.period == p]
        set_pieces = ev_p[ev_p.type == "SET PIECE"]
        # Before the period's first kick-off.
        if len(set_pieces):
            dead[p0:at(set_pieces.start.iloc[0])] = True
        # After a period-ending event.
        last = ev_p[ev_p.subtype.str.contains("END HALF")]
        if len(last):
            dead[at(last.end.iloc[-1]):p1] = True
        prev_restart = -1
        for sp in set_pieces.itertuples():
            s = sp.start
            before = ev_p[(ev_p.start < s) & (ev_p.start >= prev_restart) & (ev_p.index < sp.Index)]
            stops = [r for r in before.itertuples() if _stops_play(r)]
            if stops:
                r = stops[-1]
                d0 = r.end if r.type in ("BALL OUT", "SHOT") or "OFFSIDE" in r.subtype else r.start
            elif len(before):
                d0 = before.end.iloc[-1]
            else:
                d0 = s
            if s > d0:
                dead[at(d0):at(s)] = True
            prev_restart = s
    return np.where(dead, C.D, possession).astype(np.int8)


def load(game):
    if game in (1, 2):
        period, frame, players, ball, team = _tracking_csv(game)
    else:
        period, frame, players, ball, team = _tracking_epts(game)
    codes = labels_for(events(game), period, frame)
    L, W = C.PITCH
    halves = []
    for p in (1, 2):
        m = period == p
        b = ball[m] * (L, W)
        pl = players[m] * (L, W)
        halves.append({"ball": b.astype(np.float32), "players": pl.astype(np.float32),
                       "team": team, "labels": codes[m], "z": None})
    return assemble(halves, "MET-{}".format(game), "metrica")
