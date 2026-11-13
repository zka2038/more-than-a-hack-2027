"""S0: read the provided tactical assets once and cache them in pipeline form.

    python -m solution.pipeline.s0_tracking.run --stages outputs/stages

Writes s0_tracking.npz and s0_tags.csv (PLAN.md section 1). Reads the tactical ATD (required),
homography and Smart Tagging (both optional: missing ones become NaN / an empty table, with a
warning). Takes a minute or two on the real match.

ATD raw line: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y[,speed]" normalised 0-1 and
"NaN" for a missing observation. frame_id counts from 1; pipeline frames count from 0.

Which player each ";"-separated group belongs to is defined by the metadata's
DataFormatSpecification blocks, one per frame range, and the order can change between ranges. Each
group is therefore mapped to its player through that block's PlayerChannelRef list, so a slot index
here always means the same <Player> entry. Without such blocks the <Player> list order is used.
"""

import argparse
import difflib
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from solution.pipeline import contracts as C
from games import GAME, asset_paths, half_video_blocks, require

TAG_HALVES = {"1st half": 1, "2nd half": 2}


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_meta(meta_path):
    """(team per slot, [(start_frame_id, end_frame_id, [slot per group])], (home name, away name)).

    Home is the first <Team>: the FIFA/EPTS convention, which starter/baseline.py and kloppy follow.
    """
    root = ET.parse(meta_path).getroot()
    teams = [t for t in root.iter() if _local(t.tag) == "Team"]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    names = tuple(next((c.text or "" for c in t if _local(c.tag) == "Name"), "") for t in teams)
    home_id = teams[0].get("id")

    players = [p for p in root.iter() if _local(p.tag) == "Player" and p.get("teamId")]
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    slot = {p.get("id"): i for i, p in enumerate(players)}
    team = np.array([C.HOME if p.get("teamId") == home_id else C.AWAY for p in players],
                    dtype=np.int8)

    channel_player = {c.get("id"): c.get("playerId") for c in root.iter()
                      if _local(c.tag) == "PlayerChannel"}
    specs = []
    for spec in (s for s in root.iter() if _local(s.tag) == "DataFormatSpecification"):
        order = []
        for ref in (r for r in spec.iter() if _local(r.tag) == "PlayerChannelRef"):
            player_id = channel_player.get(ref.get("playerChannelId"))
            # Several refs (x, y, speed) per player, in group order: keep each player once.
            if player_id in slot and (not order or order[-1] != slot[player_id]):
                order.append(slot[player_id])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    if not any(order for _, _, order in specs):
        specs = [(1, 10 ** 12, list(range(len(players))))]
    return team, sorted(specs), names


def read_raw(raw_path, n_frames, n_slots, specs):
    """(ball (N,2), players (N,S,2), speed (N,S)) normalised 0-1, NaN where missing."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    k = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            t = frame_id - 1
            if not 0 <= t < n_frames:
                continue
            while k + 1 < len(specs) and frame_id > specs[k][1]:
                k += 1
            order = specs[k][2]
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N" and b[1][:1] != "N":
                ball[t] = (float(b[0]), float(b[1]))
            for group_index, group in enumerate(parts[1].split(";")):
                if group_index >= len(order) or group[:1] == "N":
                    continue
                g = group.split(",")
                players[t, order[group_index]] = (float(g[0]), float(g[1]))
                if len(g) >= 3 and g[2][:1] != "N":
                    speed[t, order[group_index]] = float(g[2])
    return ball, players, speed


def half_array(n_frames, blocks):
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in blocks:
        half[first:last + 1] = number
    return half


def camera(homography_path, n_frames):
    """(cam_xy (N,2) metres, reproj_err (N,)): the pitch point under the image centre.

    The homography maps normalised pitch coordinates to pixels, so the centre pixel goes through
    the inverse. NaN where the file has no valid matrix.
    """
    cam = np.full((n_frames, 2), np.nan, dtype=np.float32)
    error = np.full(n_frames, np.nan, dtype=np.float32)
    table = pd.read_csv(homography_path)
    frames = table["frame"].values
    keep = (frames >= 0) & (frames < n_frames)
    table, frames = table[keep], frames[keep]
    error[frames] = table["reprojection_error"].values

    columns = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    matrices = table[columns].values.astype(np.float64).reshape(-1, 3, 3)
    ok = table["valid"].values.astype(bool) & np.isfinite(matrices).all(axis=(1, 2))
    ok[ok] = np.abs(np.linalg.det(matrices[ok])) > 1e-12
    centre = np.array([C.IMAGE_SIZE[0] / 2.0, C.IMAGE_SIZE[1] / 2.0, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        point = np.linalg.inv(matrices[ok]) @ centre
        xy = point[:, :2] / point[:, 2:3]
    cam[frames[ok]] = (xy * np.array(C.PITCH)).astype(np.float32)
    return cam, error


def tag_team(value, names):
    """HOME / AWAY / UNKNOWN for a Smart Tagging Team value.

    The tagging spells the clubs its own way ("Atletic Club"), so match on similarity to the two
    names in the ATD metadata rather than on equality.
    """
    text = (value or "").strip().lower()
    if not text or text == "n/a":
        return C.UNKNOWN
    scores = [difflib.SequenceMatcher(None, text, name.strip().lower()).ratio() for name in names]
    if max(scores) < 0.5 or scores[0] == scores[1]:
        return C.UNKNOWN
    return C.HOME if scores[0] > scores[1] else C.AWAY


def read_tags(tagging_path, names, n_frames):
    rows = []
    for instance in ET.parse(tagging_path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        start = int(round(float(instance.findtext("start")) * C.FPS))
        end = int(round(float(instance.findtext("end")) * C.FPS))
        rows.append({
            "start_frame": min(max(start, 0), n_frames),
            "end_frame": min(max(end, 0), n_frames),
            "code": (instance.findtext("code") or "").strip().upper(),
            "team": tag_team(tags.get("Team"), names),
            "type": (tags.get("Type") or "").strip(),
            "half": TAG_HALVES.get((tags.get("Half") or "").strip().lower(), 0),
            "side": (tags.get("Side") or "").strip(),
        })
    table = pd.DataFrame(rows, columns=C.TABLE_SCHEMAS["s0_tags"])
    return table.sort_values(["start_frame", "end_frame"], kind="stable").reset_index(drop=True)


def run(stage_dir=None, game=GAME, data_dir=None, n_frames=None, blocks=None):
    """Build both S0 files. n_frames and blocks are overridable for small synthetic bundles."""
    n_frames = n_frames or C.N_FRAMES
    blocks = blocks if blocks is not None else half_video_blocks(game)
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    paths = asset_paths(game, data_dir)

    team, specs, names = read_meta(meta_path)
    print("S0: H = {!r}, A = {!r}, {} track slots".format(names[0], names[1], len(team)))
    ball, players, speed = read_raw(raw_path, n_frames, len(team), specs)
    scale = np.array(C.PITCH, dtype=np.float32)

    if os.path.isfile(paths["homography"]):
        cam, error = camera(paths["homography"], n_frames)
    else:
        print("S0: WARNING no homography file, cam_xy and reproj_err are NaN")
        cam = np.full((n_frames, 2), np.nan, dtype=np.float32)
        error = np.full(n_frames, np.nan, dtype=np.float32)

    arrays = {
        "ball_xy": ball * scale,
        "player_xy": players * scale,
        "player_speed": speed,
        "player_team": team,
        "half": half_array(n_frames, blocks),
        "cam_xy": cam,
        "reproj_err": error,
    }
    out = [C.save("s0_tracking", stage_dir, arrays, params={"game": game}, n=n_frames)]

    if os.path.isfile(paths["smart_tagging"]):
        tags = read_tags(paths["smart_tagging"], names, n_frames)
    else:
        print("S0: WARNING no Smart Tagging file, s0_tags.csv is empty")
        tags = pd.DataFrame(columns=C.TABLE_SCHEMAS["s0_tags"])
    out.append(C.save_table("s0_tags", stage_dir, tags, n=n_frames))

    graded = arrays["half"] > 0
    seen = np.isfinite(arrays["ball_xy"][:, 0])
    print("S0: ball seen on {:.1f}% of graded frames, {} tags ({} with a team)".format(
        100.0 * seen[graded].mean() if graded.any() else 0.0, len(tags),
        int((tags["team"] >= 0).sum()) if len(tags) else 0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stages", default=C.DEFAULT_STAGE_DIR, help="stage file folder")
    parser.add_argument("--data-dir", default=None, help="match bundle (default: games.DATA_DIR)")
    args = parser.parse_args()
    for path in run(args.stages, data_dir=args.data_dir):
        print("S0: wrote {}".format(os.path.relpath(path, C.ROOT)))


if __name__ == "__main__":
    main()
