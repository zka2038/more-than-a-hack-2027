"""Frame-by-frame viewer for the homography and the Smart Tagging.

    python tools/frame_viewer.py                      # tactical (has Smart Tagging)
    python tools/frame_viewer.py --source broadcast   # broadcast (homography only)
    python tools/frame_viewer.py --no-open            # build it, don't open the browser
    python tools/frame_viewer.py --no-tracking        # skip players and ball (faster)

Builds outputs/frame_viewer/<source>/index.html and opens it. The page plays the match video
from data/ and, on every frame, draws the pitch markings projected through that frame's
homography, plus the tracked players and ball from the ATD. If the lines sit on the painted lines
and the dots sit on the players, the matrix and the tracking are good. It also shows the Smart
Tagging phases active on that frame, and a zoomable timeline of:

    reprojection error   per frame, log scale (lower is better)
    valid                red where the frame has no homography at all
    frame-to-frame jump  how far (px) the projected pitch moves between consecutive frames;
                         spikes mean the homography jumped, not the camera
    players tracked      how many track slots are filled, against the 22 on the pitch
    ball visible         share of frames where the tracking has the ball
    Smart Tagging        one lane per code, coloured by Team

Everything is written under outputs/, which is gitignored. The page embeds data derived from the
match files, so keep it local -- do not publish or commit it (see RULES.md).

Needs numpy and pandas only. The video is decoded by the browser, not by Python.
"""

import argparse
import array
import base64
import json
import shutil
import sys
import webbrowser
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from games import GAME, GAMES, half_video_blocks, require, video_path  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from starter.baseline import team_names, team_of_each_track  # noqa: E402

# Params
TEMPLATE = HERE / "frame_viewer.html"
OUT_DIR = ROOT / "outputs" / "frame_viewer"
VIDEO_SIZE = (1920, 1080)  # Homography pixels; the page uses the real video size once loaded.
# Pitch points (normalised) whose projected movement measures the frame-to-frame jump.
JUMP_POINTS = np.array([[0.5, 0.5], [0.25, 0.5], [0.75, 0.5], [0.5, 0.2], [0.5, 0.8]])
# Tracking positions are stored as uint16 over this range of normalised pitch units (~2 mm steps),
# with 65535 meaning "not seen". It keeps the page small enough to load in a second or two.
Q_MIN, Q_SPAN, Q_MISSING = -0.25, 1.5, 65535


def b64(array, dtype):
    """Little-endian bytes as base64, which the page turns back into a typed array."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def frame_jump(matrices):
    """Mean pixel distance the JUMP_POINTS move from the previous frame. NaN on frame 0."""
    points = np.c_[JUMP_POINTS, np.ones(len(JUMP_POINTS))]
    projected = np.einsum("nij,kj->nki", matrices, points)
    pixels = projected[..., :2] / projected[..., 2:3]
    step = np.linalg.norm(np.diff(pixels, axis=0), axis=2).mean(axis=1)
    return np.concatenate([[np.nan], step])


def quantise(values):
    """Normalised coordinates -> uint16 codes the page turns back into floats."""
    codes = np.round((np.asarray(values, dtype=np.float64) - Q_MIN) / Q_SPAN * (Q_MISSING - 1))
    return np.clip(codes, 0, Q_MISSING - 1).astype("<u2")


def read_tracking(source, n_frames):
    """Players and ball from the raw ATD, packed frame by frame.

    Raw line: "<frame_id>:<slot>;<slot>;...:<ball>", each group "x,y,speed" or NaN when the slot
    is empty. frame_id counts from 1, video frames from 0. Only filled slots are kept: `offsets`
    says where each frame's players start in `slots` / `xy`.
    """
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    team_of_slot = team_of_each_track(meta)
    frames, slots = array.array("I"), array.array("B")
    xs, ys = array.array("d"), array.array("d")
    ball = np.full((n_frames, 2), np.nan)

    with open(raw) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                x, y = group.split(",", 2)[:2]
                frames.append(frame)
                slots.append(slot)
                xs.append(float(x))
                ys.append(float(y))
            if parts[2] and parts[2][0] != "N":
                x, y = parts[2].split(",", 2)[:2]
                ball[frame] = float(x), float(y)

    frames = np.frombuffer(frames, dtype=np.uint32)
    order = np.argsort(frames, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(frames, minlength=n_frames))])
    xy = np.c_[np.frombuffer(xs)[order], np.frombuffer(ys)[order]]
    ball_codes = quantise(np.nan_to_num(ball))
    ball_codes[np.isnan(ball)] = Q_MISSING
    return {
        "qmin": Q_MIN, "qspan": Q_SPAN, "missing": Q_MISSING,
        "team_of_slot": team_of_slot.tolist(),
        "team_names": list(team_names(meta)),
        "off": b64(offsets, "<u4"),
        "slot": b64(np.frombuffer(slots, dtype=np.uint8)[order], "u1"),
        "xy": b64(quantise(xy), "<u2"),
        "ball": b64(ball_codes, "<u2"),
    }, len(frames), float(np.mean(~np.isnan(ball[:, 0])))


def order_tag_teams(tag_teams, home_away):
    """Put the Smart Tagging team names in (home, away) order by matching the ATD club names.

    The two files spell clubs differently ("Atletic Club"), so this matches on any word of four or
    more letters. If nothing matches, the order is left as it was.
    """
    home_words = {word for word in home_away[0].lower().split() if len(word) >= 4}
    home = [team for team in tag_teams if home_words & set(team.lower().split())]
    if len(home) != 1:
        return tag_teams
    return home + [team for team in tag_teams if team != home[0]]


def smart_tags(source):
    """Smart Tagging intervals as plain dicts, or [] if this source has none."""
    if source != "tactical":
        return []
    return [{"code": event["code"], "start": event["start_frame"], "end": event["end_frame"],
             "tags": event["tags"]} for event in load_smart_tagging.load()]


def find_video(source, given=None):
    """The video to play: --video if given, else the expected name, else the only .mp4 there."""
    if given:
        path = Path(given).expanduser()
        if not path.is_file():
            raise IOError("--video {} is not a file".format(given))
        return path
    expected = Path(video_path(source=source))
    if expected.is_file():
        return expected
    candidates = sorted(p for p in expected.parent.glob("*") if p.suffix.lower() == ".mp4")
    if len(candidates) > 1:
        print("  several videos in {}: {} -- using the first; pick one with --video".format(
            expected.parent, ", ".join(p.name for p in candidates)))
    return candidates[0] if candidates else None


def build(source, out_dir=OUT_DIR, tracking=True, video=None):
    matrices, reprojection_error, valid = load_homography.load(source=source)
    events = smart_tags(source)
    track = None
    if tracking:
        print("reading tracking (takes a minute)...")
        track, n_points, ball_share = read_tracking(source, len(valid))

    codes = [code for code, _ in Counter(event["code"] for event in events).most_common()]
    teams = [team for team, _ in Counter(event["tags"].get("Team") for event in events)
             .most_common() if team and team.strip().upper() != "N/A"]
    if track:
        teams = order_tag_teams(teams, track["team_names"])
    video = find_video(source, video)

    payload = {
        "game": GAME,
        "source": source,
        "n": int(len(valid)),
        "fps": GAMES[GAME]["fps"],
        "pitch": list(GAMES[GAME]["pitch"]),
        "size": list(VIDEO_SIZE),
        "halves": [list(block) for block in half_video_blocks(GAME)] if source == "tactical"
        else [],
        "video": video.resolve().as_uri() if video else None,
        "H": b64(matrices.reshape(-1, 9), "<f4"),
        "err": b64(reprojection_error, "<f4"),
        "valid": b64(valid, "u1"),
        "jump": b64(frame_jump(matrices), "<f4"),
        "events": events,
        "codes": codes,
        "teams": teams,
    }

    out = Path(out_dir) / source
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, out / "index.html")
    (out / "data.js").write_text("window.VIEWER_DATA = " + json.dumps(payload) + ";\n")
    (out / "tracking.js").write_text("window.VIEWER_TRACK = " + json.dumps(track) + ";\n")

    usable = reprojection_error[valid]
    print("{} {}: {} frames, {:.1f}% with a homography, median reprojection error {:.3f}".format(
        GAME, source, len(valid), 100.0 * valid.mean(),
        float(np.median(usable)) if len(usable) else float("nan")))
    print("  {} Smart Tagging instances".format(len(events)) if source == "tactical"
          else "  no Smart Tagging for this source")
    if track:
        print("  tracking: {} player positions ({:.1f} per frame), ball visible on {:.1f}% "
              "of frames".format(n_points, n_points / len(valid), 100.0 * ball_share))
    if video:
        print("  video: {}".format(video))
    else:
        print("  no video found in {} -- the viewer will show the overlay only.\n"
              "  Pass it with --video path/to/file.mp4".format(
                  Path(video_path(source=source)).parent))
    return out / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["tactical", "broadcast"], default="tactical")
    parser.add_argument("--out", default=str(OUT_DIR), help="output folder (default: %(default)s)")
    parser.add_argument("--no-tracking", action="store_true", help="skip players and ball")
    parser.add_argument("--video", help="video file to play, if it isn't found in data/")
    parser.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    page = build(args.source, args.out, not args.no_tracking, args.video)
    print("  wrote {}".format(page))
    if not args.no_open:
        webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
