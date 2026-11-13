"""A tiny synthetic match in the provided file formats, for tests and for developing a stage
without the real data.

    from solution.pipeline import fixtures
    truth = fixtures.write_bundle(tmp_dir)          # tmp_dir/tactical/BAR-ATH_tactical_*.{xml,txt,csv}
    run_all.run(stage_dir, out_csv, data_dir=tmp_dir, n_frames=truth["n_frames"],
                blocks=truth["blocks"])

It is not football. Eight players stand near fixed spots; the ball is carried by one of them, then
passed in a straight line to another; twice per half it leaves the pitch for a stoppage. It exists
so that every stage has something with known answers to run on: positions, teams, dead intervals
and the possessing team per frame are all returned in `truth`.

Nothing here reads, or is derived from, the real match.
"""

import os

import numpy as np

from solution.pipeline import contracts as C
from games import GAME, asset_paths

N_FRAMES = 3000
BLOCKS = [(1, 200, 1399), (2, 1600, 2799)]     # (half, first frame, last frame), like games.py
SLOTS_PER_TEAM = 4
REORDER_AT = 1500                              # frame_id where the raw column order changes
HOME_NAME, AWAY_NAME = "FC Barcelona", "Athletic Club"


def simulate(n_frames=N_FRAMES, blocks=BLOCKS, seed=0):
    """Ground truth: players (N,S,2) and ball (N,2) in metres, team per slot, label per frame."""
    rng = np.random.default_rng(seed)
    n_slots = 2 * SLOTS_PER_TEAM
    team = np.repeat([C.HOME, C.AWAY], SLOTS_PER_TEAM).astype(np.int8)
    length, width = C.PITCH

    anchor = np.column_stack([rng.uniform(15, length - 15, n_slots),
                              rng.uniform(10, width - 10, n_slots)])
    t = np.arange(n_frames)[:, None]
    phase = rng.uniform(0, 2 * np.pi, (1, n_slots))
    players = np.empty((n_frames, n_slots, 2))
    players[:, :, 0] = anchor[:, 0] + 3.0 * np.sin(t / 60.0 + phase)
    players[:, :, 1] = anchor[:, 1] + 3.0 * np.cos(t / 75.0 + phase)

    ball = np.full((n_frames, 2), np.nan)
    label = np.full(n_frames, C.X, dtype=np.uint8)
    for _, first, last in blocks:
        frame, holder = first, int(rng.integers(n_slots))
        next_stop = first + 350
        while frame <= last:
            if frame >= next_stop and last - frame > 200:
                # Stoppage: the ball rolls out over the near touchline and is gone for a while.
                out_frames, dead_frames = 10, int(rng.integers(100, 160))
                start = players[frame, holder]
                steps = np.linspace(0, 1, out_frames)[:, None]
                ball[frame:frame + out_frames] = start + steps * (np.array([start[0], -3.0]) - start)
                label[frame:frame + dead_frames] = C.D
                label[frame:frame + 5] = C.H if team[holder] == C.HOME else C.A
                frame += dead_frames
                holder = int(rng.integers(n_slots))
                next_stop = frame + 350
                continue
            # Carry: the ball sits at the holder's feet.
            carry = min(int(rng.integers(40, 90)), last + 1 - frame)
            ball[frame:frame + carry] = players[frame:frame + carry, holder] + 0.3
            label[frame:frame + carry] = C.H if team[holder] == C.HOME else C.A
            frame += carry
            # Pass: a straight line to another player, who becomes the holder on arrival.
            target = int(rng.choice([s for s in range(n_slots) if s != holder]))
            flight = min(int(rng.integers(12, 25)), last + 1 - frame)
            if flight <= 0:
                break
            origin, destination = players[frame, holder], players[frame + flight - 1, target]
            steps = np.linspace(0, 1, flight + 2)[1:-1, None]
            ball[frame:frame + flight] = origin + steps * (destination - origin)
            label[frame:frame + flight] = C.H if team[holder] == C.HOME else C.A
            frame += flight
            holder = target

    # Tracker noise: the ball drops out in bursts, a player drops out now and then.
    seen_ball = ball.copy()
    for start in rng.integers(0, n_frames - 40, n_frames // 150):
        seen_ball[start:start + int(rng.integers(5, 40))] = np.nan
    seen_players = players.copy()
    for start in rng.integers(0, n_frames - 30, n_frames // 300):
        seen_players[start:start + 20, int(rng.integers(n_slots))] = np.nan

    return {"n_frames": n_frames, "blocks": list(blocks), "team": team, "label": label,
            "players": players, "ball": ball, "seen_players": seen_players,
            "seen_ball": seen_ball}


def _number(value):
    return "NaN" if not np.isfinite(value) else "{:.6f}".format(value)


def _metadata_xml(n_frames, n_slots, team, orders):
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", " <Metadata>", "  <Teams>",
             '   <Team id="T_HOME"><Name>{}</Name></Team>'.format(HOME_NAME),
             '   <Team id="T_AWAY"><Name>{}</Name></Team>'.format(AWAY_NAME),
             "  </Teams>", "  <Players>"]
    for slot in range(n_slots):
        lines.append('   <Player id="P{0}" teamId="{1}"><Name>Track {2}</Name>'
                     "<ShirtNumber>{2}</ShirtNumber></Player>".format(
                         slot, "T_HOME" if team[slot] == C.HOME else "T_AWAY", slot + 1))
    lines += ["  </Players>", "  <PlayerChannels>"]
    for slot in range(n_slots):
        for channel in ("x", "y", "speed"):
            lines.append('   <PlayerChannel id="p{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                         .format(slot, channel))
    lines += ["  </PlayerChannels>", " </Metadata>", " <DataFormatSpecifications>"]
    for start, end, order in orders:
        lines.append('  <DataFormatSpecification startFrame="{}" endFrame="{}" separator=":">'
                     .format(start, end))
        lines.append('   <StringRegister name="frameCount"/>')
        lines.append('   <SplitRegister separator=";">')
        for slot in order:
            lines.append('    <SplitRegister separator=",">'
                         + "".join('<PlayerChannelRef playerChannelId="p{}_{}"/>'.format(slot, c)
                                   for c in ("x", "y", "speed"))
                         + "</SplitRegister>")
        lines += ["   </SplitRegister>",
                  '   <SplitRegister separator=","><BallChannelRef channelId="x"/>'
                  '<BallChannelRef channelId="y"/><BallChannelRef channelId="speed"/>'
                  "</SplitRegister>",
                  "  </DataFormatSpecification>"]
    lines += [" </DataFormatSpecifications>", "</main>"]
    return "\n".join(lines) + "\n"


def write_bundle(data_dir, n_frames=N_FRAMES, blocks=BLOCKS, seed=0):
    """Write a synthetic tactical bundle under data_dir and return the simulate() truth."""
    truth = simulate(n_frames, blocks, seed)
    paths = asset_paths(GAME, data_dir)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
    n_slots = len(truth["team"])
    scale = np.array(C.PITCH)
    players = truth["seen_players"] / scale
    ball = truth["seen_ball"] / scale

    # Two column orders, as in a real export that re-orders at a substitution.
    orders = [(1, REORDER_AT - 1, list(range(n_slots))),
              (REORDER_AT, n_frames, list(range(n_slots))[::-1])]
    with open(paths["atd_meta"], "w") as handle:
        handle.write(_metadata_xml(n_frames, n_slots, truth["team"], orders))

    with open(paths["atd_raw"], "w") as handle:
        for t in range(n_frames):
            frame_id = t + 1
            order = orders[0][2] if frame_id < REORDER_AT else orders[1][2]
            groups = []
            for slot in order:
                x, y = players[t, slot]
                groups.append("NaN,NaN,NaN" if not np.isfinite(x)
                              else "{},{},{}".format(_number(x), _number(y), "1.500000"))
            bx, by = ball[t]
            ball_text = ("NaN,NaN,NaN" if not np.isfinite(bx)
                         else "{},{},{}".format(_number(bx), _number(by), "5.000000"))
            handle.write("{}:{}:{}\n".format(frame_id, ";".join(groups), ball_text))

    # Homography: normalised pitch -> pixels as a plain scaling, so the image centre is the centre
    # spot. Every 500th frame has no matrix, to exercise the `valid` handling.
    with open(paths["homography"], "w") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for t in range(n_frames):
            if t % 500 == 0:
                handle.write("{},{},nan,0\n".format(t, ",".join(["nan"] * 9)))
            else:
                handle.write("{},{},0,0,0,{},0,0,0,1,0.25,1\n".format(
                    t, C.IMAGE_SIZE[0], C.IMAGE_SIZE[1]))

    tags = [(blocks[0][1] / 25.0 - 2, blocks[0][1] / 25.0 + 3, "SET PIECES",
             {"Team": "Barcelona", "Half": "1st Half", "Type": "Kick Off (Start)"}),
            (20.0, 31.0, "BUILD UP", {"Team": "Atletic Club", "Half": "1st Half", "Side": "Left"}),
            (40.0, 44.0, "SET PIECES", {"Team": "N/A", "Half": "1st Half", "Type": "Throw-in"}),
            (70.0, 80.0, "PROGRESSION", {"Team": "Barcelona", "Half": "2nd Half"})]
    with open(paths["smart_tagging"], "w") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>\n')
        for index, (start, end, code, labels) in enumerate(tags):
            handle.write("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>"
                         .format(index, start, end, code))
            for group, text in labels.items():
                handle.write("<label><group>{}</group><text>{}</text></label>".format(group, text))
            handle.write("</instance>\n")
        handle.write("</ALL_INSTANCES></file>\n")
    return truth
