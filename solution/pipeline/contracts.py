"""File contracts between the pipeline stages. See PLAN.md, sections 1-5.

Every stage reads and writes through this module and nowhere else, so a stage that runs is a stage
that honours its contract. Do not change a schema on a stage branch: schema changes go through a
PR to main that every stage owner sees.

Conventions (PLAN.md section 1):
  - arrays are indexed by tactical video frame, length N (199319 on the real match)
  - positions are metres, x in [0, 105.3], y in [0, 68.0], ATD orientation, no per-half flip
  - teams are HOME = 0, AWAY = 1, UNKNOWN = -1
  - missing is NaN for floats and -1 for ints
  - intervals are [start_frame, end_frame), end exclusive

    from solution.pipeline import contracts as C
    s0 = C.load("s0_tracking", stage_dir)
    C.save("s1_inplay", stage_dir, {"p_alive": p, "alive": a}, params={"min_gap": 75})
    C.save_table("s1_stoppages", stage_dir, dataframe)
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, source_frames  # noqa: E402

N_FRAMES = source_frames("tactical", GAME)
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]            # (length, width) in metres
IMAGE_SIZE = (1920, 1080)               # tactical video, pixels
DEFAULT_STAGE_DIR = os.path.join(ROOT, "outputs", "stages")

HOME, AWAY, UNKNOWN = 0, 1, -1

# Submission label codes. Same order as evaluation/possession_metrics.py, on purpose.
LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3

# s2_ball.ball_src
SRC_NONE, SRC_OBSERVED, SRC_CORRECTED, SRC_INTERPOLATED, SRC_CARRIED, SRC_REDETECTED = range(6)

N = "N"   # placeholder for the frame count in a shape
S = "S"   # placeholder for the number of track slots

# name -> {array: (dtype, shape)}. "N" is the frame count, "S" the slot count.
NPZ_SCHEMAS = {
    "s0_tracking": {
        "ball_xy": ("float32", (N, 2)),
        "player_xy": ("float32", (N, S, 2)),
        "player_speed": ("float32", (N, S)),
        # Per slot, not per frame: the reader re-orders every frame's groups into <Player> list
        # order, so a slot's team is constant even where the raw column order changes.
        "player_team": ("int8", (S,)),
        "half": ("int8", (N,)),
        "cam_xy": ("float32", (N, 2)),
        "reproj_err": ("float32", (N,)),
    },
    "s1_inplay": {
        "p_alive": ("float32", (N,)),
        "alive": ("int8", (N,)),
    },
    "s2_ball": {
        "ball_xy": ("float32", (N, 2)),
        "ball_vel": ("float32", (N, 2)),
        "ball_src": ("uint8", (N,)),
        "ball_conf": ("float32", (N,)),
        "in_air": ("float32", (N,)),
    },
    "s3_touch": {
        "last_touch_team": ("int8", (N,)),
        "frames_since_touch": ("int32", (N,)),
        "touch_ll": ("float32", (N, 2)),
        "control": ("float32", (N, 2)),
    },
    "s4_possession": {
        "prob": ("float32", (N, 3)),
        "label": ("uint8", (N,)),
    },
}

STOP_CAUSES = ("out_touchline", "out_goalline", "foul", "goal", "offside", "period", "unknown")
RESTART_TYPES = ("throw_in", "corner", "goal_kick", "free_kick", "kick_off", "penalty", "unknown")
TOUCH_KINDS = ("restart", "receive", "release", "carry", "deflection", "duel", "unknown")

# name -> ordered column list. Extra columns are allowed after these; missing ones are an error.
TABLE_SCHEMAS = {
    "s0_tags": ["start_frame", "end_frame", "code", "team", "type", "half", "side"],
    "s1_stoppages": ["stop_id", "start_frame", "end_frame", "cause", "restart_type",
                     "restart_team", "restart_x", "restart_y", "conf"],
    "s3_touches": ["touch_id", "frame", "team", "slot", "x", "y", "kind", "conf", "p_home"],
}


class ContractError(ValueError):
    """A stage file does not match its contract."""


def _fail(name, message):
    raise ContractError("{}: {}".format(name, message))


def _in_set(name, key, values, allowed):
    bad = np.setdiff1d(np.unique(values), np.asarray(allowed))
    if bad.size:
        _fail(name, "{} has values outside {}: {}".format(key, list(allowed), bad[:5].tolist()))


def _in_unit(name, key, values):
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        _fail(name, "{} must be in [0, 1], found [{:.3f}, {:.3f}]".format(
            key, float(finite.min()), float(finite.max())))


def validate(name, arrays, n=None):
    """Check one .npz payload against its schema and its own invariants. Returns the frame count.

    n is the expected frame count. Leave it None to accept whatever length the arrays share (the
    synthetic fixtures are short); pass N_FRAMES to insist on the real match.
    """
    if name not in NPZ_SCHEMAS:
        raise KeyError("no npz contract {!r}; have {}".format(name, sorted(NPZ_SCHEMAS)))
    schema = NPZ_SCHEMAS[name]

    missing = sorted(set(schema) - set(arrays))
    if missing:
        _fail(name, "missing arrays {}".format(missing))
    unknown = sorted(set(arrays) - set(schema) - {"meta"})
    if unknown:
        _fail(name, "arrays not in the contract: {}".format(unknown))

    sizes = {N: n, S: None}
    for key, (dtype, shape) in schema.items():
        value = arrays[key]
        if not isinstance(value, np.ndarray):
            _fail(name, "{} is {}, not an ndarray".format(key, type(value).__name__))
        if value.dtype != np.dtype(dtype):
            _fail(name, "{} has dtype {}, contract says {}".format(key, value.dtype, dtype))
        if value.ndim != len(shape):
            _fail(name, "{} has shape {}, contract says {}".format(key, value.shape, shape))
        for got, want in zip(value.shape, shape):
            if want in sizes:
                if sizes[want] is None:
                    sizes[want] = got
                want = sizes[want]
            if got != want:
                _fail(name, "{} has shape {}, contract says {} with N={} S={}".format(
                    key, value.shape, shape, sizes[N], sizes[S]))

    _CHECKS[name](name, arrays)
    return sizes[N]


def _check_s0(name, a):
    _in_set(name, "player_team", a["player_team"], (HOME, AWAY))
    _in_set(name, "half", a["half"], (0, 1, 2))


def _check_s1(name, a):
    _in_set(name, "alive", a["alive"], (-1, 0, 1))
    _in_unit(name, "p_alive", a["p_alive"])
    graded = a["alive"] >= 0
    if np.isnan(a["p_alive"][graded]).any():
        _fail(name, "p_alive is NaN on graded frames")


def _check_s2(name, a):
    _in_set(name, "ball_src", a["ball_src"], range(6))
    _in_unit(name, "ball_conf", a["ball_conf"])
    _in_unit(name, "in_air", a["in_air"])
    has_xy = np.isfinite(a["ball_xy"]).all(axis=1)
    if (has_xy != (a["ball_src"] != SRC_NONE)).any():
        _fail(name, "ball_xy must be finite exactly where ball_src != 0")
    estimated = a["ball_src"] >= SRC_INTERPOLATED
    if (a["ball_conf"][estimated] >= 1.0).any():
        _fail(name, "an estimated position (ball_src >= 3) must have ball_conf < 1")
    if np.isnan(a["ball_conf"][has_xy]).any():
        _fail(name, "ball_conf is NaN where a position is given")


def _check_s3(name, a):
    _in_set(name, "last_touch_team", a["last_touch_team"], (UNKNOWN, HOME, AWAY))
    no_touch = a["last_touch_team"] == UNKNOWN
    if ((a["frames_since_touch"] < 0) != no_touch).any():
        _fail(name, "frames_since_touch must be -1 exactly where last_touch_team is -1")
    for key in ("touch_ll", "control"):
        if not np.isfinite(a[key]).all():
            _fail(name, "{} must be finite everywhere (use 0 for 'no evidence')".format(key))


def _check_s4(name, a):
    _in_set(name, "label", a["label"], (H, A, D, X))
    _in_unit(name, "prob", a["prob"])
    answered = a["label"] != X
    rows = a["prob"][answered]
    if np.isnan(rows).any():
        _fail(name, "prob is NaN on a frame with a label")
    if rows.size and np.abs(rows.sum(axis=1) - 1.0).max() > 1e-3:
        _fail(name, "prob rows must sum to 1 where a label is given")


_CHECKS = {"s0_tracking": _check_s0, "s1_inplay": _check_s1, "s2_ball": _check_s2,
           "s3_touch": _check_s3, "s4_possession": _check_s4}


def validate_table(name, table, n=None):
    """Check one event table against its column list and vocabularies."""
    if name not in TABLE_SCHEMAS:
        raise KeyError("no table contract {!r}; have {}".format(name, sorted(TABLE_SCHEMAS)))
    columns = TABLE_SCHEMAS[name]
    missing = [c for c in columns if c not in table.columns]
    if missing:
        _fail(name, "missing columns {}".format(missing))

    def frames_ok(column, upper):
        values = table[column].values
        if len(values) and (values.min() < 0 or (n is not None and values.max() > upper)):
            _fail(name, "{} outside the video".format(column))

    if name in ("s0_tags", "s1_stoppages"):
        frames_ok("start_frame", n)
        frames_ok("end_frame", n)
        if (table["end_frame"].values < table["start_frame"].values).any():
            _fail(name, "end_frame before start_frame")
    if name == "s0_tags":
        _in_set(name, "team", table["team"].values, (UNKNOWN, HOME, AWAY))
        _in_set(name, "half", table["half"].values, (0, 1, 2))
    if name == "s1_stoppages":
        _in_set(name, "cause", table["cause"].values.astype(str), STOP_CAUSES)
        _in_set(name, "restart_type", table["restart_type"].values.astype(str), RESTART_TYPES)
        _in_set(name, "restart_team", table["restart_team"].values, (UNKNOWN, HOME, AWAY))
        _in_unit(name, "conf", table["conf"].values.astype(float))
        starts = table["start_frame"].values
        if (np.diff(starts) <= 0).any():
            _fail(name, "rows must be sorted by start_frame, one per stoppage")
        if (starts[1:] < table["end_frame"].values[:-1]).any():
            _fail(name, "stoppages overlap")
    if name == "s3_touches":
        frames_ok("frame", None if n is None else n - 1)
        _in_set(name, "team", table["team"].values, (UNKNOWN, HOME, AWAY))
        _in_set(name, "kind", table["kind"].values.astype(str), TOUCH_KINDS)
        _in_unit(name, "conf", table["conf"].values.astype(float))
        _in_unit(name, "p_home", table["p_home"].values.astype(float))
        if (np.diff(table["frame"].values) < 0).any():
            _fail(name, "rows must be sorted by frame")
    return table


# -- cross-stage guarantees ---------------------------------------------------------------------

def check_s1_against_s0(s1, s0):
    """alive is -1 exactly on ungraded frames."""
    if ((s1["alive"] < 0) != (s0["half"] == 0)).any():
        _fail("s1_inplay", "alive must be -1 exactly where s0.half is 0")


def check_stoppages_against_s1(stoppages, s1):
    """The table and the per-frame array describe the same dead intervals."""
    dead = np.zeros(len(s1["alive"]), dtype=bool)
    for start, end in zip(stoppages["start_frame"].values, stoppages["end_frame"].values):
        dead[start:end] = True
    if (dead != (s1["alive"] == 0)).any():
        _fail("s1_stoppages", "intervals do not match the frames where s1.alive is 0")


def check_s2_against_s1(s2, s1):
    """No estimated ball position inside a stoppage (PLAN.md, stage 2 guarantees)."""
    bad = (s1["alive"] == 0) & (s2["ball_src"] > SRC_OBSERVED)
    if bad.any():
        _fail("s2_ball", "{} dead frames carry a corrected or estimated ball; only ball_src 0/1 "
                         "is allowed there".format(int(bad.sum())))


def check_s4_against_s0(s4, s0):
    """X exactly on ungraded frames: never abstain on a graded one (PLAN.md section 0)."""
    if ((s4["label"] == X) != (s0["half"] == 0)).any():
        _fail("s4_possession", "label must be X exactly where s0.half is 0")


# -- reading and writing ------------------------------------------------------------------------

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def path_of(name, stage_dir=None):
    extension = ".npz" if name in NPZ_SCHEMAS else ".csv"
    return os.path.join(stage_dir or DEFAULT_STAGE_DIR, name + extension)


def save(name, stage_dir, arrays, params=None, n=None):
    """Validate and write one .npz stage file. Returns its path."""
    arrays = {key: value for key, value in arrays.items() if key != "meta"}
    validate(name, arrays, n)
    meta = {"stage": name, "git_sha": git_sha(), "params": params or {}}
    path = path_of(name, stage_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return path


def load(name, stage_dir=None, n=None):
    """Read and validate one .npz stage file. Returns {array name: ndarray, "meta": dict}."""
    path = path_of(name, stage_dir)
    if not os.path.isfile(path):
        raise IOError("{} not found. Run the stage that writes it first "
                      "(python -m solution.pipeline.run_all).".format(path))
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files if key != "meta"}
        meta = json.loads(str(handle["meta"])) if "meta" in handle.files else {}
    validate(name, arrays, n)
    arrays["meta"] = meta
    return arrays


def save_table(name, stage_dir, table, n=None):
    validate_table(name, table, n)
    path = path_of(name, stage_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_table(name, stage_dir=None, n=None):
    path = path_of(name, stage_dir)
    if not os.path.isfile(path):
        raise IOError("{} not found. Run the stage that writes it first.".format(path))
    return validate_table(name, pd.read_csv(path), n)


# -- small helpers every stage needs ------------------------------------------------------------

def runs(mask):
    """[(start, end)] half-open runs where a boolean array is True."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def alive_segments(alive):
    """[(start, end)] maximal runs of in-play frames. Never spans a stoppage or halftime."""
    return runs(np.asarray(alive) == 1)


def write_submission(path, label_codes):
    """frame,label CSV from an array of codes (H, A, D, X = 0..3)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    letters = np.asarray(LABELS)[np.asarray(label_codes)]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, letter) for i, letter in enumerate(letters))
    return path
