"""Write a full-length submission and run the unmodified official validator."""

import argparse
import csv
from itertools import repeat
from pathlib import Path
import subprocess
import sys

import pandas as pd

from evaluation.possession_metrics import validate_submission as validate_table
from games import GAME, GAMES


ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = GAMES[GAME]["n_video_frames"]
VALIDATOR = ROOT / "evaluation" / "validate_submission.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "smoke_test.csv"


def write_submission(labels, output, overwrite=False):
    """Write one label per tactical frame; reject invalid input before opening output.

    Labels must already be in frame order and cover the entire video. Missing
    rows are never padded. Existing files require explicit overwrite permission.
    """
    labels = [label.strip().upper() if isinstance(label, str) else label
              for label in labels]
    table = pd.DataFrame({"frame": range(len(labels)), "label": labels})
    reason = validate_table(table, N_FRAMES)
    if reason:
        raise ValueError(reason)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels))
    return output


def run_official_validator(path):
    """Validate the exact saved file; raise if the official command fails."""
    subprocess.run(
        [sys.executable, str(VALIDATOR), str(Path(path).resolve())],
        check=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true", required=True,
        help="generate X for every frame to check the file pipeline without match data",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV destination (default: outputs/smoke_test.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output file")
    args = parser.parse_args(argv)

    try:
        path = write_submission(repeat("X", N_FRAMES), args.output, args.overwrite)
        print("Wrote {}: {} abstentions (X). Format smoke test only."
              .format(path, N_FRAMES), flush=True)
        run_official_validator(path)
    except FileExistsError:
        print("Output already exists: {}. Choose another path or use --overwrite."
              .format(args.output), file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print("Submission failed: {}".format(error), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
