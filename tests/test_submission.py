"""Submission contract checks using synthetic labels and the official validator."""

import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from solution import submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "submission.csv"

    def test_mixed_labels_preserve_order_and_pass_official_validator(self):
        labels = (["H", "A", "D", "X"] * (submission.N_FRAMES // 4 + 1))[
            :submission.N_FRAMES]
        submission.write_submission(iter(labels), self.output)

        with self.output.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            self.assertEqual(next(reader), ["frame", "label"])
            rows = list(reader)
        self.assertEqual(len(rows), submission.N_FRAMES)
        self.assertEqual([int(row[0]) for row in rows], list(range(submission.N_FRAMES)))
        self.assertEqual([row[1] for row in rows], labels)

        result = subprocess.run(
            [sys.executable, str(submission.VALIDATOR), str(self.output)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_invalid_input_never_replaces_an_existing_output(self):
        self.output.write_text("existing result\n", encoding="utf-8")
        cases = [
            ("too few", ["X"] * (submission.N_FRAMES - 1)),
            ("too many", ["X"] * (submission.N_FRAMES + 1)),
        ]
        for name, value in (("unknown", "Z"), ("empty", ""),
                            ("whitespace", " "), ("missing", None)):
            labels = ["X"] * submission.N_FRAMES
            labels[submission.N_FRAMES // 2] = value
            cases.append((name, labels))
        for name, labels in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    submission.write_submission(labels, self.output, overwrite=True)
                self.assertEqual(self.output.read_text(), "existing result\n")

    def test_smoke_cli_requires_explicit_overwrite_and_is_reproducible(self):
        command = [sys.executable, "-m", "solution.submission", "--smoke-test",
                   "--output", str(self.output)]
        first = subprocess.run(command, cwd=submission.ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("OK", first.stdout)
        original = self.output.read_bytes()
        with self.output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), submission.N_FRAMES)
        self.assertEqual({row["label"] for row in rows}, {"X"})

        second = subprocess.run(command, cwd=submission.ROOT, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already exists", second.stderr)
        self.assertEqual(self.output.read_bytes(), original)

        third = subprocess.run(command + ["--overwrite"], cwd=submission.ROOT,
                               capture_output=True, text=True)
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
        self.assertEqual(self.output.read_bytes(), original)

    def test_cli_propagates_official_validator_failure(self):
        with patch.object(submission, "run_official_validator",
                          side_effect=subprocess.CalledProcessError(1, ["validator"])):
            with contextlib.redirect_stdout(io.StringIO()):
                status = submission.main(["--smoke-test", "--output", str(self.output)])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
