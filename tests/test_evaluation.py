import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ghost_hunter.evaluation import run_evaluation


class TestEvaluation(unittest.TestCase):
    def test_small_paired_run_outputs_all_required_artifacts(self):
        with TemporaryDirectory() as tmp:
            data = run_evaluation(episodes=4, seed=42, grid_size=5,
                                  max_turns=15, output_dir=tmp)
            self.assertEqual(data["ai"]["episodes"], 4)
            self.assertEqual(data["baseline"]["episodes"], 4)
            self.assertTrue(data["configuration"]["ghost_trajectories_paired"])
            for name in ("summary.json", "episodes.csv", "localization_by_turn.csv",
                         "01_capture_rate.png", "02_mean_turns.png",
                         "03_localization_error.png"):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            with open(Path(tmp) / "episodes.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            by_mode = {mode: {int(row["episode_id"]) for row in rows
                              if row["mode"] == mode} for mode in ("ai", "baseline")}
            self.assertEqual(by_mode["ai"], by_mode["baseline"])
            self.assertEqual(json.loads((Path(tmp) / "summary.json").read_text()), data)


if __name__ == "__main__":
    unittest.main()
