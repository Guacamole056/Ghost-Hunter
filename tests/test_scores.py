import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ghost_hunter.core import GameSession
from ghost_hunter.scores import InvalidScoreHistory, append_score, read_scores
from ghost_hunter.ui import GameUI


def save_seed(path, seed):
    append_score(path, {"captured": False, "seed": seed})


class TestHumanScores(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "human_scores.json"

    def test_missing_file_and_existing_scores(self):
        self.assertEqual(read_scores(self.path), [])
        self.assertIsNone(append_score(self.path, {"captured": False}))
        win = {"captured": True, "turns": 4, "distance": 3}
        self.assertIsNone(append_score(self.path, win))
        self.assertEqual(read_scores(self.path), [{"captured": False}, win])
        self.assertEqual(list(self.path.parent.glob("*.bak")), [])

    def test_finish_recovers_bad_json_and_saves_once(self):
        original = b'[{ "captured":'
        self.path.write_bytes(original)
        # Exercise the game's terminal turn without opening a display.
        game = GameUI.__new__(GameUI)
        game.session = GameSession(mode="human", max_turns=1, seed=42)
        game.seed = 42
        game._result_saved = False
        with patch.object(game, "_human_score_path", return_value=self.path):
            game.take_turn("capture")
            game._save_human_score()
        self.assertTrue(game.session.finished)
        self.assertTrue(game._result_saved)
        self.assertIn("backed up", game.status)
        [backup] = self.path.parent.glob("*.bak")
        self.assertEqual(backup.read_bytes(), original)
        [saved] = read_scores(self.path)
        self.assertEqual(saved["turns"], 1)
        self.assertEqual(saved["seed"], 42)

    def test_recovers_valid_older_rows_from_mixed_history(self):
        older = {"captured": True, "turns": 6, "distance": 3, "extra": "preserve"}
        data = [older, {"captured": True, "turns": "bad"}, None, {"captured": False}]
        original = json.dumps(data).encode()
        self.path.write_bytes(original)
        with self.assertRaises(InvalidScoreHistory) as invalid:
            read_scores(self.path)
        self.assertEqual(invalid.exception.usable_rows, [older, {"captured": False}])
        new = {"captured": True, "turns": 2, "distance": 1}
        backup = append_score(self.path, new)
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(read_scores(self.path), [older, {"captured": False}, new])

    def test_each_damaged_file_has_its_own_backup(self):
        for data in (b'{"wrong": "shape"}', b'\xff\x00'):
            self.path.write_bytes(data)
            backup = append_score(self.path, {"captured": False})
            self.assertEqual(backup.read_bytes(), data)
        self.assertEqual(len(list(self.path.parent.glob("*.bak"))), 2)

    def test_write_failure_keeps_existing_history_and_cleans_temp_file(self):
        original = b'[{"captured": false}]'
        self.path.write_bytes(original)
        with patch.object(Path, "replace", side_effect=PermissionError("read-only")):
            with self.assertRaises(PermissionError):
                append_score(self.path, {"captured": False})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_independent_processes_keep_all_scores(self):
        with multiprocessing.get_context("spawn").Pool(3) as pool:
            pool.starmap(save_seed, [(self.path, seed) for seed in range(10)])
        self.assertEqual(sorted(row["seed"] for row in read_scores(self.path)), list(range(10)))


if __name__ == "__main__":
    unittest.main()
