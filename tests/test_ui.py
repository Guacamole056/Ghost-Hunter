import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from ghost_hunter.ui import GameUI


class TestGameUI(unittest.TestCase):
    def setUp(self):
        self.ui = GameUI(mode="human", grid_size=6, max_turns=10,
                         sensor_accuracy=0.60, seed=42)
        self.ui.draw()

    def tearDown(self):
        pygame.quit()

    def test_modal_scoreboard_blocks_underlying_clicks_and_keys(self):
        self.assertIn("restart", self.ui.hitboxes)
        restart_center = self.ui.hitboxes["restart"].center

        self.ui.show_scores = True
        self.ui.draw()
        # Only modal close button should be registered while scoreboard is open.
        self.assertEqual(set(self.ui.hitboxes.keys()), {"close_scores"})

        # Gameplay keys should not act behind the scoreboard.
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        self.ui.handle_events()
        self.assertEqual(self.ui.session.turns, 0)
        self.assertEqual(self.ui.session.mode, "human")
        self.assertTrue(self.ui.show_scores)

        # Clicking inside the modal where the underlying restart button sat should NOT restart.
        # Clicking close_scores closes the modal without touching the game state.
        close_center = self.ui.hitboxes["close_scores"].center
        view = self.ui._viewport()
        win_pos = (
            round(view.x + close_center[0] * view.width / self.ui.WIDTH),
            round(view.y + close_center[1] * view.height / self.ui.HEIGHT),
        )
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=win_pos))
        self.ui.handle_events()
        self.assertFalse(self.ui.show_scores)
        self.assertEqual(self.ui.session.turns, 0)

        # ESC closes modal first rather than exiting the app.
        self.ui.show_scores = True
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.ui.handle_events()
        self.assertFalse(self.ui.show_scores)
        self.assertTrue(self.ui.running)
        self.assertIsNotNone(restart_center)

    def test_accuracy_preset_updates_sensor_and_restarts(self):
        self.ui.take_turn("capture")
        self.assertEqual(self.ui.session.turns, 1)
        self.ui._click("acc_0.80")
        self.assertAlmostEqual(self.ui.sensor_accuracy, 0.80)
        self.assertAlmostEqual(self.ui.session.sensor.accuracy, 0.80)
        self.assertEqual(self.ui.session.turns, 0)

    def test_autoplay_speed_controls_and_clamping(self):
        self.ui.new_game("ai")
        self.ui.draw()
        self.assertIn("speed_slower", self.ui.hitboxes)
        self.assertIn("speed_faster", self.ui.hitboxes)
        start = self.ui.auto_interval
        self.ui._click("speed_slower")
        self.assertEqual(self.ui.auto_interval, start + 50)
        self.ui._click("speed_faster")
        self.assertEqual(self.ui.auto_interval, start)
        for _ in range(30):
            self.ui._click("speed_faster")
        self.assertEqual(self.ui.auto_interval, 50)

    def test_human_score_tracks_heatmap_usage(self):
        with TemporaryDirectory() as tmp:
            score_file = Path(tmp) / "human_scores.json"
            self.ui._human_score_path = lambda: score_file
            self.ui.heatmap_enabled = False
            self.ui.new_game("human")
            while not self.ui.session.finished:
                self.ui.take_turn("capture")
            saved = json.loads(score_file.read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 1)
            self.assertFalse(saved[0]["heatmap_used"])
            self.assertAlmostEqual(saved[0]["sensor_accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
