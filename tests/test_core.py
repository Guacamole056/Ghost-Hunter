import unittest

import numpy as np

from ghost_hunter.agents import HunterAgent, capture_threshold
from ghost_hunter.core import (
    EpisodeScenario, GameSession, GhostEnvironment, HMMTracker,
    NoisySensor, cell_index, distance_matrix, manhattan, neighbors, position_for,
)


class TestGridAndEnvironment(unittest.TestCase):
    def test_index_inverse(self):
        for n in (2, 5, 10):
            for r in range(n):
                for c in range(n):
                    self.assertEqual(position_for(cell_index((r, c), n), n), (r, c))

    def test_edges_and_neighbors(self):
        self.assertEqual(sorted(neighbors((0, 0), 10)), [(0, 1), (1, 0)])
        self.assertEqual(len(neighbors((5, 5), 10)), 4)
        env = GhostEnvironment(grid_size=10, seed=42, hunter_start=(0, 0))
        self.assertFalse(env.move_hunter("left"))
        self.assertEqual(env.hunter_position, (0, 0))
        self.assertTrue(env.move_hunter("right"))
        self.assertEqual(env.hunter_position, (0, 1))

    def test_ghost_must_move_one_cardinal_cell(self):
        env = GhostEnvironment(grid_size=10)
        env.reset(ghost_position=(3, 3))
        env.step_ghost((3, 4))
        self.assertEqual(env.ghost_position, (3, 4))
        with self.assertRaises(ValueError):
            env.step_ghost((3, 6))


class TestSensorAndHmm(unittest.TestCase):
    def test_sensor_probabilities_and_clipping(self):
        sensor = NoisySensor(accuracy=0.6, grid_size=10)
        self.assertAlmostEqual(sum(sensor.noise_distribution.values()), 1)
        self.assertAlmostEqual(sensor.likelihood(0, 0), 0.6 + 0.4/3 + 0.4/6)
        for true_distance in range(sensor.max_distance + 1):
            self.assertAlmostEqual(sum(sensor.likelihood(observed, true_distance)
                                       for observed in range(sensor.max_distance + 1)), 1)
        self.assertEqual(sensor.read_distance((0, 0), (0, 0), noise_offset=-2), 0)
        self.assertEqual(sensor.read_distance((0, 0), (9, 9), noise_offset=2), 18)

    def test_transition_rows_normalized_and_only_adjacent(self):
        tracker = HMMTracker(10, (5, 5))
        transition = tracker.transition_matrix
        np.testing.assert_allclose(transition.sum(axis=1), np.ones(100))
        for idx in range(100):
            start = position_for(idx, 10)
            for next_index in np.flatnonzero(transition[idx]):
                self.assertEqual(manhattan(start, position_for(int(next_index), 10)), 1)
        self.assertAlmostEqual(tracker.belief.sum(), 1)
        self.assertEqual(tracker.belief[cell_index((5, 5), 10)], 0)

    def test_forward_update_normalized_multiple_steps(self):
        tracker = HMMTracker(10, (5, 5))
        sensor = NoisySensor(0.6, 10)
        tracker.update(4, (5, 5), sensor)
        self.assertAlmostEqual(tracker.belief.sum(), 1)
        for i in range(20):
            tracker.predict()
            tracker.update(i % 10, (3, 3), sensor)
            self.assertAlmostEqual(tracker.belief.sum(), 1, places=12)
            self.assertGreaterEqual(float(tracker.belief.min()), 0)

    def test_failed_capture_excludes_cell(self):
        tracker = HMMTracker(10, (5, 5))
        tracker.exclude_cell((6, 6))
        self.assertAlmostEqual(float(tracker.belief.sum()), 1)
        self.assertEqual(tracker.belief[cell_index((6, 6), 10)], 0)

    def test_reproducible_sampling(self):
        sensor = NoisySensor(0.6, 10)
        s1 = EpisodeScenario.generate(master_seed=42, episode_id=5,
                                      grid_size=10, max_turns=80,
                                      hunter_start=(5, 5), sensor=sensor)
        s2 = EpisodeScenario.generate(master_seed=42, episode_id=5,
                                      grid_size=10, max_turns=80,
                                      hunter_start=(5, 5), sensor=sensor)
        s3 = EpisodeScenario.generate(master_seed=42, episode_id=6,
                                      grid_size=10, max_turns=80,
                                      hunter_start=(5, 5), sensor=sensor)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertEqual(len(s1.ghost_path), 81)
        self.assertEqual(len(s1.noise_offsets), 81)
        for before, after in zip(s1.ghost_path, s1.ghost_path[1:]):
            self.assertEqual(manhattan(before, after), 1)


class TestSessionAndAgents(unittest.TestCase):
    def test_exact_capture_vs_miss_and_movement(self):
        s = GameSession(mode="human", grid_size=4, max_turns=4, seed=11)
        self.assertNotEqual(s.ghost_position, s.hunter_position)
        self.assertTrue(s.take_action("capture"))
        self.assertFalse(s.captured)
        self.assertEqual(s.turns, 1)
        self.assertEqual(s.capture_attempts, 1)
        self.assertTrue(s.take_action("left"))
        self.assertEqual(s.distance_traveled, 1)
        self.assertEqual(s.turns, 2)

        sensor = NoisySensor(0.6, 2)
        scenario = EpisodeScenario.generate(master_seed=10, episode_id=1,
                                            grid_size=2, max_turns=4,
                                            hunter_start=(0, 0), sensor=sensor)
        exact = EpisodeScenario(ghost_path=((0, 1), (0, 0), (0, 1), (0, 0), (0, 1)),
                                noise_offsets=(0, 0, 0, 0, 0), episode_seed=1)
        win = GameSession(mode="human", grid_size=2, max_turns=4,
                          hunter_start=(0, 0), scenario=exact)
        self.assertTrue(win.take_action("capture"))  # miss, ghost moves into our cell
        self.assertTrue(win.take_action("capture"))  # capture next turn
        self.assertTrue(win.finished)
        self.assertTrue(win.captured)
        self.assertEqual(win.turns, 2)

    def test_off_board_move_does_not_consume_turn(self):
        s = GameSession(mode="human", hunter_start=(0, 0))
        self.assertFalse(s.take_action("up"))
        self.assertEqual(s.turns, 0)

    def test_baseline_uses_only_current_reading(self):
        s = GameSession(mode="baseline")
        original = s.sensor_only_belief()
        s.tracker.belief = np.zeros(s.grid_size ** 2)
        s.tracker.belief[0] = 1
        np.testing.assert_allclose(s.sensor_only_belief(), original)

    def test_agent_never_reads_true_ghost_cell(self):
        # Agent receives a session; actions must be based on public observation.
        for mode in ("ai", "baseline"):
            s = GameSession(mode=mode, episode_id=3)
            agent = HunterAgent(mode)
            for _ in range(12):
                if s.finished:
                    break
                a = agent.choose_action(s)
                self.assertIn(a, ("up", "down", "left", "right", "capture"))
                self.assertTrue(s.take_action(a))
        # Static inspection guards against accidental use of ground-truth coordinates.
        import inspect
        self.assertNotIn("ghost_position", inspect.getsource(HunterAgent.choose_action))

    def test_vectorized_distance_and_likelihood_match_scalar(self):
        n = 7
        dm = distance_matrix(n)
        for i in range(n * n):
            for j in range(n * n):
                self.assertEqual(int(dm[i, j]), manhattan(position_for(i, n), position_for(j, n)))
        sensor = NoisySensor(0.6, n)
        for reading in (0, 3, sensor.max_distance):
            vec = sensor.likelihood_vector(reading, (2, 4))
            expected = np.array([
                sensor.likelihood(reading, manhattan((2, 4), position_for(i, n)))
                for i in range(n * n)
            ])
            np.testing.assert_allclose(vec, expected)

    def test_grid_size_aware_capture_threshold_and_biased_ghost(self):
        self.assertAlmostEqual(capture_threshold(10), 0.06)
        self.assertGreater(capture_threshold(5), capture_threshold(10))
        self.assertLess(capture_threshold(16), capture_threshold(10))
        sensor = NoisySensor(0.6, 8)
        biased = EpisodeScenario.generate(
            master_seed=42, episode_id=0, grid_size=8, max_turns=30,
            hunter_start=(4, 4), sensor=sensor, ghost_bias=0.7)
        for before, after in zip(biased.ghost_path, biased.ghost_path[1:]):
            self.assertEqual(manhattan(before, after), 1)


if __name__ == "__main__":
    unittest.main()
