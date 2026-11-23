"""Game rules, noisy observations, and a from-scratch HMM forward filter."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

Position = tuple[int, int]  # (row, column), both zero-indexed
Action = Literal["up", "down", "left", "right", "capture"]
DIRECTIONS: dict[str, Position] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)
}


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos: Position, n: int) -> list[Position]:
    r, c = pos
    return [(r + dr, c + dc) for dr, dc in DIRECTIONS.values()
            if 0 <= r + dr < n and 0 <= c + dc < n]


def cell_index(pos: Position, n: int) -> int:
    return pos[0] * n + pos[1]


def position_for(index: int, n: int) -> Position:
    return divmod(index, n)


@lru_cache(maxsize=None)
def distance_matrix(n: int) -> np.ndarray:
    """Precomputed (n*n, n*n) pairwise Manhattan distance matrix."""
    coords = np.array([position_for(i, n) for i in range(n * n)], dtype=np.int16)
    matrix = (np.abs(coords[:, None, 0] - coords[None, :, 0])
              + np.abs(coords[:, None, 1] - coords[None, :, 1]))
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=None)
def transition_matrix_for(n: int) -> np.ndarray:
    """Precomputed row-stochastic transition matrix for a uniform cardinal walk."""
    size = n * n
    matrix = np.zeros((size, size), dtype=np.float64)
    for idx in range(size):
        adjacent = neighbors(position_for(idx, n), n)
        weight = 1.0 / len(adjacent)
        for next_pos in adjacent:
            matrix[idx, cell_index(next_pos, n)] = weight
    matrix.setflags(write=False)
    return matrix


class NoisySensor:
    """Clipped Manhattan-distance sensor with exactly modeled discrete noise."""

    def __init__(self, accuracy: float = 0.60, grid_size: int = 10):
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("Sensor accuracy must be in [0, 1].")
        if grid_size < 2:
            raise ValueError("Grid size must be >= 2.")
        self.accuracy = accuracy
        self.grid_size = grid_size
        remainder = 1.0 - accuracy
        self.noise_distribution = {
            -2: remainder / 6, -1: remainder / 3,
            0: accuracy, 1: remainder / 3, 2: remainder / 6,
        }
        self._offsets = np.array(list(self.noise_distribution.keys()), dtype=np.int64)
        self._probs = np.array(list(self.noise_distribution.values()), dtype=np.float64)
        span = self.max_distance + 1
        table = np.zeros((span, span), dtype=np.float64)
        for true_d in range(span):
            for offset, prob in self.noise_distribution.items():
                table[self.clip(true_d + offset), true_d] += prob
        table.setflags(write=False)
        self._likelihood_table = table
        # H(Z | d) for each true distance d (used for optional mutual information).
        cond_h = np.zeros(span, dtype=np.float64)
        for true_d in range(span):
            col = table[:, true_d]
            pos = col[col > 0.0]
            cond_h[true_d] = float(-np.sum(pos * np.log2(pos)))
        cond_h.setflags(write=False)
        self.conditional_entropy_by_distance = cond_h

    @property
    def max_distance(self) -> int:
        return 2 * (self.grid_size - 1)

    def clip(self, reading: int) -> int:
        return max(0, min(self.max_distance, int(reading)))

    def noise_sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(self._offsets, p=self._probs))

    def read_distance(self, hunter: Position, ghost: Position,
                      rng: np.random.Generator | None = None,
                      *, noise_offset: int | None = None) -> int:
        if noise_offset is None:
            if rng is None:
                raise ValueError("Supply rng or a predetermined noise_offset.")
            noise_offset = self.noise_sample(rng)
        return self.clip(manhattan(hunter, ghost) + noise_offset)

    def likelihood(self, reading: int, true_distance: int) -> float:
        """P(observed reading | true distance), including clipping at endpoints."""
        if not (0 <= reading <= self.max_distance):
            return 0.0
        if 0 <= true_distance <= self.max_distance:
            return float(self._likelihood_table[reading, true_distance])
        return sum(p for offset, p in self.noise_distribution.items()
                   if self.clip(true_distance + offset) == reading)

    def likelihood_vector(self, reading: int, hunter: Position) -> np.ndarray:
        """Vectorized P(reading | ghost at cell i) across all grid cells."""
        if not (0 <= reading <= self.max_distance):
            return np.zeros(self.grid_size ** 2, dtype=np.float64)
        dists = distance_matrix(self.grid_size)[cell_index(hunter, self.grid_size)]
        return self._likelihood_table[reading, dists]


class HMMTracker:
    """Transition and observation models with a normalized forward update.

    Belief is a vector of size n*n, with row-major indexing.
    T[i,j] = P(X_next=j | X_current=i); the ghost moves one cardinal cell.
    """

    def __init__(self, grid_size: int = 10, hunter_start: Position = (0, 0)):
        if grid_size < 2:
            raise ValueError("Grid size must be >= 2.")
        self.grid_size = grid_size
        self.transition_matrix = transition_matrix_for(grid_size).copy()
        self.belief = np.full(grid_size ** 2, 1.0 / (grid_size ** 2))
        self.reset(hunter_start)

    def reset(self, hunter_start: Position | None = None) -> None:
        size = self.grid_size ** 2
        self.belief = np.full(size, 1.0 / size, dtype=np.float64)
        if hunter_start is not None:
            idx = cell_index(hunter_start, self.grid_size)
            self.belief[idx] = 0.0  # guaranteed different ghost start
            self.belief /= self.belief.sum()

    def predict(self) -> np.ndarray:
        self.belief = self.belief @ self.transition_matrix
        return self.belief.copy()

    def update(self, reading: int, hunter: Position, sensor: NoisySensor) -> np.ndarray:
        likelihoods = sensor.likelihood_vector(reading, hunter)
        weighted = self.belief * likelihoods
        total = float(weighted.sum())
        if total <= 1e-15:
            # Guard against impossible evidence caused by a faulty caller/model.
            weighted = likelihoods
            total = float(weighted.sum())
        if total <= 1e-15:
            weighted = np.ones(self.grid_size ** 2, dtype=np.float64)
            total = float(weighted.sum())
        self.belief = weighted / total
        return self.belief.copy()

    def exclude_cell(self, position: Position) -> None:
        """A failed capture is certain evidence that the ghost isn't here *now*."""
        self.belief[cell_index(position, self.grid_size)] = 0.0
        total = self.belief.sum()
        if total > 0.0:
            self.belief /= total
        else:
            self.reset(None)
            self.belief[cell_index(position, self.grid_size)] = 0.0
            self.belief /= self.belief.sum()

    def most_likely_cell(self) -> Position:
        return position_for(int(np.argmax(self.belief)), self.grid_size)


class GhostEnvironment:
    """Keeps actual hidden state separate from what the hunter can observe."""

    def __init__(self, grid_size: int = 10, seed: int = 42,
                 hunter_start: Position = (0, 0)):
        if grid_size < 2:
            raise ValueError("Grid size must be >= 2.")
        self.grid_size = grid_size
        self.random_generator = np.random.default_rng(seed)
        self.hunter_start = hunter_start
        self.hunter_position = hunter_start
        self.ghost_position: Position = (0, 0)
        self.reset()

    def reset(self, ghost_position: Position | None = None,
              hunter_start: Position | None = None) -> None:
        if hunter_start is not None:
            self.hunter_start = hunter_start
        self.hunter_position = self.hunter_start
        if ghost_position is None:
            eligible = [position_for(i, self.grid_size)
                        for i in range(self.grid_size ** 2)
                        if position_for(i, self.grid_size) != self.hunter_position]
            ghost_position = eligible[int(self.random_generator.integers(len(eligible)))]
        self.ghost_position = ghost_position

    def step_ghost(self, predefined: Position | None = None) -> Position:
        candidates = neighbors(self.ghost_position, self.grid_size)
        if predefined is not None:
            if predefined not in candidates:
                raise ValueError("Ghost must move to a neighboring cardinal cell.")
            self.ghost_position = predefined
        else:
            self.ghost_position = candidates[int(self.random_generator.integers(len(candidates)))]
        return self.ghost_position

    def move_hunter(self, direction: str) -> bool:
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown movement direction: {direction}")
        dr, dc = DIRECTIONS[direction]
        r, c = self.hunter_position
        to = (r + dr, c + dc)
        if not (0 <= to[0] < self.grid_size and 0 <= to[1] < self.grid_size):
            return False
        self.hunter_position = to
        return True

    def check_capture(self) -> bool:
        return self.hunter_position == self.ghost_position


@dataclass(frozen=True)
class EpisodeScenario:
    """Fixed ghost path and noise offsets for fair paired strategy comparisons."""

    ghost_path: tuple[Position, ...]
    noise_offsets: tuple[int, ...]
    episode_seed: int

    @classmethod
    def generate(cls, *, master_seed: int, episode_id: int, grid_size: int,
                 max_turns: int, hunter_start: Position,
                 sensor: NoisySensor, ghost_bias: float = 0.0) -> "EpisodeScenario":
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if not 0.0 <= ghost_bias <= 1.0:
            raise ValueError("ghost_bias must be in [0, 1]")
        ghost_rng, noise_rng = [np.random.default_rng(seed)
                                for seed in np.random.SeedSequence(
                                    [master_seed, episode_id]).spawn(2)]
        eligible = [position_for(i, grid_size) for i in range(grid_size ** 2)
                    if position_for(i, grid_size) != hunter_start]
        start = eligible[int(ghost_rng.integers(len(eligible)))]
        path = [start]
        for _ in range(max_turns):
            options = neighbors(path[-1], grid_size)
            if ghost_bias == 0.0 or len(path) < 2 or len(options) == 1:
                path.append(options[int(ghost_rng.integers(len(options)))])
            else:
                # Non-backtracking momentum bias for model-mismatch robustness tests:
                # reduces the weight of immediately returning to path[-2].
                prev = path[-2]
                weights = np.array(
                    [1.0 - ghost_bias if opt == prev else 1.0 for opt in options],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                choice_idx = int(ghost_rng.choice(len(options), p=weights))
                path.append(options[choice_idx])
        noise = tuple(sensor.noise_sample(noise_rng) for _ in range(max_turns + 1))
        return cls(tuple(path), noise, episode_id)


@dataclass
class TurnRecord:
    turn: int
    action: str
    reading: int
    hunter_position: Position
    ghost_position: Position
    localization_error: int
    true_cell_probability: float
    entropy: float


class GameSession:
    """One complete episode: no hidden ghost coordinates exposed to agents."""

    MODES = ("human", "ai", "baseline")

    def __init__(self, *, mode: str = "human", grid_size: int = 10,
                 max_turns: int = 80, sensor_accuracy: float = 0.60,
                 seed: int = 42, episode_id: int = 0,
                 hunter_start: Position | None = None,
                 scenario: EpisodeScenario | None = None):
        if mode not in self.MODES:
            raise ValueError(f"Mode must be one of {self.MODES}")
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.mode = mode
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.seed = seed
        self.episode_id = episode_id
        self.hunter_start = hunter_start if hunter_start is not None else (grid_size // 2, grid_size // 2)
        if not all(0 <= v < grid_size for v in self.hunter_start):
            raise ValueError("Hunter starting position outside grid")
        self.sensor = NoisySensor(sensor_accuracy, grid_size)
        self.scenario = scenario or EpisodeScenario.generate(
            master_seed=seed, episode_id=episode_id, grid_size=grid_size,
            max_turns=max_turns, hunter_start=self.hunter_start, sensor=self.sensor)
        if len(self.scenario.ghost_path) < max_turns + 1 or len(self.scenario.noise_offsets) < max_turns + 1:
            raise ValueError("Scenario must contain initial state and all turns")
        self.environment = GhostEnvironment(grid_size=grid_size, seed=seed,
                                             hunter_start=self.hunter_start)
        self.environment.reset(ghost_position=self.scenario.ghost_path[0])
        self.tracker = HMMTracker(grid_size, self.hunter_start)
        self.turns = 0
        self.distance_traveled = 0
        self.capture_attempts = 0
        self.captured = False
        self.finished = False
        self.last_action = "start"
        self.last_capture_missed = False
        self.reading = self.sensor.read_distance(
            self.environment.hunter_position, self.environment.ghost_position,
            noise_offset=self.scenario.noise_offsets[0])
        self._baseline_cache: tuple[tuple[int, int, Position], np.ndarray] | None = None
        if self.mode != "baseline":
            self.tracker.update(self.reading, self.environment.hunter_position, self.sensor)
        self.history: list[TurnRecord] = []
        self._record()

    @property
    def hunter_position(self) -> Position:
        return self.environment.hunter_position

    @property
    def ghost_position(self) -> Position:
        """For evaluation and post-game reveal; never used by decision agents."""
        return self.environment.ghost_position

    def sensor_only_belief(self) -> np.ndarray:
        """Inference from the current reading and a flat prior, no past observations."""
        cache_key = (self.turns, self.reading, self.hunter_position)
        if self._baseline_cache is not None and self._baseline_cache[0] == cache_key:
            return self._baseline_cache[1].copy()
        weights = self.sensor.likelihood_vector(self.reading, self.hunter_position)
        total = float(weights.sum())
        belief = (weights / total if total > 0.0
                  else np.full(len(weights), 1.0 / len(weights), dtype=np.float64))
        self._baseline_cache = (cache_key, belief)
        return belief.copy()

    @property
    def visible_belief(self) -> np.ndarray:
        if self.mode == "baseline":
            return self.sensor_only_belief()
        return self.tracker.belief.copy()

    def _record(self) -> None:
        belief = self.visible_belief
        inferred = position_for(int(np.argmax(belief)), self.grid_size)
        positive = belief[belief > 0]
        self.history.append(TurnRecord(
            turn=self.turns, action=self.last_action, reading=self.reading,
            hunter_position=self.hunter_position, ghost_position=self.ghost_position,
            localization_error=manhattan(inferred, self.ghost_position),
            true_cell_probability=float(belief[cell_index(self.ghost_position, self.grid_size)]),
            entropy=float(-np.sum(positive * np.log2(positive))),
        ))

    def take_action(self, action: str) -> bool:
        """Use one turn: act at time t, then ghost moves, then observe time t+1.

        A successful capture tests the ghost's position *before* its next move.
        Returns False for invalid/off-grid movement (doesn't use a turn).
        """
        if self.finished:
            return False
        if action not in (*DIRECTIONS.keys(), "capture"):
            raise ValueError(f"Invalid action: {action}")
        self.last_capture_missed = False
        if action == "capture":
            self.capture_attempts += 1
            if self.environment.check_capture():
                self.turns += 1
                self.captured = True
                self.finished = True
                self.last_action = action
                self._record()
                return True
            self.last_capture_missed = True
            if self.mode != "baseline":
                self.tracker.exclude_cell(self.hunter_position)
        else:
            moved = self.environment.move_hunter(action)
            if not moved:
                return False
            self.distance_traveled += 1
        self.turns += 1
        self.last_action = action
        self.environment.step_ghost(self.scenario.ghost_path[self.turns])
        self.reading = self.sensor.read_distance(
            self.hunter_position, self.ghost_position,
            noise_offset=self.scenario.noise_offsets[self.turns])
        if self.mode != "baseline":
            self.tracker.predict()
            self.tracker.update(self.reading, self.hunter_position, self.sensor)
        self.finished = self.turns >= self.max_turns
        self._record()
        return True

    def result(self) -> dict:
        errors = [rec.localization_error for rec in self.history]
        return {
            "mode": self.mode,
            "episode_id": self.episode_id,
            "captured": self.captured,
            "turns": self.turns,
            "distance": self.distance_traveled,
            "capture_attempts": self.capture_attempts,
            "mean_localization_error": float(np.mean(errors)),
            "mean_true_cell_probability": float(np.mean(
                [rec.true_cell_probability for rec in self.history])),
            "mean_entropy": float(np.mean([rec.entropy for rec in self.history])),
        }
