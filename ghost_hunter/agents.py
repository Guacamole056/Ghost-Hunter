"""Fair paired policies: both use the same planner but different beliefs."""

from __future__ import annotations

import numpy as np

from .core import DIRECTIONS, GameSession, cell_index, distance_matrix


def capture_threshold(grid_size: int) -> float:
    """Grid-size-aware capture probability threshold (0.06 on a 10x10 board)."""
    return min(0.50, 6.0 / (grid_size * grid_size))


class HunterAgent:
    """Choose moves using a belief distribution, without seeing the true ghost.

    HMM strategy receives the forward-filtered posterior; the sensor-only
    baseline receives a new, memoryless distribution from the latest reading.
    Both strategies use identical capture thresholds and action scoring.
    """

    def __init__(self, strategy: str = "ai"):
        if strategy not in ("ai", "baseline"):
            raise ValueError("Strategy must be ai or baseline")
        self.strategy = strategy

    def choose_action(self, session: GameSession) -> str:
        if session.finished:
            raise ValueError("Game already finished")
        if session.mode != self.strategy:
            raise ValueError("Session and agent strategies must match")
        belief = session.visible_belief
        n = session.grid_size
        current = session.hunter_position
        current_probability = float(belief[cell_index(current, n)])

        # Attempt a capture at the current cell when its probability justifies it.
        # Captures are the ONLY way to win; simply walking onto the ghost isn't enough.
        if current_probability >= capture_threshold(n):
            return "capture"

        # One-step forecast is used ONLY to plan the next movement, not to carry
        # past measurements into the baseline's next belief. This keeps the
        # action policy identical across the two experimental conditions.
        forecast = belief @ session.tracker.transition_matrix
        dists = distance_matrix(n)
        actions: list[str] = []
        dest_indices: list[int] = []
        # 'capture' with low current probability acts as a stationary probe:
        # if it misses, the ghost moves and another noisy reading arrives.
        # Such unsuccessful probes still cost one turn and are counted.
        for action, (dr, dc) in {**DIRECTIONS, "capture": (0, 0)}.items():
            dest = (current[0] + dr, current[1] + dc)
            if 0 <= dest[0] < n and 0 <= dest[1] < n:
                actions.append(action)
                dest_indices.append(cell_index(dest, n))

        idx_arr = np.array(dest_indices, dtype=np.intp)
        forecast_probs = forecast[idx_arr]
        expected_distances = dists[idx_arr] @ forecast
        # Scale distance penalty inversely with board diameter (0.02 on 10x10).
        distance_weight = 0.36 / (2 * (n - 1))
        scores = 3.0 * forecast_probs - distance_weight * expected_distances

        # Round to 12 decimals so exact geometric ties break deterministically
        # regardless of floating-point summation order.
        options = [
            (round(float(sc), 12), round(float(prob), 12),
             round(-float(dist), 12), action)
            for sc, prob, dist, action in zip(
                scores, forecast_probs, expected_distances, actions)
        ]
        return max(options)[-1]
