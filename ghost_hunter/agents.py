"""Fair paired policies: both use the same planner but different beliefs."""

from __future__ import annotations

import numpy as np

from .core import DIRECTIONS, GameSession, manhattan, position_for, cell_index


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
        if current_probability >= 0.06:
            return "capture"

        # One-step forecast is used ONLY to plan the next movement, not to carry
        # past measurements into the baseline's next belief. This keeps the
        # action policy identical across the two experimental conditions.
        forecast = belief @ session.tracker.transition_matrix
        positions = [position_for(i, n) for i in range(n * n)]
        options = []
        # 'capture' with low current probability acts as a stationary probe:
        # if it misses, the ghost moves and another noisy reading arrives.
        # Such unsuccessful probes still cost one turn and are counted.
        for action, (dr, dc) in {**DIRECTIONS, "capture": (0, 0)}.items():
            dest = (current[0] + dr, current[1] + dc)
            if not (0 <= dest[0] < n and 0 <= dest[1] < n):
                continue
            forecast_probability = float(forecast[cell_index(dest, n)])
            distance_to_distribution = sum(
                float(prob) * manhattan(dest, pos)
                for prob, pos in zip(forecast, positions))
            # Prefer likely interception cells, then expected closeness.
            score = 3.0 * forecast_probability - 0.02 * distance_to_distribution
            options.append((score, forecast_probability,
                            -distance_to_distribution, action))
        return max(options)[-1]
