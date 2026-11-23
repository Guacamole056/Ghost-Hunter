"""Reproducible paired experiments and three required figures."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

from .agents import HunterAgent
from .core import EpisodeScenario, GameSession, NoisySensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def play_episode(mode: str, scenario: EpisodeScenario, *, master_seed: int,
                 grid_size: int, max_turns: int, sensor_accuracy: float) -> tuple[dict, list]:
    session = GameSession(mode=mode, grid_size=grid_size, max_turns=max_turns,
                          sensor_accuracy=sensor_accuracy, seed=master_seed,
                          episode_id=scenario.episode_seed, scenario=scenario)
    agent = HunterAgent(mode)
    start = time.perf_counter()
    while not session.finished:
        session.take_action(agent.choose_action(session))
    elapsed = (time.perf_counter() - start) * 1000
    result = session.result()
    result["runtime_ms"] = round(elapsed, 3)
    return result, session.history


def summary_for(results: list[dict]) -> dict:
    captures = [row for row in results if row["captured"]]
    return {
        "episodes": len(results),
        "captures": len(captures),
        "capture_rate_pct": round(100.0 * len(captures) / len(results), 2),
        "mean_turns_to_capture": round(mean(row["turns"] for row in captures), 2) if captures else None,
        "mean_movement_to_capture": round(mean(row["distance"] for row in captures), 2) if captures else None,
        "mean_localization_error": round(mean(row["mean_localization_error"] for row in results), 3),
        "mean_true_cell_probability": round(mean(row["mean_true_cell_probability"] for row in results), 4),
        "mean_runtime_ms": round(mean(row["runtime_ms"] for row in results), 3),
    }


def _make_charts(summary: dict, traces: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#24364c"
    teal = "#087e83"
    blue = "#476d94"
    modes = ("ai", "baseline")
    labels = ("HMM-guided", "Sensor-only")
    colors = (teal, blue)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 14, "axes.labelcolor": ink,
                         "text.color": ink, "axes.edgecolor": "#d3dbe0",
                         "xtick.color": ink, "ytick.color": ink,
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")
    values = [summary[m]["capture_rate_pct"] for m in modes]
    bars = ax.bar(labels, values, color=colors, width=0.52)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=6, fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Successful captures (%)")
    ax.set_title("Capture rate  |  paired seeded episodes", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.14)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "01_capture_rate.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")
    turn_values = [summary[m]["mean_turns_to_capture"] for m in modes]
    bars = ax.bar(labels, [v if v is not None else 0 for v in turn_values],
                  width=0.52, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}" if v is not None else "No captures" for v in turn_values],
                 padding=6, fontsize=12)
    ax.set_ylim(0, max([v or 0 for v in turn_values] + [1]) * 1.23)
    ax.set_ylabel("Turns (successful episodes only)")
    ax.set_title("Mean turns until capture", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.14)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "02_mean_turns.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2), layout="constrained")
    for mode, label, color in zip(modes, labels, colors):
        point_means = [(turn, float(np.mean(errors))) for turn, errors in
                       sorted(traces[mode].items()) if errors]
        ax.plot([turn for turn, _ in point_means], [error for _, error in point_means],
                label=label, color=color, linewidth=2.4)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean Manhattan localization error (cells)")
    ax.set_title("Localization error over time", loc="left", pad=16)
    ax.text(0.0, -0.21, "Mean across episodes still active at each turn; later turns have fewer samples.",
            transform=ax.transAxes, fontsize=9, color="#607081")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.14)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "03_localization_error.png", dpi=175,
                bbox_inches="tight")
    plt.close(fig)


def run_evaluation(*, episodes: int = 100, seed: int = 42,
                   grid_size: int = 10, max_turns: int = 80,
                   sensor_accuracy: float = 0.60,
                   output_dir: str | Path | None = None) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor = NoisySensor(sensor_accuracy, grid_size)
    hunter_start = (grid_size // 2, grid_size // 2)
    results: dict[str, list[dict]] = {"ai": [], "baseline": []}
    traces: dict[str, dict[int, list[int]]] = {"ai": {}, "baseline": {}}
    for episode_id in range(episodes):
        scenario = EpisodeScenario.generate(
            master_seed=seed, episode_id=episode_id, grid_size=grid_size,
            max_turns=max_turns, hunter_start=hunter_start, sensor=sensor)
        for mode in ("ai", "baseline"):
            result, history = play_episode(
                mode, scenario, master_seed=seed, grid_size=grid_size,
                max_turns=max_turns, sensor_accuracy=sensor_accuracy)
            results[mode].append(result)
            for record in history:
                traces[mode].setdefault(record.turn, []).append(record.localization_error)

    summary = {mode: summary_for(rows) for mode, rows in results.items()}
    payload = {"configuration": {"episodes_per_strategy": episodes,
                                 "master_seed": seed, "grid_size": grid_size,
                                 "max_turns": max_turns,
                                 "sensor_accuracy": sensor_accuracy,
                                 "ghost_trajectories_paired": True,
                                 "noise_offsets_paired": True},
               "ai": summary["ai"], "baseline": summary["baseline"]}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "episode_id", "captured",
                                               "turns", "distance", "capture_attempts",
                                               "mean_localization_error",
                                               "mean_true_cell_probability", "runtime_ms"])
        writer.writeheader()
        for mode in ("ai", "baseline"):
            writer.writerows(results[mode])
    with (output_dir / "localization_by_turn.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "turn", "mean_error", "active_episodes"])
        for mode in ("ai", "baseline"):
            for turn, errors in sorted(traces[mode].items()):
                writer.writerow([mode, turn, round(float(np.mean(errors)), 5), len(errors)])
    _make_charts(summary, traces, output_dir)
    return payload
