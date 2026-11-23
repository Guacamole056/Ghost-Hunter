"""Reproducible paired experiments, sensitivity sweeps, and required figures."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import mean, median, stdev

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
    result["runtime_ms_per_turn"] = round(elapsed / max(1, result["turns"]), 4)
    return result, session.history


def summary_for(results: list[dict]) -> dict:
    captures = [row for row in results if row["captured"]]
    capture_turns = [row["turns"] for row in captures]
    total_turns = sum(row["turns"] for row in results)
    total_runtime_ms = sum(row["runtime_ms"] for row in results)
    return {
        "episodes": len(results),
        "captures": len(captures),
        "capture_rate_pct": round(100.0 * len(captures) / len(results), 2),
        "mean_turns_to_capture": round(mean(capture_turns), 2) if capture_turns else None,
        "median_turns_to_capture": round(float(median(capture_turns)), 2) if capture_turns else None,
        "std_turns_to_capture": (
            round(stdev(capture_turns), 2) if len(capture_turns) >= 2
            else (0.0 if len(capture_turns) == 1 else None)
        ),
        "mean_turns_all_episodes": round(mean(row["turns"] for row in results), 2),
        "mean_movement_to_capture": round(mean(row["distance"] for row in captures), 2) if captures else None,
        "mean_localization_error": round(mean(row["mean_localization_error"] for row in results), 3),
        "mean_true_cell_probability": round(mean(row["mean_true_cell_probability"] for row in results), 4),
        "mean_entropy": round(mean(row["mean_entropy"] for row in results), 3),
        "mean_runtime_ms": round(mean(row["runtime_ms"] for row in results), 3),
        "mean_runtime_ms_per_turn": round(total_runtime_ms / max(1, total_turns), 4),
    }


def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = "#24364c"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 14, "axes.labelcolor": ink,
                         "text.color": ink, "axes.edgecolor": "#d3dbe0",
                         "xtick.color": ink, "ytick.color": ink,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def _make_charts(summary: dict, traces: dict, output_dir: Path, episodes: int) -> None:
    plt = _configure_matplotlib()

    teal = "#087e83"
    blue = "#476d94"
    modes = ("ai", "baseline")
    labels = ("HMM-guided", "Sensor-only")
    colors = (teal, blue)

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

    min_solid = max(2, min(10, episodes // 5)) if episodes >= 10 else 1
    fig, ax = plt.subplots(figsize=(9.2, 5.2), layout="constrained")
    for mode, label, color in zip(modes, labels, colors):
        points = [(turn, float(np.mean(errors)), len(errors)) for turn, errors in
                  sorted(traces[mode].items()) if errors]
        solid = [(t, e) for t, e, n in points if n >= min_solid]
        tail = [(t, e) for t, e, n in points if n < min_solid]
        if solid:
            ax.plot([t for t, _ in solid], [e for _, e in solid],
                    label=label, color=color, linewidth=2.4)
            if tail:
                bridge = [solid[-1]] + tail
                ax.plot([t for t, _ in bridge], [e for _, e in bridge],
                        color=color, linewidth=1.6, linestyle="--", alpha=0.55)
        elif tail:
            ax.plot([t for t, _ in tail], [e for _, e in tail],
                    label=label, color=color, linewidth=2.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean Manhattan localization error (cells)")
    ax.set_title("Localization error over time", loc="left", pad=16)
    note = (
        f"Mean across active episodes per turn; solid: ≥{min_solid} active episodes, "
        f"dashed: <{min_solid} (small-sample tail)."
        if min_solid > 1
        else "Mean across episodes still active at each turn; later turns have fewer samples."
    )
    ax.text(0.0, -0.21, note, transform=ax.transAxes, fontsize=9, color="#607081")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.14)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "03_localization_error.png", dpi=175,
                bbox_inches="tight")
    plt.close(fig)


def _make_sensitivity_chart(sweep: dict, output_dir: Path) -> None:
    plt = _configure_matplotlib()
    teal = "#087e83"
    blue = "#476d94"

    acc_rows = sweep["accuracy_sweep"]
    accs = [row["sensor_accuracy"] * 100 for row in acc_rows]
    ai_rates = [row["ai"]["capture_rate_pct"] for row in acc_rows]
    base_rates = [row["baseline"]["capture_rate_pct"] for row in acc_rows]
    ai_turns = [row["ai"]["mean_turns_to_capture"] or 0 for row in acc_rows]
    base_turns = [row["baseline"]["mean_turns_to_capture"] or 0 for row in acc_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), layout="constrained")
    ax1.plot(accs, ai_rates, marker="o", color=teal, linewidth=2.4, label="HMM-guided")
    ax1.plot(accs, base_rates, marker="s", color=blue, linewidth=2.4, label="Sensor-only")
    ax1.set_ylim(0, 108)
    ax1.set_xlabel("Sensor exact-reading accuracy (%)")
    ax1.set_ylabel("Capture rate (%)")
    ax1.set_title("Capture rate vs. sensor accuracy", loc="left", pad=12)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(alpha=0.14)
    ax1.legend(frameon=False)

    ax2.plot(accs, ai_turns, marker="o", color=teal, linewidth=2.4, label="HMM-guided")
    ax2.plot(accs, base_turns, marker="s", color=blue, linewidth=2.4, label="Sensor-only")
    ax2.set_xlabel("Sensor exact-reading accuracy (%)")
    ax2.set_ylabel("Mean turns to capture (successes)")
    ax2.set_title("Capture speed vs. sensor accuracy", loc="left", pad=12)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.14)
    ax2.legend(frameon=False)

    fig.savefig(output_dir / "04_sensitivity_accuracy.png", dpi=175)
    plt.close(fig)


def _run_paired_batch(*, episodes: int, seed: int, grid_size: int, max_turns: int,
                      sensor_accuracy: float, ghost_bias: float = 0.0) -> tuple[dict, dict, dict, dict]:
    sensor = NoisySensor(sensor_accuracy, grid_size)
    hunter_start = (grid_size // 2, grid_size // 2)
    results: dict[str, list[dict]] = {"ai": [], "baseline": []}
    traces: dict[str, dict[int, list[int]]] = {"ai": {}, "baseline": {}}
    entropy_traces: dict[str, dict[int, list[float]]] = {"ai": {}, "baseline": {}}
    for episode_id in range(episodes):
        scenario = EpisodeScenario.generate(
            master_seed=seed, episode_id=episode_id, grid_size=grid_size,
            max_turns=max_turns, hunter_start=hunter_start, sensor=sensor,
            ghost_bias=ghost_bias)
        for mode in ("ai", "baseline"):
            result, history = play_episode(
                mode, scenario, master_seed=seed, grid_size=grid_size,
                max_turns=max_turns, sensor_accuracy=sensor_accuracy)
            results[mode].append(result)
            for record in history:
                traces[mode].setdefault(record.turn, []).append(record.localization_error)
                entropy_traces[mode].setdefault(record.turn, []).append(record.entropy)
    summary = {mode: summary_for(rows) for mode, rows in results.items()}
    return summary, results, traces, entropy_traces


def run_evaluation(*, episodes: int = 100, seed: int = 42,
                   grid_size: int = 10, max_turns: int = 80,
                   sensor_accuracy: float = 0.60,
                   output_dir: str | Path | None = None) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, results, traces, entropy_traces = _run_paired_batch(
        episodes=episodes, seed=seed, grid_size=grid_size,
        max_turns=max_turns, sensor_accuracy=sensor_accuracy)

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
                                               "mean_true_cell_probability",
                                               "mean_entropy",
                                               "runtime_ms", "runtime_ms_per_turn"])
        writer.writeheader()
        for mode in ("ai", "baseline"):
            writer.writerows(results[mode])
    with (output_dir / "localization_by_turn.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "turn", "mean_error", "active_episodes", "mean_entropy"])
        for mode in ("ai", "baseline"):
            for turn, errors in sorted(traces[mode].items()):
                ent = entropy_traces[mode][turn]
                writer.writerow([mode, turn, round(float(np.mean(errors)), 5),
                                 len(errors), round(float(np.mean(ent)), 5)])
    _make_charts(summary, traces, output_dir, episodes)
    return payload


def run_sensitivity_sweep(*, episodes: int = 50, seed: int = 42,
                          grid_size: int = 10, max_turns: int = 80,
                          accuracies: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.00),
                          ghost_biases: tuple[float, ...] = (0.0, 0.35, 0.70),
                          output_dir: str | Path | None = None) -> dict:
    """Evaluate robustness across sensor noise levels and mismatched ghost dynamics."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_sweep = []
    for acc in accuracies:
        summary, _, _, _ = _run_paired_batch(
            episodes=episodes, seed=seed, grid_size=grid_size,
            max_turns=max_turns, sensor_accuracy=acc, ghost_bias=0.0)
        accuracy_sweep.append({"sensor_accuracy": acc, "ai": summary["ai"],
                               "baseline": summary["baseline"]})

    mismatch_sweep = []
    for bias in ghost_biases:
        summary, _, _, _ = _run_paired_batch(
            episodes=episodes, seed=seed, grid_size=grid_size,
            max_turns=max_turns, sensor_accuracy=0.60, ghost_bias=bias)
        mismatch_sweep.append({"ghost_bias": bias, "ai": summary["ai"],
                               "baseline": summary["baseline"]})

    payload = {
        "configuration": {"episodes_per_condition": episodes, "master_seed": seed,
                          "grid_size": grid_size, "max_turns": max_turns},
        "accuracy_sweep": accuracy_sweep,
        "model_mismatch_sweep": mismatch_sweep,
    }
    (output_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _make_sensitivity_chart(payload, output_dir)
    return payload
