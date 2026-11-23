"""Generate 100 paired HMM-vs-baseline episodes, charts, and optional robustness sweep."""

import argparse
import json

from ghost_hunter.evaluation import run_evaluation, run_sensitivity_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost Hunter: seeded paired evaluation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--accuracy", type=float, default=0.60)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sweep", action="store_true",
                        help="Also run sensor-accuracy and model-mismatch robustness sweeps")
    parser.add_argument("--sweep-episodes", type=int, default=50,
                        help="Episodes per condition when --sweep is enabled")
    args = parser.parse_args()
    results = run_evaluation(episodes=args.episodes, seed=args.seed,
                             grid_size=args.grid_size, max_turns=args.max_turns,
                             sensor_accuracy=args.accuracy, output_dir=args.output_dir)
    print(json.dumps(results, indent=2))
    if args.sweep:
        sweep = run_sensitivity_sweep(
            episodes=args.sweep_episodes, seed=args.seed,
            grid_size=args.grid_size, max_turns=args.max_turns,
            output_dir=args.output_dir)
        print("\nSensitivity & model-mismatch sweep saved (sensitivity.json, 04_sensitivity_accuracy.png).")
        print(json.dumps({
            "accuracy_sweep": [
                {"accuracy": r["sensor_accuracy"],
                 "ai_capture_pct": r["ai"]["capture_rate_pct"],
                 "ai_turns": r["ai"]["mean_turns_to_capture"],
                 "baseline_capture_pct": r["baseline"]["capture_rate_pct"]}
                for r in sweep["accuracy_sweep"]
            ],
            "model_mismatch_sweep": [
                {"ghost_bias": r["ghost_bias"],
                 "ai_capture_pct": r["ai"]["capture_rate_pct"],
                 "ai_turns": r["ai"]["mean_turns_to_capture"],
                 "baseline_capture_pct": r["baseline"]["capture_rate_pct"]}
                for r in sweep["model_mismatch_sweep"]
            ],
        }, indent=2))
    print(f"\nCharts, raw episode CSV, and summary saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
