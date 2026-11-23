"""Generate 100 paired HMM-vs-baseline episodes and three evaluation charts."""

import argparse
import json

from ghost_hunter.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost Hunter: seeded paired evaluation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--accuracy", type=float, default=0.60)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    try:
        results = run_evaluation(episodes=args.episodes, seed=args.seed,
                                 grid_size=args.grid_size, max_turns=args.max_turns,
                                 sensor_accuracy=args.accuracy, output_dir=args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2))
    print(f"\nCharts, raw episode CSV, and summary saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
