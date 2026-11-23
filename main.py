"""Run Ghost Hunter: python main.py [--mode human|ai|baseline]."""

import argparse

from ghost_hunter.ui import GameUI


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghost Hunter interactive Pygame demo")
    parser.add_argument("--mode", choices=("human", "ai", "baseline"), default="human")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--accuracy", type=float, default=0.60, help="Probability of drawing a zero noise offset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        ui = GameUI(mode=args.mode, grid_size=args.grid_size, max_turns=args.max_turns,
                    sensor_accuracy=args.accuracy, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    ui.run()


if __name__ == "__main__":
    main()
