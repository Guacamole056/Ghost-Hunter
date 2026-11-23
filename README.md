# Ghost Hunter: Tracking an Invisible Enemy

**FAST-NUCES · AL-2002 Artificial Intelligence Lab · Fall 2026**

A playable 10 × 10 Pygame game about finding an invisible moving ghost. You never see the ghost's real position during play. You receive a **noisy distance reading** and can inspect a **probability heatmap** from a Hidden Markov Model (HMM). You can hunt manually, watch an AI hunter, or watch a sensor-only baseline. The comparison is backed by 100 paired, seeded simulations and three charts.

The project implements the provided proposal, **Idea No. 10 · Part A — Game**. It runs locally and needs **no dataset, account, API key, pretrained model, or internet connection after installing dependencies**.

## 1. Run it (step-by-step)

You need **Python 3.10 or newer**. Check it by opening Command Prompt / Terminal and running `python --version` (use `py` on Windows if `python` isn't recognized).

### Windows (Command Prompt or VS Code terminal)

```bat
cd path\to\ghost_hunter_project
py -m pip install -r requirements.txt
py main.py
```

If `py` is not recognized, replace it with `python` in both commands.

### macOS / Linux

```bash
cd /path/to/ghost_hunter_project
python3 -m pip install -r requirements.txt
python3 main.py
```

For the automatic AI demo directly:

```bash
python main.py --mode ai
```

For the baseline demo:

```bash
python main.py --mode baseline
```

### Controls

| Key or control | Action |
| --- | --- |
| **1 / 2 / 3** | Switch between Human / HMM AI / Sensor-only baseline |
| **W A S D** or **arrow keys** | Move one cell in human mode |
| **C** | Attempt capture at your **current** cell |
| **SPACE** | Advance one AI/baseline turn |
| **A** | Start/pause AI/baseline autoplay |
| **[ / ]** | Slow down / speed up autoplay |
| **H** | Toggle belief heatmap |
| **TAB** | Show human vs AI scoreboard |
| **R** | Restart the same seeded episode |
| **N** | Start a new seed / different episode |
| **P** | Save a screenshot in `results/screenshots/` |
| **ESC** | Exit |

All important mode controls (including sensor accuracy presets, human capture, and AI autoplay speed) also have clickable buttons.

> **Game rule:** Moving onto the ghost does **not** automatically capture it. To win, attempt **C** on the cell where the ghost currently is. A failed attempt uses one turn, provides definite evidence about that cell, and then the ghost moves. This is deliberately consistent for both humans and AI.

### What you'll see

- **Board:** the cyan ring represents the hunter; the ghost remains invisible until the episode ends.
- **Sensor reading:** a Manhattan distance to the hidden ghost, with noise from −2 to +2 cells.
- **Heatmap:** each cell color represents the hunter's current probability estimate. Brighter means more likely.
- **Best location:** highest-probability cell and its confidence percentage.
- **Scores:** turns, distance traveled, attempts, successful captures; press **TAB** to compare stored human results with AI evaluation results.

## 2. Reproduce the evaluation

```bash
python run_evaluation.py
```

The default is **100 paired episodes per automated strategy**, **master seed 42**, **10 × 10 grid**, **80-turn limit** and **60% exact sensor readings**.

Files are saved in `results/`:

| File | What it contains |
| --- | --- |
| `summary.json` | Configuration and aggregate results for both strategies |
| `episodes.csv` | All 200 episode results and timing data |
| `localization_by_turn.csv` | Mean localization error and active-episode count for each turn |
| `01_capture_rate.png` | Capture success comparison |
| `02_mean_turns.png` | Mean turns among successful captures |
| `03_localization_error.png` | Mean grid localization error across turns |
| `human_scores.json` | Auto-created after you finish human episodes |

Custom comparison and optional sensitivity/model-mismatch sweep:

```bash
python run_evaluation.py --episodes 100 --seed 42 --grid-size 10 --max-turns 80 --accuracy 0.6
python run_evaluation.py --sweep
```

**Paired fairness:** both hunters face the same generated ghost trajectory, the same noise **offset** at each turn, and the same movement/capture rules. Because their hunter positions differ, the numeric distance readings can differ, as they should. Both hunters use the **same action-selection policy**: the only difference is HMM accumulated belief versus belief reconstructed from the current observation alone.

### Included measured results (seed 42, 100 episodes/strategy)

| Metric | HMM hunter | Sensor-only |
| --- | ---: | ---: |
| Capture success | **100/100 (100%)** | **51/100 (51%)** |
| Mean turns to capture (successes) | **21.42** (median 18.0) | 34.63 (median 35.0) |
| Mean movement to capture (successes) | **17.68** | 27.78 |
| Mean localization error (cells) | **3.125** | 5.952 |
| Mean true-cell probability | **0.1761** | 0.0583 |
| Mean belief entropy (bits) | **3.496** | 5.058 |
| Mean runtime per turn (ms) | 0.0628 | **0.0591** |

These are measurements for this particular seeded simulation and policy, **not a guarantee** across all noise settings and seeds. Per-episode and per-turn runtimes are in `summary.json` and depend on the machine. Running with `--sweep` also evaluates sensor accuracy (`20%–100%`) and mismatched ghost dynamics (`sensitivity.json`, `04_sensitivity_accuracy.png`). Human scores are collected separately and are not presented as a paired controlled experiment.

## 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

20 tests check transitions, valid moves, vectorized distance/likelihood equivalence, observation likelihood including clipping, forward normalization, evidence from failed captures, deterministic seeds, grid-size-aware thresholds, biased-ghost scenarios, baseline memorylessness, capture mechanics, chart/sweep export generation, and UI modal/control behavior.

## 4. Project folder structure

```text
ghost_hunter_project/
├── main.py                  # Launch interactive Pygame window
├── run_evaluation.py        # 100 paired experiments and charts
├── requirements.txt
├── requirements-report.txt # Optional: regenerate report PDF
├── ghost_hunter/
│   ├── core.py              # GhostEnvironment, NoisySensor, HMMTracker, GameSession
│   ├── agents.py            # Shared planner, two inference strategies
│   ├── evaluation.py        # Results, metrics, chart generation
│   ├── ui.py                # Pygame game and scoreboard
│   └── __init__.py
├── scripts/
│   └── build_report_pdf.py  # Optional PDF report generator
├── tests/
│   ├── test_core.py
│   ├── test_evaluation.py
│   └── test_ui.py
├── docs/
│   ├── PROJECT_REPORT.md
│   ├── ALGORITHM_EXPLAINED.md
│   ├── DEMO_GUIDE.md
│   └── Ghost_Hunter_Project_Report.pdf
└── results/
    ├── summary.json
    ├── episodes.csv
    ├── localization_by_turn.csv
    └── 01_*.png, 02_*.png, 03_*.png
```

## 5. HMM in three lines

1. **Predict:** `belief_next = belief_current @ transition_matrix`
2. **Observe:** multiply every candidate cell by the probability of the new noisy distance reading.
3. **Normalize:** divide all weights by their sum so the grid probabilities add to 1.

On the **first** reading, use the uniform initial prior (excluding the hunter's starting cell), without an extra prediction. If a capture misses, exclude the hunter's cell from the current posterior before predicting the ghost's next move.

## 6. Implementation notes

- No PyTorch, TensorFlow, machine-learning dataset, or pretrained weights are used. HMM math is written with **NumPy**.
- The ghost takes exactly one cardinal step per turn, uniformly among valid neighboring cells; the HMM transition matrix uses the same model.
- The **baseline** ignores previous observations at every new turn; both agents may use a single movement forecast for action planning, but only the HMM retains belief across turns.
- An unsuccessful capture costs a turn even if intended as a stationary probe. Capture attempts are tracked separately from movement.
- Game and evaluation follow the exact same rules. Dynamic map clipping and probability normalization are tested.
- The interface uses Pygame's built-in fonts and shapes; no image downloads or local font files are needed.

## 7. Proposal coverage

The source contains all five in-scope features: playable hidden-ghost grid, configurable noisy sensor, HMM+heatmap, autonomous belief-guided hunter, and human vs AI score comparison. The baseline and automated evaluation add the planned scientific comparison. Online multiplayer, real-world sensors, 3D, neural networks, and story generation are intentionally not included.

For the science and demonstration narrative, read `docs/PROJECT_REPORT.md` and `docs/DEMO_GUIDE.md`.

To regenerate the optional printable PDF report after changing the evaluation results:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_report_pdf.py
```
