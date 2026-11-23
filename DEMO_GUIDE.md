# Demo guide and suggested screenshots

This is an easy demonstration sequence for presenting the university project. Every screenshot is from the **real running game**; use **P** to save it automatically to `results/screenshots/`.

| Step | What to do | What to explain / capture |
| --- | --- | --- |
| 1 | Run `python main.py` | Main window with hidden ghost, sensor reading, and probability grid |
| 2 | Press **H** off, then on | Compare an unassisted grid with the HMM heatmap |
| 3 | Press **W/A/S/D** a few times | Each move consumes one turn, the ghost moves, and a new noisy reading arrives |
| 4 | Press **C** on a suspected cell | Explain how failed capture gives definite negative evidence |
| 5 | Press **2**, then **A** | Watch HMM AI autoplay and the probability heatmap update |
| 6 | Press **3**, then **A** | Show current-reading-only baseline with the same seeded environment |
| 7 | Press **TAB** | Show collected human scores next to automated evaluation results |
| 8 | Run `python run_evaluation.py` | Show the three generated charts and CSV results |
| 9 | Run `python -m unittest discover -s tests -v` | Show passing correctness and reproducibility checks |

## One-minute viva explanation

"The ghost is never directly visible. We only receive its noisy distance from the hunter. Every cell is a possible hidden state. Our HMM predicts the ghost's next position using the transition matrix, multiplies those probabilities by how likely the new sensor reading is at each cell, and normalizes everything. The heatmap shows this posterior. The AI decides where to go using those beliefs. We also built a baseline that throws away previous measurements, so the experiment can check whether memory and probabilistic inference actually help. We run 100 matched episodes with seed 42 and save our results and charts."

## Viva questions to practice

**Why HMM?** The true ghost position is hidden, but observations and position transitions have known probabilities.

**Why is the environment partially observable?** The hunter knows its own location but not the exact ghost location.

**What is the forward algorithm?** A recursive predict-and-update computation of the probability of each hidden state given observations so far.

**What if all posterior probabilities become zero?** The implementation has a defensive normalization fallback; under the matching simulation model this should not happen for generated observations.

**What is fair about the experiment?** Same ghost movement, same sensor error offsets, same control rules and planner; only the belief source changes.

**Why can a missed capture be useful?** It rules out the current cell at that instant, and it can hold the hunter in a promising interception cell as the ghost moves.

**Why no training dataset?** The model uses known simulated transition and observation probabilities rather than learned neural-network weights.
