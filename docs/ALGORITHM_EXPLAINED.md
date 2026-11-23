# How the AI actually works

This document follows the class design and evaluation plan in the proposal. All coordinates are stored as `(row, column)` (zero-indexed in Python, one-indexed in the UI).

## 1. Hidden state and world model

A 10 × 10 grid has 100 hidden ghost states. The ghost starts uniformly at random anywhere except the hunter's known initial position. Every time a move or failed capture consumes a turn, the ghost moves to exactly one legal cardinal neighbor, chosen uniformly. Ghost movement does not depend on the hunter's actions, so a single pre-generated trajectory can be used fairly for multiple strategies.

Define `T[i,j] = P(ghost at cell j next | ghost at cell i now)`. Every valid neighbor gets probability `1 / number_of_neighbors(i)`. Every other transition receives probability 0. Boundary cells have fewer neighbors, and each transition row sums to exactly 1.

## 2. What the sensor observes

True distance is Manhattan distance: `d = abs(hunter_row - ghost_row) + abs(hunter_col - ghost_col)`.

The sensor adds one integer error:

| Error | Probability (accuracy 0.60) |
| ---: | ---: |
| −2 | 0.0667 |
| −1 | 0.1333 |
| 0 | 0.6000 |
| +1 | 0.1333 |
| +2 | 0.0667 |

Readings are clipped into `[0, 2 × (grid_size − 1)]`. Therefore the observation likelihood **sums probabilities for all noise offsets that clip to the same observed reading**; simply looking up the noise probability is incorrect near the distance boundaries.

`NoisySensor.read_distance()` generates observations. `NoisySensor.likelihood()` calculates `P(reading | candidate ghost cell)`.

## 3. HMM forward algorithm

Let `b_t(j)` be the probability the ghost occupies cell `j` after all readings up through turn `t`.

**Predict:**

`prior_{t+1}(j) = sum_i [ b_t(i) × T[i,j] ]`

**Update using observed noisy distance `z`:**

`weight(j) = P(z | ghost=j, hunter_position) × prior_{t+1}(j)`

**Normalize:**

`b_{t+1}(j) = weight(j) / sum_k weight(k)`

On turn 0, the tracker starts from a uniform prior excluding the hunter's starting cell and updates immediately with the first measurement (there is no ghost movement before that observation).

On a failed capture, we know the ghost is **not** at the hunter's current cell before it moves. The tracker first sets that cell's posterior to zero, renormalizes, then predicts the next ghost step and updates using the next observation.

A heatmap displays `b_t` without displaying the true ghost cell. Its brightest cell is the posterior mode (`argmax`). The displayed percentage is its belief probability.

## 4. How the hunter chooses actions

Both AI and baseline use the **same deterministic decision policy**:

1. If the current cell has belief probability at least `0.06`, attempt capture immediately.
2. Otherwise forecast the ghost's next-cell distribution with one step of the movement model.
3. Score each legal cardinal move **and** a stationary capture/probe using:

   `score(destination) = 3 × next_ghost_probability(destination) − 0.02 × expected_Manhattan_distance_from(destination)`

4. Pick the best-scoring action using deterministic tie-breakers. A low-probability stationary probe may miss, but it consumes one turn and gets the next sensor reading while leaving the hunter in a possible interception cell.

The forecast helps intercept the ghost rather than chase its last estimated location. A capture always uses a whole turn. Neither agent is allowed to read `ghost_position` as part of action selection.

**Main comparison variable:**

- **HMM hunter:** its probabilities are the recursive `HMMTracker` posterior, which accumulates information across turns.
- **Sensor-only baseline:** it rebuilds a flat-prior posterior using the *latest* noisy reading only. It never feeds a past observation or past posterior into the next estimate. Its planner can forecast one step from that current estimate, but cannot remember prior evidence.

## 5. Reproducible comparison

For episode ID `e`, `EpisodeScenario.generate()` uses `SeedSequence([42, e])` and splits random streams for movement and sensor noise. Both strategies receive the **same** ghost trajectory and **same** sequence of noise offsets. Numeric sensor readings can be different because the two strategies move to different positions. They share the same action rules, maximum turns, and movement model.

The comparison reports:

- Capture success rate
- Mean turns until capture (only successfully captured episodes)
- Mean movement distance until capture (only successfully captured episodes)
- Localization error in grid cells (Manhattan distance from most likely cell to true ghost cell)
- Runtime per episode (in milliseconds)

Localization curves average across episodes **still active** at each turn. Episode counts are included so that later portions, which contain fewer surviving episodes, are not mistaken for perfectly balanced samples. Human results are recorded separately from this paired experiment.

## 6. Important limits

The motion and sensor models are known and correctly specified in this simulation; robustness to incorrect models is not tested. There are no walls, ghost adversarial decisions, occlusions, or real-world sensors. The 100-episode findings are specific to the specified seed and settings. The greedy policy is an interpretable heuristic rather than a globally optimal partially observable Markov decision process solver.
