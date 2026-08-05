"""Tests for the seed-GT-control trajectory plot's data extraction.

The plot reads each run's `leaderboard.md`; the only logic worth testing is the
parse (markdown -> per-round scores) and the seed-vs-best-adversary reduction.
Both are checked against hand-computed values from a tiny fixed leaderboard.
"""
from scripts.plot_seed_gt_control import (
    parse_post_admit_scores,
    seed_and_best_adversary,
)

# Two post-admit rounds (plus a post-data block that must be IGNORED). Scores
# are chosen so the seed (pi_1) is NOT always the per-round max, exercising the
# best-adversary reduction.
SAMPLE = """## round 0 — post-data

```
[round 0 | post-data]
  # 1  pi_1  1.000  |####|
  # 2  pi_2  0.000  |....|
```

## round 0 — post-admit (pi_3)

```
[round 0 | post-admit (pi_3)]
  # 1  pi_1  0.980  |####|  Δ-0.020
  # 2  pi_3  0.917  |###.|  (new)
  # 3  pi_2  0.000  |....|  Δ ·
```

## round 1 — post-admit (pi_4)

```
[round 1 | post-admit (pi_4)]
  # 1  pi_4  0.972  |####|  (new)
  # 2  pi_1  0.959  |####|  Δ-0.026
  # 3  pi_3  0.398  |#...|  Δ-0.009
```
"""


def test_parse_post_admit_scores_ignores_post_data_and_keeps_order():
    rounds = parse_post_admit_scores(SAMPLE)
    assert rounds == [
        {"pi_1": 0.980, "pi_3": 0.917, "pi_2": 0.000},
        {"pi_4": 0.972, "pi_1": 0.959, "pi_3": 0.398},
    ]


def test_seed_and_best_adversary_excludes_the_seed():
    rounds = parse_post_admit_scores(SAMPLE)
    seed, best_adv = seed_and_best_adversary(rounds, seed_label="pi_1")
    # Seed trajectory is pi_1's own score each round.
    assert seed == [0.980, 0.959]
    # Best adversary = max score among NON-seed labels each round:
    #   round 0: max(pi_3=0.917, pi_2=0.000) = 0.917
    #   round 1: max(pi_4=0.972, pi_3=0.398) = 0.972
    assert best_adv == [0.917, 0.972]
