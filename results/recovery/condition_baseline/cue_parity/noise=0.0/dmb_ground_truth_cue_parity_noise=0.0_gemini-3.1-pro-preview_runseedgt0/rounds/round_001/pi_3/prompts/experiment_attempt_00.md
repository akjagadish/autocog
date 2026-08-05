# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended heuristics (e.g. TTB, EQW, Tallying, WADD) make distinguishable predictions: avoid degenerate pairs where every heuristic agrees, and prefer pairs that dissociate single-feature focus from feature-summing strategies. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Lexicographic Parity (Tightened)

**Parameters:**
- validities: validities
- k_frac: [0.5, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Lexicographic Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = len(validities)
    
    # Determine the number of top cues to consider based on k_frac
    k_frac = float(parameters["k_frac"])
    k = int(np.round(k_frac * n_features))
    if k < 1:
        k = 1
        
    # Get indices of the top k validities (stable sort to preserve original order on ties)
    top_k_idx = np.argsort(validities, kind='mergesort')[-k:]
    
    a_top = a[top_k_idx]
    b_top = b[top_k_idx]
    
    # Count strict A wins within the top K valid cues
    a_wins = int(np.sum(a_top > b_top))
    
    # Odd number of A-wins -> A (0); even -> B (1)
    winner = 0 if (a_wins % 2 == 1) else 1
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Cue-Parity from Tallying, we manipulate the number of cues on which option A strictly beats option B. Tallying predicts a monotonic preference for A as the number of A-wins increases relative to B-wins. Cue-Parity, in contrast, predicts a non-monotone, oscillating preference based solely on whether the number of A-wins is odd or even. For example, when A dominates B on 2 or 4 features, Tallying strongly predicts A, but Cue-Parity predicts B (because 2 and 4 are even). Conversely, when A wins on 1 feature and B wins on 2 or 3, Tallying predicts B, but Cue-Parity predicts A (because 1 is odd). We include these crucial dissociations as well as baseline trials where both models agree.

[1] To strongly dissociate Tallying from Cue-Parity, this design focuses on trials where option A and option B win on an EQUAL number of features (i.e., a_wins == b_wins). Tallying predicts complete indifference (50/50 guessing) whenever the tally is tied, because it merely compares the counts. Cue-Parity, however, makes completely deterministic predictions even when the counts are tied, because it only checks whether a_wins is odd or even. For example, if both options win on exactly 1 feature, Tallying predicts a tie, whereas Cue-Parity deterministically predicts A (since 1 is odd). If both options win on exactly 2 features, Tallying predicts a tie, whereas Cue-Parity deterministically predicts B (since 2 is even). We include these tied-tally trials alongside asymmetric trials (a_wins != b_wins) to ensure a robust quantitative dissociation across the full parameter space.

[2] To dissociate Cue-Parity (which computes parity over ALL features) from Lexicographic Parity (which computes parity only over the top K most valid features), we design trials where the parity of A-wins changes depending on whether the least valid features are included. By making A strictly beat B on specific lower-validity cues, the two models will predict opposite choices for most values of the k_frac parameter.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate Lexicographic Parity from Cue-Parity, we manipulate whether the parity of A-wins changes when only the most valid cues are considered versus when all cues are considered. Lexicographic Parity restricts its parity calculation to the top K valid features (where K depends on k_frac), whereas Cue-Parity always computes parity across all features. By designing trials where option A strictly beats B on an odd number of the top features but an even number of features overall (and vice versa), the two models yield strictly opposite deterministic predictions. We vary the distribution of A-wins across the high-validity and low-validity features to ensure robust dissociation regardless of the exact k_frac value.",
  "validities": [
    0.98,
    0.92,
    0.85,
    0.7,
    0.6,
    0.52
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2935,
  "candidates_token_count": 552,
  "total_token_count": 4999
}
```
