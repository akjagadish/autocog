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
**Description:** Faced with abstract options and no trial-by-trial feedback, subjects abandon feature-based compensatory or non-compensatory strategies. Instead, they fall back on a strong, idiosyncratic side bias (e.g., a spatial preference for the left or right option, or a sequential preference for the first or second option). Each subject has a fixed probability of choosing Option A, which varies uniformly across the population from 0 to 1. This mechanism produces behavior that is completely independent of the feature validities or values, resulting in an aggregate alignment of ~50% with any feature-based heuristic (like Take The Best or Tallying), but driven by individual-level spatial/sequential biases rather than trial-level random guessing.

**Parameters:**
- p_a: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The subject ignores the stimulus features and relies entirely on their idiosyncratic side bias.
    p_a = float(parameters["p_a"])
    
    # Return the fixed choice probabilities for Option A and Option B
    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. While they may occasionally attempt to tally features or look at the most valid cue, the lack of grounding leads to a near-uniform choice probability across all stimulus pairs.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Calculate a weak tallying signal
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Massive lapse rate dominates the choice
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design systematically pits Take The Best (TTB) against Tallying by presenting pairs where the highest-validity discriminating cue favors one option, but the sheer number of winning cues favors the other. For example, in Trial 1, Option A wins only on the most valid cue, whereas Option B wins on three lower-validity cues. TTB predicts a choice for A because it stops at the first discriminating cue, completely ignoring the subsequent cues. Tallying predicts a choice for B because it counts the total number of winning cues (3 to 1) regardless of their validities. By varying which cue is the highest discriminating one and balancing the positions, we can quantitatively dissociate the one-reason stopping rule of TTB from the compensatory feature-summing of Tallying.

[1] This design perfectly dissociates Tallying from Take The Best (TTB) across 8 distinct trial pairs. In every pair, the highest-validity discriminating cue favors one option (which TTB will strictly choose), while the sheer number of winning cues favors the other option (which Tallying will strictly choose). By varying the index of the first discriminating cue and the margin of the tally, we ensure robust quantitative separation between the one-reason stopping rule and the compensatory counting rule.

[2] This design systematically varies the degree of conflict between Take The Best (TTB) and Tallying to dissociate a pure Tallying strategy from a mixture of TTB and Tallying. By including trials where TTB and Tallying strongly disagree (Tallying difference of 3), weakly disagree (Tallying difference of 1), agree, or where Tallying is tied but TTB is decisive, we can observe whether choices reflect a pure counting rule (which would track the tally difference monotonically and predict 50/50 on tally ties) or a mixture (which would show a persistent bias toward the TTB winner even when the tally strongly opposes it, and break tally ties decisively in the direction of TTB).

[3] This design isolates the predictions of pure Tallying from a mixture of Tallying and Take The Best (TTB) by focusing on Tallying ties and strict Tallying margins. On trials where Tallying results in a tie (equal number of winning cues for A and B), the pure Tallying model predicts exactly 50/50 guessing regardless of the inverse temperature parameter. In contrast, the Mixture model predicts a systematic deviation from 50/50 in the direction of the highest-validity discriminating cue (TTB's choice). Furthermore, by including trials where Tallying favors one option by a narrow margin but TTB favors the other, we can cleanly separate the sensitivity to tally-differences from the persistent bias of the TTB component.

[4] This design systematically varies the tally difference between Option A and Option B across a wide range (from a 6-0 sweep to a 4-2 narrow win, and their inverses) to sharply distinguish the Advocated Theory (high noise/random guessing) from the Competing Theory (Tallying). Tallying, constrained by its parameter bounds (epsilon <= 0.5, beta >= 0.1), must predict a monotonically increasing preference for the option with more winning features, with a substantial choice probability for a 6-0 sweep. In contrast, the Advocated Theory predicts that due to massive lapse rates (epsilon >= 0.8), choice probabilities will remain virtually flat and near 50/50 across all trials, completely insensitive to the overwhelming feature evidence. A flat empirical choice curve would thus be uniquely predicted by the Advocated Theory.

[5] This design systematically varies the tally difference between Option A and Option B (from a 6-0 sweep down to a 3-3 tie, and the reverse) to quantitatively dissociate the Advocated Theory (Tallying) from the Competing Theory (Random Guessing). The Advocated Theory, characterized by low lapse rates and moderate-to-high beta, predicts a strong, monotonically increasing preference for the option with more winning features, reaching near-deterministic choices for large tally differences. In contrast, the Competing Theory, dominated by massive lapse rates (epsilon >= 0.8), predicts that choice probabilities will remain virtually flat and near 50/50 across all trials, entirely insensitive to overwhelming feature evidence. Observing a steep empirical choice curve would thus uniquely support the Advocated Theory.

[6] To dissociate the Advocated Theory (trial-by-trial random guessing due to massive lapse rates) from the Competing Theory (fixed idiosyncratic side bias), we present a diverse set of trials ranging from overwhelming evidence for one option to highly conflicting cue profiles. Both theories predict that subjects will ignore the feature evidence. However, they differ drastically in their predictions for within-subject behavior: the Advocated Theory predicts that every subject's choice proportion for Option A will tightly cluster around 0.5 (reflecting trial-by-trial high-noise guessing), whereas the Competing Theory predicts a wide, uniform distribution of individual choice proportions across the population (reflecting strong, fixed individual side biases, where some subjects always choose A and others always choose B).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "The Advocated Theory posits a fixed idiosyncratic side bias, predicting that a subject's probability of choosing Option A remains exactly constant across all trials, completely ignoring feature evidence. The Competing Theory posits high response noise but retains a weak sensitivity to the tally of winning features (due to epsilon < 1.0 and beta > 0). By presenting pairs with extreme tally differences (e.g., 5-0 vs 0-5), the Competing Theory predicts a measurable shift in the choice probability for A across these conditions (e.g., higher when A has 5 winning features than when B has 5). In contrast, the Advocated Theory predicts zero shift in choice probability across these conditions for any given subject, maintaining a flat line across all tally differences.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2956,
  "candidates_token_count": 437,
  "total_token_count": 4702
}
```
