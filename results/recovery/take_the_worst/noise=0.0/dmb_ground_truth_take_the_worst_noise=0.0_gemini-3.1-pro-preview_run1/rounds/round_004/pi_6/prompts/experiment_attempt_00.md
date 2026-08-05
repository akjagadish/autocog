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
**Description:** Weighted Tallying: People compare options by computing a weighted sum of feature wins, where the weights are a heavily compressed function of the stated cue validities (w_i = validity_i ** gamma, with gamma close to 0). This allows the strategy to behave almost exactly like simple Tallying when one option has a clear numerical advantage in feature wins, but smoothly and compensatorily resolves ties using all available validity information rather than defaulting to a strict lexicographic tie-breaker.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 0.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Compress validities to create weights that are close to 1 (Tallying-like) 
    # but still retain ordinal validity information for compensatory tie-breaking.
    weights = val ** gamma
    
    # Calculate weighted feature wins (ignoring ties on individual features)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
[0] This design systematically pits Take The Best (TTB) against Tallying by varying which option wins on the highest-validity discriminating cue versus which option wins on the sheer number of cues. By assigning strictly descending validities to the 4 features, TTB will always select the option that wins on the earliest discriminating feature. In contrast, Tallying simply counts the number of winning features, completely ignoring their validities. The selected trials include strong dissociations where TTB chooses one option (because it wins on the single most valid cue) while Tallying confidently chooses the other (because it wins on a larger number of lower-validity cues), as well as cases where TTB strongly prefers one option but Tallying predicts a tie.

[1] This design systematically pits Tallying against Take The Best (TTB) across 5 features with descending validities. Tallying ignores validities and simply counts the number of features favoring each option, whereas TTB relies exclusively on the highest-validity discriminating feature. The trial pairs are constructed to create sharp dissociations: in many trials, one option wins on the single most valid cue (thus chosen by TTB) while the other option wins on a sheer majority of lower-validity cues (thus chosen by Tallying). We also include trials where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction based on the most valid cue.

[2] This design systematically pits the Weighted Additive (WADD) theory against the Tallying heuristic. WADD integrates all features by weighting them directly by their validities, allowing strong features to outweigh a larger number of weak features. Tallying, on the other hand, ignores validities and simply counts the number of features favoring each option. By setting validities to a spread of [1.0, 0.9, 0.6, 0.5, 0.5], we create trials where WADD strongly prefers an option with fewer but highly valid features, while Tallying prefers the option with a greater number of less valid features. We also include trials where WADD predicts a tie (e.g., one 1.0 validity feature vs. two 0.5 validity features) but Tallying predicts a clear winner, as well as trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a clear winner due to the validity differences.

[3] This design systematically pits Tallying against the Weighted Additive (WADD) theory. Tallying simply counts the number of features favoring each option, ignoring validities, while WADD computes a weighted sum using the validities as weights. By setting validities to [1.0, 0.9, 0.5, 0.5, 0.5], we create strong dissociations. For example, in some trials, one option wins on the two most valid cues (WADD favors it) while the other option wins on the three least valid cues (Tallying favors it). We also include trials where WADD predicts a tie (e.g., one 1.0 validity feature vs. two 0.5 validity features) but Tallying predicts a clear winner, as well as trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a clear winner due to the validity differences.

[4] To dissociate Top-K Tallying from standard Tallying, we use an environment with 10 features where the advocated theory assumes subjects ignore the least valid cues (since k is parameterized between 4 and 8), while standard Tallying counts all 10 features equally. By creating trials where the top 8 features are either tied in tally or favor Option A, but the bottom 2 features strongly favor Option B, standard Tallying will consistently choose Option B (or predict a tie), whereas Top-K Tallying will either predict a tie or strongly favor Option A, completely ignoring the bottom 2 cues.

[5] To dissociate Tallying from Top-K Tallying (where K is bounded between 4 and 8), we use an environment with 10 features of strictly descending validities. Standard Tallying counts all 10 features equally. By contrast, Top-K Tallying completely ignores at least the bottom 2 features (and up to the bottom 6). We construct trials where Option A wins exclusively on the lowest-validity features (e.g., features 8, 9, 10), while Option B wins on a smaller or equal number of the highest-validity features. In these cases, standard Tallying will favor Option A (or predict a tie), whereas Top-K Tallying will ignore Option A's wins entirely and strongly favor Option B (or predict a tie).

[6] To dissociate Tallying with Lexicographic Tie-Breaking from standard Tallying, we construct critical trials where the two options are tied in their feature tallies. Standard Tallying predicts a 50/50 guess in these cases, regardless of the validities. By contrast, Tallying with Lexicographic Tie-Breaking predicts that subjects will use the highest-validity discriminating feature to break the tie. The design includes trials with exact tally ties where the most valid cue favors Option A or Option B, allowing a clear quantitative dissociation between the two theories. We also include filler trials where one option wins the tally to ensure subjects remain engaged with the primary strategy.

[7] To cleanly dissociate standard Tallying from Tallying with Lexicographic Tie-Breaking, the design focuses heavily on trials where the two options have an equal number of feature wins (a tally tie). According to the advocated theory (standard Tallying), subjects will guess uniformly at random on these trials because the tally is tied. According to the competing theory, subjects will break the tie using the single most valid discriminating feature. By varying which option is favored by the highest-validity cue during tally ties, we can observe whether choices remain near 50% or shift strongly toward the tie-breaker cue. Filler trials where one option wins the tally are included to prevent subjects from noticing that most trials are ties.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate Weighted Tallying from standard Tallying, we exploit trials where the two options have an equal number of feature wins (a tally tie). Standard Tallying completely ignores validities and predicts a 50/50 guess whenever the tallies are tied. Weighted Tallying, however, applies a heavy compression to the validities (gamma close to 0) which preserves ordinal validity information, allowing it to smoothly and compensatorily break ties. We include trials where the highest-validity cue is outweighed by multiple slightly-lower validity cues (compensatory tie-breaking), which separates Weighted Tallying from simple Lexicographic Tie-Breaking, as well as standard tie trials and unambiguous filler trials where both theories agree.",
  "validities": [
    1.0,
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0
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
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0
    ],
    [
      0,
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
  "prompt_token_count": 3651,
  "candidates_token_count": 500,
  "total_token_count": 6090
}
```
