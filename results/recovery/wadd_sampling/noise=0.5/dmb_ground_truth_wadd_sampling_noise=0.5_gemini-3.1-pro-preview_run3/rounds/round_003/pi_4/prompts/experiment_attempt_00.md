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
**Description:** Linear Weighted Additive Theory: Decision makers evaluate options by summing the features of each option, weighted linearly by their predictive validity (specifically, the validity's edge over chance, validity - 0.5). This represents a moderate integration strategy that avoids the extreme dominance of top cues seen in log-odds (Bayesian) weighting, while still differentiating cue importance unlike Equal-Weight/Tallying.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Linear WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to linear weights (edge over chance)
    weights = val - 0.5
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Tallying (Equal Weights) Theory with Very Soft Softmax: Decision-makers evaluate options by assigning equal weight to all features, simply counting the number of positive features for each option. However, their choices are not highly deterministic; they employ a 'very soft' decision rule where the difference in tallies yields only a mild preference (e.g., ~55/45 splits) rather than an overwhelming one, reflecting high levels of noise, uncertainty, or guessing in human decision-making in this domain.

**Parameters:**
- beta: [0.01, 0.5]
- epsilon: [0.0, 0.1]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Equal Weights expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The experiment is designed to pit Take The Best (TTB) against Tallying. TTB decides based solely on the highest-validity discriminating cue, ignoring the total number of cues each option wins. Tallying ignores cue validities and simply counts the number of cues each option wins. By creating option pairs where Option A wins on the highest-validity discriminating cue but Option B wins on a strictly greater number of lower-validity cues, the two models make completely opposite predictions. The trials systematically vary which cue is the highest discriminating cue and the margin of the tallying win.

[1] This design systematically pits Tallying against Take The Best (TTB) by creating choice pairs where the option favored by the highest-validity discriminating cue is opposed by the option that wins on a greater number of less valid cues. By varying the position of the highest discriminating cue and the margin of the tally, we can isolate whether choices are driven by single-cue frugality (TTB) or unweighted cue summation (Tallying).

[2] This design pits Weighted Additive (WADD) Theory with Log-Odds against Tallying. WADD integrates features by weighting them with the log-odds of their validities, allowing a single highly valid cue to outweigh multiple less valid cues, or differentiating between options when the number of winning cues is tied. Tallying simply counts the number of features each option wins on, ignoring validities. By using validities [0.95, 0.75, 0.65, 0.55], we create strong disparities between log-odds weights. We include trials where one option wins on a single highly valid cue while the other wins on three less valid cues (WADD favors the former, Tallying favors the latter), as well as trials where both options win on an equal number of cues but with different validities (WADD strongly prefers one option, Tallying predicts a tie).

[3] To strongly dissociate Tallying (which simply counts the number of features each option wins on, ignoring validities) from Weighted Additive (WADD) with log-odds (which scales features non-linearly by their validities), we use a set of five features with a steep drop-off in validity (0.90, 0.80, 0.60, 0.55, 0.55). This ensures that the log-odds weight of the first feature is substantially larger than the sum of the bottom three or four features. In these trials, Option A often wins on a sheer majority of low-validity features (strongly favored by Tallying), while Option B wins on only one or two highly valid features (strongly favored by WADD). We also include trials where both options win on an equal number of features (Tallying predicts a tie), but the validities heavily favor one option (WADD strongly predicts that option).

[4] To dissociate WADD with Log-Odds from Linear WADD, we exploit the non-linear transformation of log-odds. A highly valid feature (e.g., 0.99) has a massive log-odds weight (approx 4.6), which easily dominates the sum of several moderate features (e.g., 0.85, 0.80, 0.75). In contrast, the Linear WADD theory weights features by their edge over chance (validity - 0.5). For 0.99, the linear weight is 0.49, which is easily outvoted by combinations like 0.85 + 0.80 (0.35 + 0.30 = 0.65). By pitting a single extremely valid feature against a combination of moderately valid features, we create a strong quantitative dissociation where Log-Odds strongly predicts Option A and Linear WADD predicts Option B.

[5] To dissociate Linear WADD from Log-Odds WADD, we exploit the difference between linear weights (validity - 0.5) and non-linear Bayesian weights (log-odds). Log-odds weighting assigns exponentially greater importance to highly valid cues, whereas linear weighting treats the differences more additively. By setting validities to [0.92, 0.82, 0.72, 0.62, 0.52], we create choice pairs where a single higher-validity cue is pitted against a combination of lower-validity cues. For example, comparing cue 1 (0.92) against cues 2 and 4 (0.82 and 0.62): Linear WADD assigns cue 1 a weight of 0.42, and cues 2+4 a combined weight of 0.32 + 0.12 = 0.44, thus favoring the combination. In contrast, Log-Odds WADD assigns cue 1 a weight of ~2.44, and cues 2+4 a combined weight of ~1.51 + 0.48 = 1.99, thus strongly favoring the single highly valid cue. These systematic reversals isolate the integration function.

[6] This design systematically dissociates the Tallying (Equal Weights) theory from the Linear Weighted Additive (Linear WADD) theory. Tallying simply counts the number of positive features, ignoring validities, and predicts choices based on the difference in tallies. Linear WADD weights features by their edge over chance (validity - 0.5). By using validities of [0.95, 0.85, 0.75, 0.65, 0.55], the linear weights are [0.45, 0.35, 0.25, 0.15, 0.05]. The trials are constructed to pit these two integration strategies against each other. For example, in some trials, one option wins on a sheer majority of features (strongly favored by Tallying), while the other option wins on fewer, but highly valid features, such that the linear sum of validities favors it (Linear WADD). Other trials include ties in Tallying where Linear WADD strongly predicts one option, or ties in Linear WADD where Tallying strongly predicts one option.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design aims to quantitatively dissociate Linear Weighted Additive (Linear WADD) from Tallying (Equal Weights) with a very soft softmax. Tallying simply counts the number of positive features and applies a low-beta softmax, resulting in choices that lean towards the option with more positive features but remain close to 50/50. Linear WADD, in contrast, applies a linear weight (validity - 0.5) to each feature and can have a much higher beta, leading to strong, deterministic predictions when the weighted sums differ. By using a specific validity spread [0.95, 0.85, 0.75, 0.65, 0.55], we create trials where Tallying predicts a clear advantage for one option (e.g., Option B has 3 features vs. Option A's 1 feature) while Linear WADD predicts a tie, or where Tallying predicts a tie (both have 2 features) but Linear WADD strongly predicts Option A due to higher validities. Additionally, we include reversal trials where Option B wins the tally but Option A wins the weighted linear sum.",
  "validities": [
    0.95,
    0.85,
    0.75,
    0.65,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
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
      1,
      0,
      0,
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
      1,
      0,
      1,
      0,
      0
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
      0,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
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
      0,
      1,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
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
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
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
  "prompt_token_count": 3424,
  "candidates_token_count": 662,
  "total_token_count": 5950
}
```
