# experiment_attempt_00

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** Tallying with Right-to-Left Position-Based Tie-Breaking

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    # Position-based lexicographic tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in range(len(a) - 1, -1, -1):
            if a[i] > b[i]:
                scores[0] += 1.0
                break
            elif b[i] > a[i]:
                scores[1] += 1.0
                break
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(p), p=p))
```


## THEORY 2
**Description:** Tallying with Recency-Driven Tie-Breaking: Decision makers primarily evaluate options using a Tallying heuristic, counting the number of features where one option strictly dominates the other. When the tallies are tied, they do not guess randomly. Instead, they rely on the cues most recently encoded into working memory to break the tie. Assuming a standard left-to-right reading order, the right-most features are the most recently processed. The model resolves ties by scanning features from right to left, favoring the option that wins on the first differentiating feature encountered.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary strategy: Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Recency-driven tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in reversed(range(len(a))):
            if a[i] > b[i]:
                score_a += 1.0
                break
            elif b[i] > a[i]:
                score_b += 1.0
                break
                
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with epsilon-greedy lapse
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To maximally distinguish Take The Best (TTB) from Tallying, we need trials where the highest-validity discriminating cue points to one option, while the sheer number of winning cues points to the other. By assigning a clear descending order of validities across 4 features, we can construct trials where option A wins on the single most valid cue, but option B wins on several lower-validity cues. TTB will choose A, whereas Tallying will choose B. We also include trials where Tallying predicts a tie (equal number of winning cues for both options) but TTB strongly prefers one option based on the highest-validity cue.

[1] To maximally dissociate Tallying from Take The Best (TTB), we use a 5-feature design with a randomized validity order to prevent simple left-to-right reading strategies. We construct 'compensatory' trials where one option wins on the single most valid discriminating cue (triggering a TTB choice) while the other option wins on a strictly greater number of lower-validity cues (triggering a Tallying choice). We also include 'tie' trials where both options win on an equal number of cues (causing Tallying to guess uniformly) but one option's wins are on higher-validity cues (causing TTB to strongly prefer it).

[2] To maximally distinguish WADD from Tallying, we use a 5-feature design with a wide spread of validities. Tallying relies exclusively on the count of winning features, ignoring their validities, whereas WADD integrates all features weighted by their subjective validity. The design includes 'Tallying-tie' trials where both options win on an equal number of features but WADD strongly prefers one option due to higher validities. It also includes 'compensatory' trials where Tallying prefers the option with more winning features, but WADD favors the option with fewer but higher-validity features. This structural dissociation allows us to identify whether subjects are weighting by validity or simply counting feature wins.

[3] To maximally distinguish Tallying from the Weighted Additive Model (WADD), we use a 6-feature design with a wide range of validities to create structural dissociations. Tallying relies solely on the count of winning features, completely ignoring their validities, whereas WADD integrates all features weighted by their subjective validity. The design includes 'Tallying-tie' trials where both options win on an equal number of features but WADD strongly prefers one option due to higher validities. It also includes 'compensatory' trials where Tallying prefers the option with more winning features, but WADD favors the option with fewer but higher-validity features. This allows us to cleanly identify whether subjects are weighting by validity or simply counting feature wins.

[4] To maximally dissociate Take The Best (TTB) from Tallying, this experiment uses a 5-feature design with randomized validity ordering to prevent simple left-to-right reading. The trials are constructed to create direct conflicts between the two heuristics. In 'compensatory' trials, one option wins on the single highest-validity discriminating feature (triggering a TTB choice) while the other option wins on a greater number of lower-validity features (triggering a Tallying choice). Additionally, the design includes 'tie' trials where both options win on an equal number of features (causing Tallying to guess uniformly) but one option's wins are on higher-validity features (causing TTB to strongly prefer it).

[5] To effectively dissociate Tallying from Take The Best (TTB), we utilize a 6-feature design with randomized validity order to prevent simple left-to-right reading. TTB makes decisions based solely on the highest-validity discriminating feature, while Tallying counts the total number of features favoring each option, ignoring validities. We design 'compensatory' trials where one option excels on the most valid feature (triggering TTB) but the other option wins on a greater number of less valid features (triggering Tallying). We also include 'tie' trials where both options win on an equal number of features (causing Tallying to guess uniformly) but one option's wins include the highest-validity discriminating feature (causing TTB to strongly prefer it).

[6] To maximally distinguish Theory 1 (Tallying with Right-to-Left Position-Based Tie-Breaking) from Theory 2 (Standard Tallying), we must focus on trials where both options have an equal number of feature wins (a tie in Tallying). Theory 2 predicts a 50/50 uniform random choice on these trials. In contrast, Theory 1 resolves the tie by scanning the features from right to left (from the last index to the first) and picking the option that wins on the first differentiating feature it encounters. The design includes several 'tie' trials where the right-most differentiating feature systematically favors either Option A or Option B. We also include 'non-tie' trials where both theories agree on the prediction, serving as a baseline for general Tallying behavior.

[7] To maximally distinguish Standard Tallying (Theory 1) from Tallying with Right-to-Left Tie-Breaking (Theory 2), the experiment focuses on 'tie' trials where both options win on an equal number of features. In these trials, Theory 1 predicts a 50/50 uniform random choice, whereas Theory 2 breaks the tie by scanning features from right to left and choosing the option that wins on the first differentiating feature it encounters. The design includes various tie configurations where the right-most differentiating feature systematically favors either Option A or Option B, as well as non-tie baseline trials where both theories agree.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "Both theories implement exactly the same logic (Tallying with right-to-left tie-breaking), despite using slightly different Python syntax (`range(len(a) - 1, -1, -1)` vs `reversed(range(len(a)))`). Since the predictions are mathematically identical under all conditions, no experiment can quantitatively dissociate them. This design provides a standard set of tallying 'tie' and 'non-tie' trials to verify the shared right-to-left tie-breaking behavior.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
  ],
  "trial_a_ratings": [
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
      0,
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
      0,
      1,
      1,
      1
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
      1,
      1,
      0,
      0
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
      1,
      0,
      1,
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
      1,
      0,
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
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3377,
  "candidates_token_count": 384,
  "total_token_count": 4678
}
```
