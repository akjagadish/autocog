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


## THEORY 2
**Description:** Equal-Weight Additive Rule: People integrate all available cues but ignore their differential validities, assigning equal weight to each cue. They compute an overall score for each option by summing its positive features. This is a fully compensatory process that discards validity magnitudes but retains the quantity of evidence. The option with the higher total score is favored, with response noise introduced via a softmax function and a uniform lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal-Weight expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Sum all features (equal weighting)
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To best distinguish Take The Best (TTB) from Tallying, we need trials where the two heuristics make opposing choices. TTB relies solely on the first discriminating cue in order of descending validity, whereas Tallying counts the total number of winning features irrespective of their validities. We design a 5-feature experiment with clearly spread validities. In the critical trials, one option wins on the highest-validity discriminating cue (so TTB chooses it), while the other option wins on a greater number of lower-validity cues (so Tallying chooses it). We also include trials where Tallying results in a tie (predicting a 50/50 guess) but TTB makes a deterministic prediction.

[1] To maximally distinguish Tallying from Take The Best (TTB), we use a 4-feature design where the highest validity cue strongly contradicts the sum of the remaining cues. In trials 1-4, one option wins on the single most valid discriminating cue (so TTB chooses it), while the other option wins on a greater number of lower-validity cues (so Tallying chooses it). In trials 5-6, the options tie in the total number of winning cues (Tallying predicts a 50/50 guess), but they differ on the highest validity cue (TTB predicts a deterministic choice).

[2] To maximally distinguish Weighted Additive (WADD) from Tallying, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. We include 'conflict' trials where WADD prefers an option that wins on a few high-validity features, while Tallying prefers the alternative because it wins on a strictly greater number of lower-validity features. We also include 'tie' trials where both options win on an equal number of features (making Tallying guess uniformly), but the validities heavily favor one option under WADD. Finally, we include concordant trials with varying margins to help constrain the response noise and softmax temperature parameters for both models.

[3] To maximally distinguish Tallying from Weighted Additive (WADD) decision making, we use a 6-feature design with a steep drop-off in validities. Tallying completely ignores validities and simply counts the number of winning features, while WADD integrates both the number of winning features and their weights. We include 'conflict' trials where Tallying prefers one option because it wins on more features, but WADD prefers the other because it wins on fewer, much higher-validity features. We also include 'WADD tie' trials where the weighted sum exactly equals the sum of multiple lower-validity features, causing WADD to guess while Tallying strongly prefers the option with more winning features. Finally, 'Tallying tie' trials are included where the number of winning features is equal, causing Tallying to guess, but WADD strongly prefers the option with higher-validity features.

[4] Equal-Weight Additive (Theory 1) and Tallying (Theory 2) are mathematically equivalent when applied to binary features. Both models compute unweighted sums of feature values (or wins) and pass the difference through a softmax. Since the difference in total positive features between two options is exactly equal to the difference in the number of winning features, the models yield identical choice probabilities for any binary stimulus pair. Therefore, this design uses a diverse set of feature vectors to confirm this equivalence empirically across varying degrees of feature overlap and score differences.

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
  "rationale": "Theory 1 (Tallying) and Theory 2 (Equal-Weight Additive) are mathematically identical for binary features. Tallying computes the number of strict feature-wise wins for each option, while Equal-Weight computes the total number of positive features. Because any feature where both options have a 1 contributes equally to the Equal-Weight scores of both options (and is ignored by Tallying), the difference between the scores of the two options is exactly the same under both models. Since the softmax choice rule relies solely on the score differences, both theories predict identical choice probabilities for any pair of binary vectors. Therefore, this design includes a diverse set of feature combinations with varying degrees of overlap and score differences to empirically confirm this equivalence.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0
    ],
    [
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      1,
      1
    ],
    [
      0,
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
  "prompt_token_count": 2985,
  "candidates_token_count": 444,
  "total_token_count": 4419
}
```
