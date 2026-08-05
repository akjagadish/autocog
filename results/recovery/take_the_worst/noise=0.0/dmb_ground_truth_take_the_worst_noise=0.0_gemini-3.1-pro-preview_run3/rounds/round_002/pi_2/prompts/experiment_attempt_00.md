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


## COMPETING THEORY
**Description:** Strategy Selection (Mixture of Deterministic Tallying and Probabilistic WADD): Decision-makers probabilistically select between a frugal, unweighted strategy (Tallying) and a fully compensatory, validity-weighted strategy (WADD). Critically, Tallying operates as a deterministic rule (choosing the option with more winning cues, or guessing on ties) rather than a probabilistic score-based process. This breaks the assumption that Tallying consistency scales with the absolute difference in cue counts, allowing the model to capture high consistency in scenarios with small cue count differences (e.g., Exp 4) and lower consistency in scenarios with large cue count differences (e.g., Exp 2). The WADD strategy remains probabilistic and tempers the extremeness of the Tallying predictions.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_tally: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # 1. Deterministic Tallying strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # 2. WADD strategy scores
    val = np.asarray(parameters["validities"], dtype=float)
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores_wadd = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_tally = float(parameters["w_tally"])
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of the two strategies
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design leverages the core difference between Take The Best (TTB) and Tallying: TTB relies exclusively on the single most valid discriminating cue, ignoring all other cues, while Tallying counts the number of winning cues across all features, ignoring their validities. By using five features with strictly descending validities, we can construct trials where the option that wins on the highest-validity cue (favored by TTB) loses on the majority of the lower-validity cues (favored by Tallying). We also include trials where Tallying predicts a tie (guessing) while TTB has a strict preference, providing a clear quantitative dissociation.

[1] To strongly dissociate Tallying from Take The Best (TTB), we use five features with strictly descending validities. TTB decides solely based on the highest-validity cue that discriminates between the two options. Tallying, however, counts the total number of winning cues for each option, completely ignoring their validities. The design includes 'opposition' trials where the option with the highest valid discriminating cue (favored by TTB) is outnumbered by the total number of lower-validity cues favoring the other option (favored by Tallying). We also include 'tie' trials for Tallying where both options win on an equal number of cues (leading to guessing under Tallying) while TTB makes a deterministic prediction based on the single most valid cue. This mixture isolates the feature-summing behavior of Tallying from the one-reason stopping rule of TTB.

[2] To quantitatively dissociate Weighted Additive (WADD) from Tallying, this design uses five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. Option bias in WADD will further shift preferences asymmetrically, a pattern Tallying cannot capture.

[3] To quantitatively dissociate Tallying from Weighted Additive (WADD) with Option Bias, we use five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. This mixture isolates the feature-summing behavior of Tallying from the validity-weighted integration of WADD.

[4] This design tests a stark quantitative dissociation between the Advocated Theory (Mixture of Deterministic Tallying + Probabilistic WADD) and the Competing Theory (Probabilistic Tallying). By setting validities to [1.0, 0.5, 0.5, 0.6, 0.6], we create paired trials where the WADD score difference between options is held exactly constant, but the Tallying cue-count difference varies (e.g., a difference of 1 vs. 3). Under the Competing Theory, choice probability scales continuously with the cue-count difference, predicting a much stronger preference in the 3-difference trial than the 1-difference trial. Under the Advocated Theory, Tallying is deterministic (outputting 1.0 for the winner as long as the difference is > 0), and the WADD component is identical across both trials; thus, it predicts the exact same choice probability for both trials. We also include Tally-tie trials where WADD breaks the tie, which Probabilistic Tallying cannot capture.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the Advocated Theory (Probabilistic Tallying) from the Competing Theory (Mixture of Deterministic Tallying + Probabilistic WADD), this design holds the Tallying score difference constant across multiple trials while varying the WADD score difference. Under the Advocated Theory, choice probabilities are entirely determined by the difference in the number of winning cues (the Tally difference). Thus, any two trials where Option A wins on 2 cues and Option B wins on 1 cue will yield the exact same choice probability. Under the Competing Theory, Tallying acts as a deterministic baseline, but the final choice probability is heavily modulated by the WADD score difference. By constructing pairs that all have a Tally difference of +1 for Option A but widely varying WADD differences (e.g., +1.0 vs +0.2), the Advocated Theory predicts a flat consistency curve across these pairs, whereas the Competing Theory predicts varying choice probabilities. We also include 'Tally tie' trials where the Advocated Theory predicts exactly 50% guessing, but the Competing Theory predicts a preference driven by WADD.",
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
      1,
      0,
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
      0,
      1,
      1,
      0,
      0
    ],
    [
      0,
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
      0
    ],
    [
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
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
      0,
      0,
      0,
      1,
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
      0,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3485,
  "candidates_token_count": 514,
  "total_token_count": 6776
}
```
