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
**Description:** When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

**Parameters:**
- bias: [0.4, 0.6]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** First-Feature Heuristic (Left-to-Right Reading Bias) with High Lapse: Subjects minimize cognitive effort by inspecting only the first (leftmost) feature in the array. However, because this task is demanding and feedback is absent, subjects rely heavily on random guessing on the vast majority of trials, only occasionally applying the first-feature rule. This results in behavior that is near-random but retains a faint trace of the leftmost feature's influence.

**Parameters:**
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"First-Feature Heuristic expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a_first = stim[0, 0]
    b_first = stim[1, 0]
    
    epsilon = float(parameters["epsilon"])
    
    if a_first > b_first:
        p_core = np.array([1.0, 0.0])
    elif b_first > a_first:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    # Blend deterministic first-feature rule with a random lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we must construct choice pairs where the single most valid discriminating cue points to one option, but the sum of the remaining less-valid cues points to the other option. By using a set of validities [0.9, 0.8, 0.7, 0.6], TTB will always choose the option that has a 1 on the highest-validity cue that differs between the two options. WADD, however, integrates all features, allowing multiple lower-validity cues to overrule a single high-validity cue. The trial pairs mix cases where TTB and WADD strongly disagree, cases where they agree, and cases where WADD predicts a tie but TTB strongly prefers one option.

[1] To cleanly separate WADD from TTB, we use a 5-feature design with a linear descent in validities. The core logic of the dissociation is compensatory vs. non-compensatory choice: TTB stops at the single most valid discriminating cue, while WADD integrates all cues. We construct pairs where the option favored by the highest-validity discriminating cue is overwhelmingly penalized on all lower-validity cues. In these trials, TTB reliably picks the option with the top cue, whereas WADD's weighted sum strongly favors the alternative. We also include trials where the top cue and the sum of lower cues agree, to serve as a baseline for overall accuracy and noise estimation.

[2] To quantitatively dissociate Tallying (Equal-Weight) from WADD, we use a 5-feature design with an extreme bimodal distribution of validities: two very high validities and three very low validities. Tallying simply counts the number of positive features, completely ignoring their validities, whereas WADD computes a weighted sum. We construct critical trials where Tallying prefers the option with more positive features (e.g., 3 low-validity cues), but WADD strongly prefers the alternative (e.g., 2 high-validity cues). We also include tie-breaking trials where Tallying sees an equal number of positive features (and thus guesses randomly), while WADD has a strong preference based on cue validity. Baseline trials where both models agree are included to estimate overall response noise.

[3] This design quantitatively dissociates the Weighted Additive (WADD) rule from the Tallying (Equal-Weight) heuristic. By using five features with a steep drop-off in validities (two high, three low), we create critical trials where a smaller number of highly valid features strongly outweighs a larger number of low-validity features. In these trials, WADD predicts preference for the option with fewer but better features, while Tallying counts purely positive cues and prefers the alternative. The design includes trials with direct opposition, trials where one model predicts a tie while the other shows a strong preference, and baseline trials where both agree, enabling a precise estimation of noise and strategy fit.

[4] To dissociate Random Guessing from the Weighted Additive (WADD) rule, we must vary the difference in weighted feature sums across trials. Random Guessing predicts a flat choice probability (modulo an idiosyncratic side bias) regardless of the stimulus. In contrast, WADD predicts that choice probabilities will track the difference in weighted sums, approaching deterministic choice for large differences and dropping to 50% only when the weighted sums are equal. The design includes trials where Option A strongly dominates, strongly loses, wins or loses by a small margin, and ties with Option B. This range of WADD score differences will yield an S-shaped response curve under WADD, but a flat horizontal line under Random Guessing.

[5] To quantitatively dissociate the Weighted Additive (WADD) rule from Random Guessing, we vary the difference in weighted feature sums systematically across trials. Under Random Guessing, choice probabilities remain flat (ignoring the stimulus entirely, modulo a side bias). Under WADD, the probability of choosing Option A should track the difference in weighted feature sums, forming an S-shaped psychometric curve. We use a 4-feature design and construct pairs that span a wide range of WADD score differences: from Option A strongly dominating, to moderate advantages, slight advantages, exact ties, and corresponding advantages for Option B. This gradient of score differences will yield a monotonic response curve under WADD but a flat horizontal line under Random Guessing.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate the Random Guessing baseline from the First-Feature Heuristic (with high lapse), we must manipulate the value of the first (leftmost) feature while varying the remaining features. Random Guessing predicts a constant choice probability for Option A across all trials, driven solely by an idiosyncratic side bias, regardless of any feature values. The First-Feature Heuristic predicts that choice probability for Option A will systematically increase when Option A has a 1 and Option B has a 0 on the first feature, decrease when the reverse is true, and drop exactly to 50% when the first features are tied. By including trials where the first feature strongly favors A, strongly favors B, or is tied, we can test whether the choice probabilities show this step-like function or remain perfectly flat.",
  "validities": [
    0.9,
    0.7,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      0,
      1,
      1
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      1,
      1
    ],
    [
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1
    ],
    [
      0,
      0,
      1
    ],
    [
      1,
      0,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 2739,
  "candidates_token_count": 356,
  "total_token_count": 3870
}
```
