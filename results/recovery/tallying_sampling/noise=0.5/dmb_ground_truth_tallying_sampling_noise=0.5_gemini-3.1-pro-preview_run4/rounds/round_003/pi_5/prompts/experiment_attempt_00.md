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
**Description:** Non-linear Validity Scaling Theory: Decision-makers use a single compensatory strategy to integrate cues, but they subjectively transform the provided cue validities by raising them to a fractional exponent (gamma). This non-linear scaling flattens the differences between cue validities. As gamma approaches 0, the validities become equal, naturally mimicking the Tallying (equal-weight) heuristic. As gamma approaches 1, the model recovers pure Weighted Additive (WADD) behavior. This provides a mathematically elegant, single-process account of the empirical pull towards equal weighting without assuming a discrete mixture of distinct decision strategies.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform cue validities via fractional exponent
    subjective_validities = validities ** gamma
    
    # Calculate compensatory scores
    scores = stim @ subjective_validities
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory: Decision-makers are heterogeneous in their strategies. A proportion of choices are made using a compensatory Weighted Additive (WADD) strategy that integrates cue validities and subjective weights, while the remainder rely on a simpler Tallying (Equal Weight) heuristic that merely counts the number of positive features. Behavior on any given trial is a probabilistic mixture of these two strategies, with the mixture proportion varying across individuals. This naturally explains why aggregate behavior falls between the pure predictions of WADD and Tallying.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_mix * p_wadd + (1.0 - w_mix) * p_tally
    
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
[0] To dissociate Take The Best (TTB) from Weighted Additive (WADD), we must contrast a non-compensatory decision rule with a compensatory one. TTB stops at the first discriminating cue (the one with the highest validity), ignoring all others. WADD integrates all cues, weighting them by their validity. We use 5 features with a spread of validities. In the critical trials, one option is favored by the single most valid discriminating cue, while the other option is favored by a coalition of several lower-validity cues. TTB will reliably choose the former, whereas WADD will choose the latter because the sum of the lower-validity weights exceeds the highest validity weight.

[1] To quantitatively dissociate WADD (which integrates all features according to their validities) from TTB (which stops at the first discriminating feature), we use a 5-feature design with linearly spaced validities. The trials systematically pit the single best discriminating cue (favoring Option A) against a coalition of lower-validity cues (favoring Option B). We vary the size and specific composition of the opposing coalition. This not only separates WADD from TTB on overall choice proportions but also allows us to measure the compensatory threshold: how many lower-validity cues are required for WADD to flip its preference, whereas TTB remains rigidly anchored to the highest-validity cue across all such trials.

[2] The design contrasts Tallying (Equal Weight) with WADD by setting up a stark opposition between feature count and feature validity. By using a skewed validity distribution (two highly valid cues and three less valid cues), we can create 'dissociation trials' where one option has more positive features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). Including trials where the two models agree ensures a robust behavioral baseline.

[3] To quantitatively dissociate the Weighted Additive (WADD) theory from the Tallying (Equal Weight) theory, we manipulate the trade-off between feature count and feature validity. Tallying predicts choices strictly based on the unweighted sum of positive features. WADD predicts choices based on the validity-weighted sum. By utilizing a skewed validity distribution (two highly predictive cues and three less predictive cues), we create strong dissociation trials where one option has fewer positive features but higher overall validity (favored by WADD), while the other has more positive features but lower overall validity (favored by Tallying). We also include trials where Tallying predicts a tie but WADD has a strict preference, and baseline trials where both models agree, ensuring a robust test of whether participants integrate cue validities or simply count features.

[4] To quantitatively dissociate Strategy Mixture Theory (WADD + Tallying) from pure WADD, we manipulate the alignment between the validity-weighted sum of features (WADD) and the unweighted count of positive features (Tallying). In 'conflict' trials, one option has fewer but highly valid features, while the other has more but less valid features, causing WADD and Tallying to prefer opposite options. In 'partial agreement' trials, WADD has a clear preference but Tallying sees a tie (equal feature counts). Pure WADD predicts choices driven strictly by the weighted sums. Strategy Mixture Theory, however, predicts that the Tallying component will pull choice probabilities toward the Tallying preference (or toward 0.5 in the case of a Tallying tie), systematically dampening the WADD-predicted choice shares on these specific trials compared to trials where both heuristics agree.

[5] To quantitatively dissociate pure WADD from Strategy Mixture Theory (WADD + Tallying), this design creates 'WADD-tie' trials where the validity-weighted sums of the two options are exactly equal, but the unweighted feature counts (Tallying) strongly favor one option. For instance, with validities [1.0, 0.5, 0.5, 0.5, 0.5], one highly valid feature (1.0) equals the sum of two less valid features (0.5 + 0.5). Pure WADD predicts indifference (50% choice probability) on these trials, as the weighted sums are identical. In contrast, Strategy Mixture Theory predicts a systematic deviation from 50% toward the option with more positive features, because the Tallying component of the mixture will break the tie. By observing choice behavior on these specific trials, we can directly measure the presence of a Tallying component without confounding it with general noise or WADD's compensatory trade-offs.

[6] This design quantitatively dissociates Strategy Mixture Theory from Non-linear Validity Scaling Theory by exploiting how each model combines compensatory integration with equal-weighting (Tallying). Non-linear Scaling assumes a single integrated score using transformed validities ($v^\gamma$), predicting smooth, continuous shifts in choice probability as features are added or removed. In contrast, Strategy Mixture Theory mixes the *probabilities* of two distinct strategies. By keeping the Weighted Additive (WADD) advantage strongly in favor of Option A (using highly valid cues) while systematically varying the Tallying advantage (from strongly favoring B to strongly favoring A) using low-validity cues, the Strategy Mixture Theory predicts distinct, step-like shifts in choice probabilities. Because the Tallying component's probability saturates quickly based on feature counts, mixing it with WADD produces a specific linear interpolation of probabilities that cannot be captured by a single non-linearly scaled score.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Non-linear Validity Scaling Theory from Strategy Mixture Theory by exploiting the mathematical properties of concave scaling. Strategy Mixture Theory assumes decision-makers linearly sum cue validities (WADD) and mix this probabilistically with a Tallying (equal-weight) heuristic. In contrast, Non-linear Scaling Theory assumes validities are subjectively transformed via a fractional exponent (gamma < 1) before integration. Because a fractional exponent is a concave function, the difference between transformed low validities is larger than the difference between transformed high validities (e.g., 0.6^0.5 - 0.5^0.5 > 1.0^0.5 - 0.9^0.5). We create 'Perfect Tie' trials where Option A and Option B have identical linear WADD sums and identical Tally scores (e.g., A has validities 1.0 and 0.5; B has validities 0.9 and 0.6). Strategy Mixture predicts a perfect 50/50 tie on these trials. However, Non-linear Scaling rigidly predicts a preference for Option B, because the concave transformation boosts the 'middle' validities more than the 'extreme' validities. We also include 'Conflict' trials (where WADD favors A but Tally favors B) to ensure mixture parameters and beta can be properly identified.",
  "validities": [
    1.0,
    0.9,
    0.6,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      1,
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
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1
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
      1,
      0,
      1,
      0
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
      0
    ],
    [
      0,
      1,
      1,
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
      0,
      1,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3408,
  "candidates_token_count": 636,
  "total_token_count": 10209
}
```
