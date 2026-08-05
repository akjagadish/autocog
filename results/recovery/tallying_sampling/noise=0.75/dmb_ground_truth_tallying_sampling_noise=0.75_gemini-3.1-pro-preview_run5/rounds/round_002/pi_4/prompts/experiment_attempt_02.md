# experiment_attempt_02

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
**Description:** Mixture of Tallying and Take-The-Best (TTB): Decision makers are heterogeneous in their strategy use. While the majority of choices are made using a compensatory equal-weight heuristic (Tallying), a smaller proportion of decisions rely on a non-compensatory, one-reason heuristic (Take-The-Best), which evaluates cues sequentially by validity and stops at the first discriminating cue. This mixture model captures both the dominant compensatory behavior and the minority non-compensatory behavior, providing a better fit to aggregate human data than either heuristic alone.

**Parameters:**
- w_ttb: [0.0, 0.25]
- beta_tally: [0.0, 2.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    # Strategy 1: Tallying (Equal-Weight)
    scores_tally = np.sum(stim, axis=1)
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy 2: Take-The-Best (TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # Mixture
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To strongly dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 5-feature design where the highest-validity cue frequently conflicts with a coalition of lower-validity cues. TTB, being a non-compensatory one-reason decision heuristic, will invariably choose the option favored by the single most valid discriminating cue. In contrast, WADD integrates all features weighted by their validities; thus, when multiple lower-validity cues align against the highest-validity cue, their combined weight can surpass it, leading WADD to choose the opposite option. This set of 8 mirrored trials systematically pits the top cue against various combinations of the remaining cues to guarantee opposing predictions across the two models.

[1] To quantitatively dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, we use a 5-feature design with a steep but compensatory validity structure. The validities are chosen such that the highest-validity cue alone is less predictive than the sum of several lower-validity cues. By systematically pitting the most valid discriminating cue against a coalition of less valid cues, TTB (which stops at the first discriminating cue) will consistently select the option favored by the top cue, whereas WADD will integrate all available information and select the option with the greater overall weighted evidence. The trials include cases where the top cue is the first overall, as well as cases where the top cue ties and the discrimination falls to the second or third cue, ensuring the dissociation is robust across the cue hierarchy.

[2] This design dissociates the Tallying (Equal-Weight) heuristic from the Weighted Additive (WADD) model. Tallying simply counts the number of positive features and chooses the option with the higher count, completely ignoring cue validities. WADD, on the other hand, computes a weighted sum of the features based on their validities. By using a 6-feature design with a steep drop-off in validities between the top three and bottom three cues, we can pit quantity against quality. In several critical trials, Option A possesses a greater number of positive features (all from the low-validity tier), making it the clear choice for Tallying. However, Option B possesses fewer positive features but from the high-validity tier, such that their weighted sum exceeds that of Option A, making Option B the choice for WADD. Control trials where both models agree are included to balance the design and allow for accurate parameter estimation.

[3] This design quantitatively dissociates the advocated Weighted Additive (WADD) model from the competing Tallying (Equal-Weight) heuristic. Tallying assumes subjects simply count the number of positive features and choose the option with the higher tally, ignoring the differential validities of the features. WADD, in contrast, computes a weighted sum of the features based on their validities. We use a 5-feature design with a steep drop-off in validities between the top two cues and the bottom three cues. In critical trials, one option possesses fewer positive features but from the high-validity tier (favored by WADD), while the other option possesses a greater number of positive features but from the low-validity tier (favored by Tallying). The design also includes trials where both options have an equal number of positive features, but one is favored by WADD due to higher validities, causing Tallying to predict a tie (guessing) while WADD predicts a deterministic choice.

[4] To uniquely dissociate pure Tallying from a Mixture of Tallying and Take-The-Best (TTB), we use a 6-feature design that systematically pits the raw count of positive features against the highest-validity cue. Pure Tallying evaluates options solely based on the difference in the total number of positive cues, predicting a 50/50 split when tallies are tied, and a strong preference when tallies differ, completely ignoring validities. The Mixture model, however, posits that a fraction of choices are driven by TTB, which strictly follows the highest-validity discriminating cue. By including 'tally-tied' trials where the top cue favors one option, and 'tally-advantaged' trials where the top cue opposes the tally, we can detect if the highest-validity cue exerts an independent pull. If the pure Tallying model is correct, choice probabilities will map perfectly onto tally differences, showing no sensitivity to which specific cue is the highest validity.

[5] To quantitatively dissociate the pure Tallying (Equal-Weight) heuristic from a Mixture of Tallying and Take-The-Best (TTB), we use a 5-feature design that systematically manipulates the agreement between the raw count of positive features and the single highest-validity discriminating cue. Pure Tallying predicts choice probabilities entirely based on the difference in tallies, completely ignoring cue validities; it strictly predicts a 50/50 split when tallies are tied. The Mixture model, however, incorporates a proportion of choices driven by TTB, which follows the highest-validity cue. By including 'tally-tied' trials where the top cue favors one option, and 'tally-opposed' trials where the option with fewer total positive features possesses the highest-validity cue, we can isolate the independent pull of TTB. If the pure Tallying model is correct, choice probabilities will map perfectly onto tally differences and show no preference in tied-tally trials regardless of which specific cue is positive.

[6] To quantitatively dissociate pure Tallying from a Mixture of Tallying and Take-The-Best (TTB), we use a 5-feature design that systematically pits the raw tally of positive features against the highest-validity cue. Pure Tallying predicts choice probabilities entirely based on the difference in tallies, completely ignoring cue validities; notably, it predicts an exact 50/50 split when tallies are tied. The Mixture model incorporates a proportion of choices driven by TTB, which strictly follows the highest-validity discriminating cue. By including 'tally-tied' trials where the top discriminating cue favors one option, we can isolate the independent pull of TTB. If the pure Tallying model is correct, choice probabilities will map perfectly onto tally differences and show no preference in tied-tally trials regardless of which specific cue is positive. Conflicting trials where the tally favors one option but the highest validity cue favors the other further help to estimate the mixture weight.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates a pure Tallying (Equal-Weight) heuristic from a Mixture of Tallying and Take-The-Best (TTB). Pure Tallying predicts choice probabilities based exclusively on the difference in the raw count of positive features between the two options, predicting exactly 50% for tied tallies and monotonically increasing probabilities as the tally difference grows, regardless of which specific features are positive. The Mixture model, however, incorporates a proportion of choices driven by TTB, which strictly follows the highest-validity discriminating cue. To isolate this, the design uses a 6-feature setup with scrambled validities and includes 'tally-tied' trials where the highest-validity cue favors one option, predicting a departure from the 50/50 split expected by pure Tallying. It also includes 'tally-opposed' trials where the tally favors one option (e.g., 4 vs 2 or 3 vs 1) but the highest-validity cue favors the other, which will dampen or reverse the choice probability predicted by pure Tallying. By systematically varying the tally difference (0, 1, 2) against the direction of the highest-validity cue, we can precisely estimate the mixture proportion of TTB.",
  "validities": [
    0.65,
    0.95,
    0.55,
    0.85,
    0.5,
    0.75
  ],
  "trial_a_ratings": [
    [
      1,
      1,
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
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      1,
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
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
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
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
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
      1,
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
      0,
      1
    ],
    [
      0,
      1,
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
  "prompt_token_count": 3739,
  "candidates_token_count": 672,
  "total_token_count": 5773
}
```
