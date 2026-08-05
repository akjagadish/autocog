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
**Description:** Take-The-Best with Cue-Reading Errors: Decision-makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in strictly descending order of validity. However, they have a non-zero probability of misreading or misremembering individual cue values. This preserves the strict non-compensatory stopping rule while introducing variance that mimics compensatory behavior in certain environments.

**Parameters:**
- p_error: [0.0, 0.2]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_error = float(parameters["p_error"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by true validities descending
    order = np.argsort(validities)[::-1]
    
    n_samples = 200
    p_a = 0.0
    
    for _ in range(n_samples):
        # Independent probability of misreading each cue
        err_a = np.random.rand(len(a)) < p_error
        err_b = np.random.rand(len(b)) < p_error
        
        # Apply errors (flip 0 to 1, and 1 to 0)
        a_noisy = np.abs(a - err_a)
        b_noisy = np.abs(b - err_b)
        
        for idx in order:
            if a_noisy[idx] > b_noisy[idx]:
                p_a += 1.0
                break
            elif b_noisy[idx] > a_noisy[idx]:
                break
        else:
            p_a += 0.5
            
    p_a /= n_samples
    p_core = np.array([p_a, 1.0 - p_a])
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Take The Best (TTB) heuristic posits a lexicographic decision rule where individuals search through features in descending order of validity. They stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. If no feature discriminates, they guess randomly. To account for empirical levels of noise, the choice is mixed with a lapse rate (epsilon) that can span up to 1.0 (pure guessing).

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] This design dissociates Tallying from the Weighted Additive (WADD) rule by pitting feature quantity against feature quality. Tallying simply counts the number of features on which an option is superior, ignoring the validities (weights) of those features. WADD, on the other hand, weights each feature's value by its validity. By providing a steep drop-off in validities (e.g., highly predictive first and second features, but moderately predictive third, fourth, and fifth features), we can construct trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by WADD). We also include trials where both models agree, and trials where Tallying sees a tie but WADD strongly prefers one option.

[1] This design systematically dissociates the Weighted Additive (WADD) model from the Tallying heuristic by pitting feature quantity against feature quality. By using a set of validities with a steep drop-off between the top two features and the remaining four, we can construct pairs where one option is supported by a greater number of low-validity features (favored by Tallying) while the other option is supported by fewer but highly predictive features (favored by WADD). The design includes a mix of trials where the models strongly disagree, trials where Tallying perceives a tie but WADD has a clear preference, and baseline trials where both models agree.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) rule, we construct trials where the single most valid discriminating feature (which dictates TTB's choice) points to one option, but a coalition of multiple less valid features points to the other option. Because WADD is compensatory, the combined weight of several less valid features can overcome a disadvantage on the highest validity cue, leading WADD to choose the opposite of TTB. By using a set of validities with a gradual decline (0.95 down to 0.70), we ensure that WADD has enough weight in the lower features to compensate. We include trials that test this dissociation at different levels of the validity hierarchy, as well as trials where both models agree, to provide a comprehensive contrast.

[3] To quantitatively dissociate the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic, we use a set of validities where the most valid feature is highly predictive, but the combined weight of several less valid features can overcome it. TTB strictly follows the single highest-validity discriminating feature, ignoring all others. WADD, however, integrates all features weighted by their validities. We design trials where TTB's chosen option is supported only by the top feature, while the competing option is supported by a coalition of lower-validity features whose combined weight exceeds the top feature's weight. This directly pits the lexicographic stopping rule against full compensatory integration.

[4] To dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + WADD) model, we vary the compensatory support for the unchosen option while keeping the TTB prediction constant. In these trials, the highest-validity discriminating feature always favors Option A. However, the number of lower-validity features favoring Option B systematically decreases across trials. A pure TTB model predicts a constant probability of choosing Option A across these trials, as it ignores all features beyond the first discriminating one. In contrast, the Strategy Selection model predicts that the probability of choosing Option A will increase as the compensatory support for Option B diminishes, because the WADD component's preference shifts from strongly favoring B to favoring A.

[5] To quantitatively dissociate the Strategy Selection Theory (TTB + WADD) from pure Take The Best (TTB), we hold the TTB prediction constant while systematically varying the compensatory support for the alternative option. In Trials 1-4, the most valid feature always favors Option A, meaning pure TTB will predict a constant probability of choosing A across these trials. However, the remaining features are manipulated such that the WADD model's preference shifts from strongly favoring B to strongly favoring A. The Strategy Selection model, which mixes TTB and WADD, predicts that the choice probability for Option A will increase monotonically across these trials as the WADD component's support for A increases. Trials 5-8 replicate this logic at lower levels of the validity hierarchy (when top features are tied).

[6] To quantitatively dissociate pure Take The Best (TTB) from TTB with a Downstream Tallying Mixture, we hold the TTB prediction constant while systematically varying the downstream tally (the equal-weight sum of all unexamined cues lower in validity than the discriminating cue). Pure TTB ignores all cues after the first discriminating one, predicting a constant choice probability across trials where the first discriminating cue is the same. In contrast, the Downstream Tallying Mixture model evaluates the remaining cues. By shifting the downstream tally from heavily favoring the unchosen option to heavily favoring the TTB-chosen option, the mixture model predicts a monotonically increasing choice probability, whereas pure TTB predicts a flat line.

[7] This design quantitatively dissociates pure Take The Best (TTB) from the TTB + Downstream Tallying Mixture model. We hold the primary TTB prediction constant (by keeping the highest-validity discriminating cue the same) while systematically varying the tally of the remaining, unexamined cues. In trials 1-5, the first cue discriminates, and the downstream tally shifts from strongly favoring Option B to strongly favoring Option A. Pure TTB predicts a constant probability of choosing A across these trials. The mixture model, however, predicts a monotonically increasing probability of choosing A. Trials 6-8 replicate this logic but with the third cue as the discriminating cue, testing whether the downstream tally effect persists when the decision process reaches deeper into the cue hierarchy.

[8] To quantitatively dissociate pure Take The Best (TTB) from TTB with Cue-Reading Errors, we hold the highest-validity discriminating cue constant while systematically varying the support of the remaining downstream cues. Pure TTB assumes perfect cue reading and strictly ignores all cues following the first discriminating cue, predicting a constant choice probability across these trials. In contrast, the Cue-Reading Errors model posits a non-zero probability of misreading cues. If a discriminating top cue is accidentally misread as tied (e.g., both 1 or both 0), the decision process continues to the downstream cues. Therefore, as the downstream cues shift from strongly favoring the alternative option to strongly favoring the TTB-chosen option, the error-prone model predicts an increasing choice probability for the TTB-chosen option, whereas pure TTB predicts a flat line.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate pure Take The Best (TTB) from TTB with Cue-Reading Errors, we systematically manipulate the 'depth' of the first truly discriminating cue by varying the number of preceding tied cues. Pure TTB assumes perfect cue reading: it flawlessly skips over any number of tied cues and stops at the first discriminating cue, predicting a constant choice probability for the option favored by that cue regardless of its depth in the hierarchy. In contrast, the Cue-Reading Errors model assumes a non-zero probability of misreading each cue. Under this model, every preceding tied cue carries a risk of being misread as a discriminating cue. If a tie is misread as a discrimination, the search halts prematurely, resulting in a choice that is effectively random (since the error is equally likely to artificially favor either option). Therefore, as the number of leading tied cues increases, the probability of successfully reaching the true discriminating cue decreases. The error-prone model predicts that the choice probability for the target option will regress toward 0.5 as the depth of the discriminating cue increases, whereas pure TTB predicts a flat line across all such trials.",
  "validities": [
    0.95,
    0.9,
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
      0,
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
      1,
      0,
      0,
      0
    ],
    [
      1,
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
      0,
      1,
      0
    ],
    [
      1,
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
      0,
      0
    ],
    [
      0,
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
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
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
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
      1,
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
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3499,
  "candidates_token_count": 635,
  "total_token_count": 6205
}
```
