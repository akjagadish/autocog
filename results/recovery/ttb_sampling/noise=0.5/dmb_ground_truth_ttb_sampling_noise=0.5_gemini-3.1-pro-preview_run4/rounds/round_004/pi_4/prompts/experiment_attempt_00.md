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
**Description:** Stochastic Take-The-Best (Search Error)

**Parameters:**
- p_skip: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Search through features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_skip = float(parameters["p_skip"])
    epsilon = float(parameters["epsilon"])
    
    p_A_core = 0.0
    p_B_core = 0.0
    prob_reach = 1.0
    
    # For each feature, there is a chance (1 - p_skip) to evaluate it correctly.
    # If it discriminates, we stop. Otherwise, or if skipped, we continue to the next.
    for idx in order:
        if a[idx] > b[idx]:
            p_A_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
        elif b[idx] > a[idx]:
            p_B_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
            
    # If all features are skipped or none discriminate, guess randomly
    p_A_core += prob_reach * 0.5
    p_B_core += prob_reach * 0.5
    
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply general response lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Feature Value Averaging (Dilution Effect): Decision-makers do not merely sum the evidence (validities) of all positive features; instead, they compute a partially or fully averaged value of the positive features. Adding lower-validity features to an option 'dilutes' its overall perceived quality, potentially making it less attractive than an option with fewer, but higher-validity, features. This naturally accounts for empirical effects where adding positive features paradoxically decreases an option's choice probability.

**Parameters:**
- gamma: [0.0, 2.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.zeros(2)
    for i in range(2):
        n_pos = np.sum(stim[i])
        if n_pos > 0:
            # When gamma=0, this is Weighted Additive (WADD).
            # When gamma=1, this is pure Feature Averaging.
            scores[i] = np.sum(stim[i] * validities) / (n_pos ** gamma)
        else:
            scores[i] = 0.0
            
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p_core = p / np.sum(p)
    
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] The design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with Weighted Additive (WADD) (which scales feature values by their validities). By using a set of validities with a large spread between the highest and lowest values, we can create a sharp dissociation between the two models. In several critical trials, option A is superior on a larger number of low-validity features, while option B is superior on fewer but much higher-validity features. Tallying will consistently prefer the option with more positive features, while WADD will prefer the option with the higher weighted sum.

[1] This design sharply dissociates WADD from Tallying by leveraging a steep drop-off in feature validities. By pairing two highly valid features with several low-validity features, we create scenarios where Tallying (which simply counts the number of winning features) and WADD (which weights each win by its validity) make conflicting predictions. Specifically, some trials feature an option that wins on many low-validity features (favored by Tallying) against an option that wins on fewer, but highly valid features (favored by WADD). We also include trials where Tallying predicts a strict tie but WADD predicts a strong preference, ensuring a comprehensive quantitative dissociation.

[2] This design quantitatively dissociates Take The Best (TTB) from Weighted Additive (WADD) by contrasting a single highly-valid feature against multiple lower-validity features. In each trial, Option A is superior on the highest-validity feature that discriminates between the two options. TTB, being a non-compensatory lexicographic heuristic, will invariably select Option A because it stops searching as soon as it finds this first discriminating feature. In contrast, Option B is designed to be superior on a larger number of lower-validity features such that their sum outweighs the single highest-validity feature. WADD, being a compensatory model, will integrate all features and therefore consistently prefer Option B.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic by pitting a single, highly-valid feature against multiple lower-validity features. In the critical trials, Option A is superior on the single most valid feature that discriminates between the two options. TTB, which stops searching at the first discriminating feature, will invariably choose Option A. However, Option B is superior on several lower-validity features whose combined validities exceed that of the single feature favored by Option A. WADD integrates all features and will therefore consistently prefer Option B. We vary the specific validities involved (e.g., the top feature vs. features 2 and 3; the second feature vs. features 3 and 4) to ensure the dissociation is robust across the validity hierarchy.

[4] This design dissociates standard Take-The-Best (TTB) from Stochastic Take-The-Best (Search Error). Standard TTB deterministically stops at the first discriminating feature, meaning it will always choose the option favored by the highest-validity discriminating feature (subject only to a uniform lapse rate). Stochastic TTB, however, assumes a probability 'p_skip' of missing a discriminating feature and continuing the search. By pitting a single high-validity feature favoring Option A against multiple lower-validity features favoring Option B, Stochastic TTB predicts a systematic, graded probability of choosing Option B (when the top feature is skipped), whereas standard TTB predicts Option B is only chosen via random lapses. By varying how many top features favor Option A before the lower features favor Option B, we can observe the exponential decay of errors predicted by the Stochastic TTB model.

[5] To quantitatively dissociate Stochastic Take-The-Best (Search Error) from standard Take-The-Best (TTB), we exploit the fact that standard TTB's predictions depend exclusively on the single highest-validity discriminating feature. Any features below it are entirely ignored. In contrast, Stochastic TTB assumes a probability 'p_skip' of missing a discriminating feature and continuing the search. By systematically increasing the number of lower-validity features that favor Option B while keeping the top discriminating feature favoring Option A, standard TTB predicts a constant, high probability of choosing Option A across all conditions. Stochastic TTB, however, predicts that as the number of 'backup' features favoring Option B increases, the probability of choosing Option B will systematically rise, because there are more opportunities for the search process to land on a feature favoring B after skipping the initial feature favoring A.

[6] To dissociate Strategy Selection (Mixture of TTB and WADD) from Stochastic Take-The-Best (Search Error), we exploit their differing sensitivities to the accumulation of lower-validity cues. In Stochastic TTB, the probability of choosing the option favored by lower-validity cues depends almost entirely on the rank of the first cue that favors it (since the model must skip higher-validity cues to reach it). In contrast, the Strategy Selection model includes a WADD component that integrates all cues, meaning the probability of choosing an option will scale with the overall sum of its favorable cues. By holding the rank of the first feature favoring Option B constant (always the 2nd most valid feature) while varying the number of subsequent lower-validity features that favor Option A versus Option B, Stochastic TTB predicts a relatively flat, low probability of choosing B across these trials. The Strategy Selection model, however, predicts a sharp decrease in the probability of choosing B as the WADD component shifts from favoring B to strongly favoring A.

[7] To quantitatively dissociate Stochastic Take-The-Best (Search Error) from Strategy Selection (Mixture of TTB and WADD), we vary the number of lower-validity 'backup' features favoring Option B while keeping the top discriminating feature favoring Option A. In Stochastic TTB, the probability of choosing Option A is predominantly determined by the first discriminating feature (yielding 1 - p_skip probability for A). Adding more features that favor B only captures the small remaining probability mass (p_skip) that leaks past the first feature, resulting in very minor decreases in the probability of choosing A. In contrast, the Strategy Selection model incorporates a Weighted Additive (WADD) component that evaluates the sum of all features. When Option B has many lower-validity features, WADD strongly favors B over A, causing a massive shift in the Strategy Selection model's predicted choice probabilities. By comparing trials where B has one backup feature versus many, we can observe either a tiny shift (supporting Stochastic TTB) or a massive shift (supporting Strategy Selection).

[8] This design quantitatively dissociates Feature Value Averaging (Dilution Effect) from Stochastic Take-The-Best (Search Error) by testing whether adding positive, but lower-validity, features to an option decreases its choice probability. Stochastic TTB searches features in descending order of validity; adding lower-validity features to the option that already wins on the highest-validity discriminating feature will either have no effect or slightly increase its choice probability (by providing 'backup' features if the top one is skipped). In contrast, the Feature Value Averaging model predicts a 'dilution effect': adding lower-validity features reduces the average value of the option's positive features. By comparing baseline trials (where Option A has a single high-validity feature and Option B has multiple lower-validity features) against 'diluted' trials (where Option A is given additional low-validity features), we can observe whether Option A's choice probability drops (supporting Averaging) or remains high (supporting Stochastic TTB).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Stochastic Take-The-Best (Search Error) from Feature Value Averaging (Dilution Effect). In Stochastic TTB, search proceeds in descending order of validity. Adding any positive features to an option can only increase its choice probability, as they provide 'backup' evidence if higher-validity features are skipped due to search errors. In contrast, the Feature Value Averaging model posits that adding lower-validity features to an option 'dilutes' its overall value by dragging down the average validity of its positive features. We test this by starting with a baseline trial where Option A has a single high-validity feature and Option B has a single medium-validity feature. Across subsequent trials, we progressively add low-validity features to Option A. Stochastic TTB predicts Option A's choice probability will monotonically increase (or remain very high), whereas the Averaging model predicts a strong 'dilution effect' where Option A's choice probability drops significantly, potentially leading to a preference reversal in favor of Option B.",
  "validities": [
    0.95,
    0.8,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
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
      0
    ],
    [
      1,
      0,
      1,
      1
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
      0,
      0
    ],
    [
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0
    ],
    [
      0,
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
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3836,
  "candidates_token_count": 453,
  "total_token_count": 5893
}
```
