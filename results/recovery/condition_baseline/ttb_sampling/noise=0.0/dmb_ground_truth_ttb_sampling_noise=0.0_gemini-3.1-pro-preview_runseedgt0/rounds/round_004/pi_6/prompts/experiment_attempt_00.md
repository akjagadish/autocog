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
**Description:** Strategy Mixture Theory: The population is heterogeneous in its decision-making strategies. A large majority of individuals (around 90%) rely on the strict, non-compensatory Take The Best (TTB) heuristic, which explains the dominant lexicographic patterns in most experiments. However, a smaller proportion of the population (around 10%) uses a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities. This mixture preserves the high accuracy of TTB on benchmark tasks while successfully capturing the right-tail sensitivity to opposing cues observed in certain experiments.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- wadd_prop: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    wadd_prop = float(parameters["wadd_prop"])
    
    # Between-subject mixture: 10% of the population uses WADD, 90% uses TTB
    is_wadd = (wadd_prop < 0.10)
    
    if is_wadd:
        # Weighted Additive (WADD) strategy
        val_clip = np.clip(val, 0.001, 0.999)
        log_odds = np.log(val_clip / (1.0 - val_clip))
        ev = np.sum(log_odds * (a - b))
        if ev > 1e-5:
            scores = np.array([1.0, 0.0])
        elif ev < -1e-5:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
    else:
        # Take The Best (TTB) strategy
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner == 0:
            scores = np.array([1.0, 0.0])
        elif winner == 1:
            scores = np.array([0.0, 1.0])
        else:
            scores = np.array([0.5, 0.5])
            
    # Apply softmax response noise and lapse rate
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
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
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use a 4-feature design with strictly descending validities. We design trials where the highest-validity discriminating cue strongly favors one option (driving TTB's choice), while the sheer number of lower-validity cues favors the other option (driving Tallying's choice). We also include trials where Tallying sees a tie (equal number of wins) but TTB has a clear preference, and vice versa. This allows us to separate the non-compensatory, validity-ordered stopping rule of TTB from the compensatory, validity-ignoring counting rule of Tallying.

[1] This design uses 5 features with strictly descending validities to strongly dissociate Tallying from Take The Best (TTB). We include 'compensatory' trials where the highest-validity discriminating cue strongly favors one option (driving TTB's choice), while a larger number of lower-validity cues favors the other option (driving Tallying's choice). We also include trials where Tallying results in a tie (equal number of wins for both options) but TTB has a clear preference due to the validity of the first discriminating cue. This directly contrasts TTB's non-compensatory, validity-ordered stopping rule against Tallying's compensatory, validity-ignoring counting rule.

[2] To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 5-feature design with closely spaced high validities and one lower validity. TTB's strict lexicographic rule dictates that the option with the highest-validity discriminating cue will always be chosen, completely ignoring all lower-validity cues. In contrast, WADD integrates all cues weighted by a non-linear scaling of their log-odds. By setting the validities such that the top validity is only slightly higher than the subsequent ones, we create scenarios where the sum of the weights of the second and third cues securely overpowers the weight of the first cue, even when WADD's scaling parameter (gamma) is high. This directly contrasts TTB's 'one-reason' stopping rule with WADD's compensatory integration.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we use a 5-feature design with a closely spaced set of validities. TTB relies on a strict lexicographic stopping rule, choosing the option favored by the single highest-validity discriminating cue and ignoring all lower-validity cues. In contrast, WADD integrates all cues by weighting them according to a non-linear scaling of their log-odds. By selecting validities that are relatively close to each other, the combined weight of several lower-validity cues in WADD can easily overpower the single highest-validity cue, even when the log-odds are exponentiated by a high gamma parameter. This creates strong compensatory scenarios where WADD predicts one choice (due to the aggregate evidence) while TTB predicts the opposite (due to the single best cue).

[4] To quantitatively dissociate deterministic Take The Best (TTB) from Probabilistic Take The Best (Plackett-Luce Cue Search), we use a 5-feature design with spaced validities. TTB assumes a strict lexicographic stopping rule where the choice is entirely determined by the single highest-validity discriminating cue, meaning the number of lower-validity cues favoring the opposite option should have zero effect on choice probabilities. In contrast, Probabilistic TTB models cue search as a Plackett-Luce process where the probability of encountering a cue first is proportional to its exponentiated validity. Consequently, Probabilistic TTB predicts that adding more lower-validity cues favoring the opposing option will cumulatively draw choice probability away from the option favored by the best cue. By comparing trials where the best cue faces off against a single lower-validity cue versus trials where it faces off against multiple lower-validity cues, we can sharply distinguish TTB's invariant choice probabilities from Probabilistic TTB's sensitivity to the number of opposing cues.

[5] To quantitatively dissociate Probabilistic Take The Best (Plackett-Luce Cue Search) from deterministic Take The Best (TTB), we use a 6-feature design with a linear spread of validities. Deterministic TTB assumes a strict lexicographic stopping rule where the choice is entirely determined by the single highest-validity discriminating cue. Thus, TTB predicts that the probability of choosing the option favored by the best cue remains constant, regardless of how many lower-validity cues favor the opposing option. In contrast, Probabilistic TTB models cue search as a Plackett-Luce process where the probability of encountering a cue first is proportional to its exponentiated validity. Consequently, Probabilistic TTB predicts that adding more lower-validity cues favoring the opposing option will cumulatively draw choice probability away from the option favored by the best cue. By comparing trials where the best cue faces off against a single lower-validity cue versus trials where it faces off against an increasing number of lower-validity cues, we can sharply distinguish TTB's invariant choice probabilities from Probabilistic TTB's sensitivity to the aggregate weight of opposing cues.

[6] To quantitatively dissociate Take The Best (TTB) from the Sequential Evidence Accumulation over log-odds model, we use a 5-feature design with moderately high, descending validities. TTB employs a strict lexicographic stopping rule, halting at the very first discriminating cue regardless of the remaining evidence. The competing model, however, accumulates the log-odds of the validities and only stops when a threshold (theta) is reached. Because the top validity is set to 0.75 (log-odds ~ 1.1), it is not high enough to reliably exceed the competing model's upper threshold limits (up to 2.5) on its own. By pitting the highest-validity discriminating cue against all subsequent lower-validity cues, TTB will deterministically choose the option favored by the single best cue, whereas the evidence accumulation model will integrate the opposing cues and ultimately favor the other option.

[7] To quantitatively dissociate Sequential Evidence Accumulation over log-odds from Take The Best (TTB), we use a 5-feature design with a descending set of validities. TTB employs a strict lexicographic stopping rule, choosing the option favored by the single highest-validity discriminating cue and ignoring all subsequent cues. In contrast, the Sequential Evidence Accumulation model accumulates the log-odds of the validities and halts only when a threshold (theta) is reached. By setting the top validity to 0.85 (log-odds ~ 1.73), it is not high enough to reliably exceed the accumulation model's upper threshold limits (up to 2.5) on its own. By pitting the highest-validity discriminating cue against multiple subsequent lower-validity cues, TTB will deterministically choose the option favored by the single best cue, whereas the evidence accumulation model (for higher theta values) will integrate the opposing cues and ultimately favor the other option.

[8] To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Mixture Theory (90% TTB + 10% WADD), we use a 5-feature design with one highly valid cue and several moderately valid cues. Pure TTB relies exclusively on the single highest-validity discriminating cue and completely ignores the rest. Thus, TTB predicts identical choice probabilities for the option favored by the best cue, regardless of how many lower-validity cues favor the alternative. The Strategy Mixture Theory, however, assumes a 10% subpopulation uses a compensatory Weighted Additive (WADD) strategy. By contrasting 'compensatory' trials (where the best cue favors Option A but the sum of all lower cues strongly favors Option B) with 'non-compensatory' trials (where the best cue and the sum of lower cues both favor Option A), we can detect the 10% WADD component. Pure TTB predicts no difference in choice rates between these trial types, whereas the Mixture model predicts a detectable ~10% drop in preference for the TTB-favored option on the compensatory trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Mixture Theory (which posits 90% TTB and 10% WADD), we employ a 6-feature design. We create a gradient of validities such that the top cue strongly favors one option, while the combination of several lower-validity cues can outweigh it under WADD's log-odds integration. Pure TTB predicts that the choice probability for the option favored by the best cue will be identical across trials, regardless of the lower-validity cues. The Strategy Mixture Theory, however, predicts a systematic deviation (~10%) on trials where WADD opposes TTB. By contrasting trials where WADD and TTB agree versus trials where they conflict, we can detect the presence of the minority WADD strategy.",
  "validities": [
    0.85,
    0.75,
    0.7,
    0.65,
    0.6,
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
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
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
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
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
  "prompt_token_count": 4583,
  "candidates_token_count": 492,
  "total_token_count": 6694
}
```
