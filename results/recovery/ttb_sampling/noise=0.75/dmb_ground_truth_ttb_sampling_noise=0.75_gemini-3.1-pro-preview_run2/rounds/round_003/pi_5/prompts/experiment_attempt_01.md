# experiment_attempt_01

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
**Description:** Probabilistic Cue Selection (Stochastic TTB): Human decision-makers use a non-compensatory, one-reason heuristic but are stochastic in their cue retrieval. Instead of strictly ranking features by validity, subjects sample a feature to evaluate with a probability proportional to its validity (via a softmax). If the sampled feature discriminates between the options, they base their choice entirely on that feature. If it does not discriminate, they sample again. This maintains the non-compensatory nature of the decision while naturally introducing variability in which cue is selected, offering a mechanistic explanation for choice noise without relying on compensatory tallying.

**Parameters:**
- gamma: [0.0, 50.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between A and B
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, the decision is a random guess
        p_core = np.array([0.5, 0.5])
    else:
        # The subject repeatedly samples features proportional to exp(gamma * validity)
        # until a discriminating feature is found. The probability that the first
        # discriminating feature found is feature i is equivalent to a softmax over
        # the validities restricted to the set of discriminating features.
        z = gamma * validities[discrim]
        z = z - np.max(z)  # For numerical stability
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The chosen discriminating feature dictates the choice entirely.
        # Sum the probabilities of sampling a feature that favors A vs B.
        favor_a = (a[discrim] > b[discrim])
        favor_b = (b[discrim] > a[discrim])
        
        p_a = np.sum(w[favor_a])
        p_b = np.sum(w[favor_b])
        
        p_core = np.array([p_a, p_b])
        
    # Blend with a uniform lapse rate for general response noise/inattention
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, we use a 5-feature design with a skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, ignoring the validities entirely. WADD weights each feature by its validity, meaning a few highly valid features can outweigh several less valid ones. The trial pairs are designed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option. This mix of congruent, incongruent, and tie trials provides a robust test to identify which strategy subjects are using.

[1] To quantitatively dissociate Tallying from the Weighted Additive (WADD) rule, this design uses a 5-feature task with a highly skewed distribution of validities. Tallying simply counts the number of features on which an option is superior, strictly ignoring the magnitude of the validities. In contrast, WADD weights each feature by its validity, allowing a few highly valid features to outweigh a larger number of less valid ones. The trial pairs are strategically constructed to include strong dissociations (where Tallying predicts one option because it wins on more features, but WADD predicts the other because it wins on the most important features), as well as tie-breaking trials (where Tallying predicts a tie due to an equal number of winning features, while WADD strongly prefers the option with the higher-validity features).

[2] To quantitatively dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) rule, we use a 5-feature design with a set of validities that do not form a non-compensatory environment (i.e., the validity of the best feature is strictly less than the sum of the validities of the remaining features). TTB is a lexicographic strategy that bases its choice entirely on the single most valid discriminating feature, ignoring all other features. In contrast, WADD computes a compensatory weighted sum of all features. The trial pairs are specifically designed such that the option favored by the single most valid feature (which TTB will choose) is opposed by multiple lower-validity features that cumulatively outweigh the best feature (which WADD will choose). By including both congruent trials (where both models agree) and incongruent trials (where the 'best' cue points one way but the 'rest' of the cues point the other), we can clearly distinguish whether subjects are using a one-reason heuristic or a compensatory integration strategy.

[3] To quantitatively dissociate the Weighted Additive (WADD) rule from the Take-The-Best (TTB) heuristic, we employ a 5-feature design where the validities form a compensatory environment (the most valid cue is outweighed by the sum of the remaining cues). TTB is a non-compensatory lexicographic strategy that decides based solely on the single most valid discriminating feature. In contrast, WADD integrates all features weighted by their validities, allowing multiple lower-validity cues to compensate for a disadvantage on the highest-validity cue. The trial pairs contain strong incongruencies where the option favored by the single best cue is opposed by the option favored by a combination of lesser cues. We also include congruent pairs to act as catch trials and ensure subjects do not artificially adapt to a purely incongruent task.

[4] To quantitatively dissociate pure Take-The-Best (TTB) from a Hybrid TTB+Tallying model, we exploit a key property of the pure TTB model: its predicted choice probability depends entirely on the single most valid discriminating feature and is invariant to the number of remaining cues that support or oppose that choice. In contrast, the Hybrid model mixes TTB with a Tallying process, which counts the total number of winning features for each option. By holding the TTB prediction constant (e.g., Option A always wins on the most valid feature) while parametrically varying the Tallying support (from strongly opposing A to strongly supporting A), the pure TTB model predicts a flat, constant choice probability across these conditions, whereas the Hybrid model predicts a graded modulation of choice probability. The trial set includes a full spectrum of Tallying support levels for both Option A and Option B TTB-winners.

[5] This design quantitatively dissociates the pure Take-The-Best (TTB) heuristic from the Hybrid (TTB + Tallying) model by testing for strict preference reversals. Pure TTB is strictly non-compensatory: it predicts that the option favored by the single most valid discriminating feature will ALWAYS be preferred (choice probability > 0.5), regardless of how many lower-validity cues oppose it. In contrast, the Hybrid model mixes TTB with a compensatory Tallying process. If the Tallying process strongly opposes the TTB prediction (e.g., the TTB winner is opposed by 5 or 6 lower-validity cues), the Hybrid model allows the choice probability to cross the 50% threshold, resulting in a preference reversal. By parametrically increasing the number of opposing cues (from 1 to 6) against the single best cue, we can detect whether subjects exhibit this compensatory preference reversal (supporting the Hybrid model) or maintain a constant, non-compensatory preference (supporting pure TTB).

[6] To quantitatively dissociate Take-The-Best (TTB) from Probabilistic Cue Selection (Stochastic TTB), we manipulate the number and validities of subordinate discriminating features while holding the single best discriminating feature constant. Under pure TTB, the decision is strictly based on the most valid discriminating feature; the presence, number, or validities of other lower-ranking discriminating features have no effect on the predicted choice probabilities (the pre-noise score is always 1 for the TTB winner and 0 for the loser). In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). Thus, Stochastic TTB predicts that the choice probability for the option favored by the best cue will decrease as more subordinate cues favor the opposing option, whereas pure TTB predicts a constant choice probability across these conditions. By systematically varying the opposing subordinate cues, we can directly test whether choice probabilities are invariant to subordinate cues (TTB) or sensitive to them (Stochastic TTB).

[7] To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), we exploit their fundamentally different treatments of subordinate discriminating features. Standard TTB is strictly lexicographic: the choice and its predicted probability (before uniform lapse) depend entirely on the single highest-validity discriminating feature, with all other features strictly ignored. In contrast, Stochastic TTB samples among ALL discriminating features with probabilities proportional to exp(gamma * validity). By systematically varying the number and validities of the subordinate cues that oppose the highest-validity cue, we can test whether choice probabilities remain constant (as predicted by pure TTB) or shift in proportion to the opposing cues' validities (as predicted by Stochastic TTB). We include trials where the best cue is opposed by multiple highly valid cues, potentially causing a preference reversal under Stochastic TTB but not TTB.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), this design strictly controls the number of discriminating features while systematically varying the validity of the opposing feature. Standard TTB's predicted choice probability depends only on the single highest-validity discriminating feature; it predicts perfectly constant choice probabilities across any trials where the best discriminating feature is the same, regardless of the validity of the single opposing feature. In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). By creating pairs where exactly two features discriminate (one favoring A, one favoring B) and holding the best feature constant while lowering the validity of the opposing feature, Stochastic TTB predicts a graded increase in the choice probability for A, whereas standard TTB predicts a flat, constant probability.",
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
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
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
      1
    ],
    [
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      1,
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
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1,
      1,
      1
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
      1,
      1,
      1,
      1
    ],
    [
      0,
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
      1
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1
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
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3983,
  "candidates_token_count": 657,
  "total_token_count": 6958
}
```
