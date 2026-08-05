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
**Description:** Biased Random Guessing: Due to cognitive overload from processing multiple conflicting binary cues with explicit validities, subjects abandon systematic cue integration and resort to random guessing. However, their guessing is not perfectly uniform; instead, they exhibit a parameterized spatial or positional bias (e.g., a baseline preference for Option A over Option B due to reading order). This allows the model to capture near-random choice behavior while gracefully absorbing slight empirical deviations from exactly 50%.

**Parameters:**
- p_a: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess, but with a spatial/positional bias.
    p_a = float(parameters["p_a"])
    
    # Return the biased probabilities for Option A and Option B.
    return np.array([p_a, 1.0 - p_a])
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
**Description:** Take-The-Best (TTB) heuristic with extreme noise: Decision-makers avoid cognitive overload by not integrating all information. Instead, they search through cues in descending order of their explicitly stated validities. The choice is based entirely on the first cue that discriminates between the two options. However, due to the high cognitive demands of the task or lack of motivation, their behavior is overwhelmingly stochastic, requiring extremely high lapse rates and very low softmax temperatures to capture the near-random empirical choices.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable")
    
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No discriminating cue found
        p_core = np.array([0.5, 0.5])
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * scores
        z -= np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] The design aims to strongly dissociate Take The Best (TTB) from Tallying. TTB relies solely on the most valid discriminating cue, entirely ignoring the number of cues supporting each option. Tallying simply counts the number of supporting cues, ignoring their validities. We use 5 features with descending validities. In several critical trials, one option is favored by the single highest-validity cue (winning under TTB) while the other option is favored by a larger number of lower-validity cues (winning under Tallying). We also include trials where Tallying produces a tie (equal number of winning cues) but TTB makes a deterministic prediction based on the most valid cue.

[1] This design systematically dissociates Tallying from Take The Best (TTB) by pitting the single most valid discriminating cue against a larger quantity of less valid cues. Under TTB, the option with the higher value on the most valid discriminating feature is chosen deterministically, completely ignoring the sheer number of features supporting the other option. In contrast, Tallying simply counts the number of features favoring each option, ignoring validities entirely. The trial set includes cases where TTB strongly prefers one option while Tallying prefers the other, as well as cases where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction based on the top cue.

[2] This design dissociates the advocated theory (WADD with log-odds weighting) from Tallying by contrasting high-validity features with a larger quantity of lower-validity features. The log-odds transformation in WADD heavily stretches the weights of highly valid cues (e.g., 0.95 validity becomes ~2.94, while 0.55 becomes ~0.20). Thus, WADD can strongly prefer an option supported by a single highly valid cue over one supported by multiple weaker cues, whereas Tallying simply counts the number of winning features irrespective of validity. The trials include cases where WADD and Tallying make opposing predictions, as well as cases where Tallying predicts a tie but WADD has a strong preference.

[3] This design quantitatively dissociates Tallying from Weighted Additive (WADD) with log-odds validities. Under Tallying, subjects simply count the number of features favoring each option, treating all validities equally. Under WADD, the validities are transformed via log-odds, dramatically increasing the weight of highly valid cues (e.g., 0.90 becomes ~2.20, whereas 0.75 becomes ~1.10 and 0.55 becomes ~0.20). By pitting a single highly valid cue against multiple moderately valid cues whose log-odds sum precisely equals the single cue's weight, we create trials where WADD predicts indifference (a tie) while Tallying strongly prefers the option with more winning cues. Other trials pit the single best cue against a larger number of weak cues such that WADD prefers the single best cue, but Tallying prefers the multiple weak cues. We also include trials where Tallying predicts a tie but WADD exhibits a strong preference.

[4] This design aims to strongly dissociate the WADD (log-odds) theory from the Cognitive Overload / Random Choice theory. The Random Choice theory posits that subjects simply guess (50/50) on every trial regardless of the feature values. The WADD theory predicts systematic preferences based on the log-odds transformed validities. We include trials with overwhelming evidence for one option (where WADD predicts near-deterministic choice), trials where a single highly valid cue is pitted against multiple weaker cues, and trials with more balanced evidence. This spectrum of evidence strengths allows us to observe whether choice probabilities scale with the expected value difference (as WADD predicts) or remain flat at 0.5 (as Random Choice predicts).

[5] To quantitatively dissociate the Cognitive Overload (Random Choice) theory from the WADD (log-odds) theory, we must present trials across a wide spectrum of evidence strength. The Cognitive Overload theory predicts a flat 50/50 choice probability across all trials, regardless of the features presented. In contrast, the WADD theory predicts choice probabilities that scale with the difference in expected value (calculated using log-odds transformed validities). We include trials where one option completely dominates, trials where a single highly valid cue is pitted against multiple weaker cues, and trials with moderate evidence differences. If subjects are truly overloaded and guessing randomly, their choice distributions will remain at 0.5 even when one option overwhelmingly dominates the other (e.g., possessing all positive features).

[6] To dissociate TTB with extreme noise from Pure Random Choice, we need trials where TTB identifies a clear winner. Although TTB's extreme noise pushes predictions close to 50/50, it still predicts a slight but systematic deviation favoring the option with the highest-validity discriminating cue. Random Choice predicts exactly 50/50 across all trials. By using trials where one option is favored by the most valid cue, we can statistically detect this slight deviation from 0.5.

[7] To quantitatively dissociate Cognitive Overload (pure Random Choice) from Take-The-Best (TTB) with extreme noise, we use pairs of options that either have a clear TTB winner or are completely identical. The Random Choice theory predicts exactly 50/50 choice probabilities across all trials, regardless of the features. The competing TTB theory, despite its extreme noise (epsilon between 0.8 and 1.0), still predicts a slight but systematic deviation from 50/50 in favor of the option with the highest-validity discriminating cue. By including trials where the highest valid cue favors A, trials where it favors B, and trials where all cues are tied (producing exactly 50/50 in both theories), we can test whether there is any systematic deviation from random guessing when a discriminating cue is present.

[8] To quantitatively dissociate the Noisy Take-The-Best (TTB) theory from the Biased Random Guessing theory, we need to manipulate which option is favored by the most valid discriminating cue. The Biased Random Guessing theory posits a constant probability of choosing Option A (p_a) across all trials, regardless of the stimulus features. In contrast, the Noisy TTB theory predicts that despite high levels of noise, choice probabilities will systematically shift in the direction of the TTB winner. By including trials where A is the TTB winner, trials where B is the TTB winner, and trials where both options are identical (a TTB tie), we can test whether the choice probability remains flat (as Biased Guessing predicts) or varies symmetrically around 0.5 based on the highest-validity discriminating cue (as Noisy TTB predicts).

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate Biased Random Guessing from Noisy Take-The-Best (TTB), we manipulate the identity of the TTB winner across trials. Biased Random Guessing predicts a completely flat response profile (a constant probability of choosing A) across all trials, regardless of the stimulus features. In contrast, Noisy TTB predicts that choice probabilities will systematically shift depending on which option is favored by the most valid discriminating cue, even under extreme noise. By including trials where A is the TTB winner, trials where B is the TTB winner, and trials where both options are identical, we can test whether the choice probability for A remains constant or fluctuates based on the TTB winner.",
  "validities": [
    0.9,
    0.7,
    0.6
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      1
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      0,
      1
    ],
    [
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      1
    ],
    [
      1,
      0,
      0
    ],
    [
      1,
      0,
      1
    ],
    [
      1,
      1,
      0
    ],
    [
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
  "prompt_token_count": 3326,
  "candidates_token_count": 314,
  "total_token_count": 4169
}
```
