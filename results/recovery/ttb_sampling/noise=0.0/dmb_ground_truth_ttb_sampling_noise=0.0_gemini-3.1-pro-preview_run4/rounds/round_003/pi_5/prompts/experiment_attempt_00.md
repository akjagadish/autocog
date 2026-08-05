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
**Description:** Decision-makers use a 'Noisy-Perception Take-The-Best' heuristic. They search through cues in descending order of validity, stopping at the first cue that appears to discriminate between the options. However, their perception of each option's feature value is subject to independent noise: with probability p_error, a feature's value (0 or 1) is misread. Consequently, multiple lower-validity cues can cumulatively influence the choice if earlier high-validity cues are misperceived as tied. The perception error is restricted to be very low, ensuring the model remains predominantly non-compensatory, while a decoupled lapse rate accounts for baseline errors.

**Parameters:**
- p_error: [0.0, 0.1]
- epsilon: [0.0, 0.25]
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
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    p_e = float(parameters["p_error"])
    epsilon = float(parameters["epsilon"])
    
    P_A = 0.0
    P_B = 0.0
    P_rem = 1.0
    
    # Search through features in descending order of validity
    for idx in order:
        val_a = a[idx]
        val_b = b[idx]
        
        # Probabilities of perceived cue values given true cue values
        if val_a > val_b:
            p_a_wins = (1.0 - p_e) ** 2
            p_b_wins = p_e ** 2
        elif val_b > val_a:
            p_a_wins = p_e ** 2
            p_b_wins = (1.0 - p_e) ** 2
        else:  # val_a == val_b
            p_a_wins = p_e * (1.0 - p_e)
            p_b_wins = p_e * (1.0 - p_e)
            
        p_tie = 1.0 - p_a_wins - p_b_wins
        
        P_A += P_rem * p_a_wins
        P_B += P_rem * p_b_wins
        P_rem *= p_tie
        
    # If all features tie, guess randomly
    P_A += P_rem * 0.5
    P_B += P_rem * 0.5
    
    p_core = np.array([P_A, P_B])
    
    # Blend with uniform lapse
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
**Description:** People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design tests the quantitative dissociation between Tallying (which counts the number of features won and ignores validities) and WADD (which computes a validity-weighted sum). By using five features with a mix of high and low validities, we can construct trials where one option wins on a greater number of low-validity features, while the other wins on fewer high-validity features. In these crucial trials, Tallying predicts the option with more winning features, whereas WADD predicts the option with the higher weighted sum. Additional control trials where both models agree or where Tallying predicts a tie (but WADD has a strict preference) ensure the full parameter space of both models can be identified.

[1] This design tests the quantitative dissociation between WADD (which computes a validity-weighted sum) and Tallying (which counts the number of features won, ignoring validities). By using six features with a mix of high and low validities, we construct trials where one option wins on a greater number of low-validity features, while the other wins on fewer high-validity features. In these crucial trials, Tallying predicts the option with more winning features, whereas WADD predicts the option with the higher weighted sum. Additional trials where Tallying predicts a tie but WADD has a strict preference, or where both models agree, ensure the full parameter space of both models can be identified.

[2] This design tests the quantitative dissociation between Take-The-Best (TTB) and Weighted Additive (WADD) models. By using five features with a steeply descending set of validities, we can construct trials where one option wins on the single highest-validity discriminating feature (the only feature TTB considers), while the other option wins on multiple lower-validity features whose combined weight exceeds the highest validity. In these crucial trials, TTB predicts the option with the best single feature, whereas WADD predicts the option with the higher weighted sum of all features. Additional control trials where both models agree, or where the highest-validity feature is tied so TTB relies on the second-best feature but WADD still favors the other option, ensure the full parameter space of both models can be identified.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take-The-Best (TTB) heuristic. By using a five-feature environment with a linearly descending set of validities, we construct critical trials where Option A wins on the single highest-validity discriminating feature (leading TTB to invariably choose A), but Option B wins on multiple lower-validity features whose combined weight exceeds that of Option A's single advantage (leading WADD to choose B). We also include control trials where both models agree and where they make subtler compensatory trade-offs, ensuring robust estimation of decision noise and strategy parameters across the parameter space.

[4] This design tests the quantitative dissociation between a pure Take-The-Best (TTB) model and a Strategy Mixture model (which occasionally uses Tallying). By employing five features with descending validities, we construct critical trials where the highest-validity cue points to one option, but the majority of the remaining cues point to the other option. In these trials, pure TTB will consistently choose the option favored by the single best cue, whereas the Mixture model predicts systematic deviations (a lower choice probability for the TTB-favored option) due to the Tallying component favoring the option with more positive cues. Additional trials where both strategies agree, or where Tallying predicts a tie while TTB has a strict preference, allow precise estimation of the lapse rate versus the specific tallying mixing proportion.

[5] To isolate the Strategy Mixture model from pure TTB, we use five features to create trials where the most valid cue favors Option A, but a simple count of positive features (Tallying) favors Option B. Pure TTB views any deviations from the highest-validity cue solely as random noise (constant lapse rate epsilon). In contrast, the Strategy Mixture model predicts a specific, systematic increase in 'errors' on these conflict trials due to the p_tally parameter, while maintaining high accuracy on trials where TTB and Tallying agree. By including a spectrum of trials—full agreement, direct conflict, and partial ties—we can robustly distinguish whether deviations from TTB are purely random or driven by a compensatory Tallying process.

[6] To dissociate pure Take-The-Best (TTB) from Noisy-Perception TTB, we exploit the fact that pure TTB predicts a constant error rate (driven by the uniform lapse rate epsilon) across all trials where the highest-validity discriminating cue points to one option, regardless of the lower-validity cues. In contrast, Noisy-Perception TTB assumes that cue values can be misperceived. Consequently, if the highest-validity cue is misperceived as tied, the model falls back on lower-validity cues. By creating 'conflict' trials (where the top cue favors Option A but all remaining cues favor Option B) and 'alignment' trials (where the top cue and backups all favor Option A), pure TTB predicts identical choice probabilities for Option A, whereas Noisy-Perception TTB predicts significantly more 'errors' (choices for B) on conflict trials because the lower-validity cues act as a trap if the top cue is misread.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To dissociate pure Take-The-Best (TTB) from Noisy-Perception TTB, we vary the number of tied cues that precede the first discriminating cue. Pure TTB stops at the first discriminating cue and ignores all others, predicting a constant choice probability (1 - epsilon/2) regardless of how many cues were tied beforehand. In contrast, Noisy-Perception TTB assumes that cue values can be misperceived. A tied cue has a probability of being falsely perceived as discriminating, which would cause the search to stop prematurely and inject random noise. Consequently, Noisy-Perception TTB predicts a systematic decay in choice certainty (probabilities pulling closer to 0.5) as the number of preceding tied cues increases. By including trials where the first discrimination occurs at cue 1, 2, 3, 4, or 5 (with all subsequent cues tied), we can cleanly isolate this false-discrimination effect from other compensatory mechanisms.",
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
    ],
    [
      1,
      1,
      1,
      0,
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
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
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
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
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
      0,
      0,
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
      1,
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
  "prompt_token_count": 3599,
  "candidates_token_count": 685,
  "total_token_count": 7758
}
```
