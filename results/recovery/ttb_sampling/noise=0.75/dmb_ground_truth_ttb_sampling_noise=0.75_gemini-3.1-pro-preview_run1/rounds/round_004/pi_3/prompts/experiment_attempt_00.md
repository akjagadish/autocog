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
**Description:** Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

**Parameters:**
- beta: [0.0, 2.5]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Strict Take-The-Best with Uniform Lapse: Decision-makers rely on a strict lexicographic heuristic, ranking cues by subjective validity and making a deterministic choice based solely on the highest-validity discriminating cue. To account for empirical noise and inattention, decisions are subject to a uniform lapse rate, where the decision-maker simply guesses randomly on a fixed proportion of trials rather than scaling their confidence by the cue's validity.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[f] > a[f]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design quantitatively dissociates Tallying from WADD by pitting a few high-validity features against several lower-validity features. In Tallying, all features are weighted equally, so an option with more positive features will be chosen regardless of validity. In WADD, the subjective validities determine the weighted sum, allowing a smaller number of highly valid features to outweigh a larger number of less valid ones. The trials include a mix of cases where the two heuristics strongly disagree (e.g., 3 low-validity features vs. 2 high-validity features), cases where they agree, and cases where Tallying predicts a tie while WADD predicts a strict preference.

[1] To quantitatively dissociate WADD from Tallying, we use a set of 5 features with highly skewed validities. Tallying decides solely based on the count of features favoring each option, treating all validities as equal. WADD, on the other hand, weights each feature by its validity. The trial pairs are designed to create direct conflicts: cases where one option wins on more features (favored by Tallying) but the other option wins on fewer, more predictive features (favored by WADD). We also include trials where Tallying predicts a tie (equal number of wins) while WADD predicts a strict preference, and cases where both models agree, to serve as a baseline.

[2] This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive rule (WADD) by systematically pitting a single high-validity feature against multiple lower-validity features. TTB stops at the first discriminating feature, meaning it will favor an option that excels on the highest-validity cue regardless of its values on all other cues. In contrast, WADD integrates all cues weighted by their validities, meaning that a deficit on the best cue can be compensated by advantages on several lesser cues. The trials include extreme compensations (where WADD strongly disagrees with TTB), partial compensations (where WADD predicts a tie but TTB strongly prefers one option), and baseline trials where both models agree.

[3] This design quantitatively dissociates the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic. By using 5 features with linearly decreasing validities, we can construct trials where the single best discriminating feature strongly favors one option (driving TTB's choice), while the sum of multiple lower-validity features favors the other option (driving WADD's choice). We also include trials where the highest validity feature is tied, forcing TTB to drop to the next feature, while WADD integrates all features. This allows us to observe whether subjects compensate for a deficit on the most valid cue by accumulating evidence from lesser cues.

[4] To quantitatively dissociate TTB with Probabilistic Stopping from the Strategy Mixture Model (TTB + Tallying), this design varies the number of features favoring the non-TTB option while keeping the highest valid discriminating feature constant across sets of trials. Under the advocated TTB model, the probability of choosing the TTB-favored option depends solely on the validity of the single best discriminating feature; thus, choice probabilities should remain constant across trials that share the same top discriminator. In contrast, the Mixture model predicts that the choice probability will shift systematically as the Tallying component pulls the decision toward the option with the greater total number of winning features.

[5] To quantitatively dissociate the Strategy Mixture Model (TTB + Tallying) from TTB with Probabilistic Stopping, this design varies the total number of features favoring each option while holding the highest-validity discriminating feature constant across subsets of trials. Under the competing TTB model, the choice probability depends solely on the validity of the first discriminating feature, so trials with the same top discriminator will yield identical choice probabilities. In contrast, the advocated Mixture model integrates a Tallying component, meaning choice probabilities will systematically shift depending on how many total features favor the TTB-winning option versus the TTB-losing option.

[6] To quantitatively dissociate TTB with Probabilistic Stopping (advocated) from Probabilistic Cue Search (competing), this design holds the highest-validity discriminating cue constant across subsets of trials while varying the number and distribution of lower-validity discriminating cues. Under the advocated TTB model, choice probability depends exclusively on the validity of the first (highest-validity) discriminating cue; thus, it predicts identical choice probabilities across all trials sharing the same top discriminating cue, regardless of other cue values. In contrast, the competing Probabilistic Cue Search model samples all discriminating cues proportionally to their validities. Therefore, adding lower-validity cues that favor the opposing option will systematically pull the choice probability away from the top cue's favored option under PCS, while leaving the TTB prediction completely unaffected.

[7] To quantitatively dissociate Probabilistic Cue Search (advocated) from Take The Best with Probabilistic Stopping (competing), this design holds the highest-validity discriminating cue constant across subsets of trials while varying the number of lower-validity discriminating cues that favor the opposing option. Under the competing TTB model, choice probability is determined solely by the validity of the first (highest-validity) discriminating cue, predicting identical choice probabilities across all trials sharing the same top discriminator. In contrast, the advocated Probabilistic Cue Search model samples all discriminating cues proportionally to their validities. Thus, adding more lower-validity cues that favor the opposing option will systematically shift the choice probability away from the top cue's favored option under PCS, while leaving the TTB prediction completely unaffected.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To quantitatively dissociate TTB with Probabilistic Stopping from Strict TTB with Uniform Lapse, this design varies the validity of the first discriminating feature across trials. Under the competing Strict TTB model, the choice probability for the TTB-favored option is determined by a uniform lapse rate and thus remains constant across all trials, regardless of whether the deciding feature is highly valid or barely better than chance. In contrast, the advocated Probabilistic Stopping model scales the choice probability via a softmax function based on the validity of the discriminating feature. By creating a set of trials where the highest-validity discriminating feature systematically shifts from the most valid cue down to the least valid cue, we can observe whether subjects' choice consistency degrades as the deciding cue's validity decreases, which is uniquely predicted by the advocated model.",
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
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1
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
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ],
    [
      1,
      0,
      0,
      1,
      1
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
      1,
      1,
      0,
      1,
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
      0,
      1,
      1,
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
  "prompt_token_count": 3326,
  "candidates_token_count": 513,
  "total_token_count": 5474
}
```
