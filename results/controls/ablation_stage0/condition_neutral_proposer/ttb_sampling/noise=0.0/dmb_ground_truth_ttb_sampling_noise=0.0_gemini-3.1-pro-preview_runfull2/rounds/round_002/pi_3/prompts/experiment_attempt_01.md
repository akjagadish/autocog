# experiment_attempt_01

## System Prompt

You are a neutral experimental designer in computational cognitive science, working in the Decision Making (Binary Features) domain.

You are given two candidate theories of decision making, THEORY 1 and THEORY 2, each operationalized as code. Your only goal is to design an experiment whose data will best distinguish the two theories — you have no stake in either theory.

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

## THEORY 1
**Description:** Take The Best (TTB): People compare two options by ordering features by their subjective validity and searching through them sequentially. The search stops at the first feature that discriminates between the two options (i.e., one option has a higher value than the other), and the decision is based entirely on that single feature. This non-compensatory strategy ignores all other features, preventing any compensatory trade-offs. If no feature discriminates, the learner guesses. Response noise is modeled via an independent lapse rate epsilon, which replaces the deterministic TTB choice with a uniform random pick.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(validities, kind='stable')[::-1]
    
    # Default to guessing if no cue discriminates
    p_core = np.array([0.5, 0.5])
    
    # Sequential search for the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## THEORY 2
**Description:** Stochastic Take-The-Best (STTB): People use a non-compensatory, one-reason heuristic to compare options, but their search order is probabilistic rather than strictly deterministic. The probability of examining a cue next is determined by a softmax over the subjective validities of the remaining unexamined cues. The search stops at the first feature that discriminates between the two options, and the decision is based solely on that feature. If the selected feature ties, it is ignored and the search continues. If all features are exhausted without a discriminator, the decision maker guesses. This model interpolates between strict Take-The-Best (at high inverse temperature) and the Minimalist heuristic with random cue search (at zero inverse temperature).

**Parameters:**
- beta: [0.0, 200.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("STTB expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    def get_prob(available_cues):
        if len(available_cues) == 0:
            return np.array([0.5, 0.5])
        
        v = validities[available_cues]
        z = beta * v
        z = z - np.max(z)  # numerical stability
        p = np.exp(z)
        p = p / np.sum(p)
        
        ans = np.zeros(2)
        for i, cue_idx in enumerate(available_cues):
            if a[cue_idx] > b[cue_idx]:
                ans[0] += p[i]
            elif b[cue_idx] > a[cue_idx]:
                ans[1] += p[i]
            else:
                new_cues = [c for c in available_cues if c != cue_idx]
                ans += p[i] * get_prob(new_cues)
        return ans

    n_features = len(validities)
    p_core = get_prob(list(range(n_features)))
    
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
[0] To maximally distinguish Tallying (Theory 1) from Weighted Additive (Theory 2), we need a set of trials where the two models make conflicting predictions. Tallying decides solely based on the count of features favoring each option, treating all features as equally important. In contrast, WADD computes a weighted sum using the feature validities. By using a skewed distribution of validities (e.g., [0.95, 0.85, 0.65, 0.55, 0.50]), we can construct trials where one option wins on a smaller number of highly valid features (favored by WADD) while the other option wins on a larger number of less valid features (favored by Tallying). The design also includes trials where Tallying predicts a tie but WADD predicts a clear winner, as well as baseline agreement trials.

[1] To maximally distinguish WADD (Theory 1) from Tallying (Theory 2), we need a set of validities that are highly skewed, allowing a small number of high-validity features to outweigh a larger number of low-validity features. The trials are designed to include strong dissociations where WADD prefers the option with fewer, but highly valid features, while Tallying prefers the option with more, but less valid features. We also include trials where Tallying predicts a tie but WADD has a clear preference, and baseline trials where both models agree.

[2] To maximally distinguish Take The Best (TTB) from the Weighted Additive rule (WADD), we utilize a highly compensatory set of feature validities. TTB decides solely based on the single most valid feature that discriminates between the two options, entirely ignoring the remaining features. In contrast, WADD computes a weighted sum of all features. By constructing trials where one option excels on the single most valid discriminating feature (which TTB will choose) while the other option excels on multiple slightly less valid features (which WADD will choose because their sum outweighs the single high-validity feature), we can strongly dissociate the predictions of the two models.

[3] To maximally distinguish Theory 1 (Weighted Additive, WADD) from Theory 2 (Take The Best, TTB), we use a set of five features with gently decaying validities. This allows us to create 'compensatory' trials where one option is favored by the single most valid discriminating cue (which TTB relies upon exclusively), while the other option is favored by a larger number of slightly less valid cues (which WADD will favor since the sum of their validities outweighs the single best cue). By varying which feature is the first to discriminate, we can systematically dissociate the two models across the entire validity hierarchy.

[4] To distinguish Take The Best (TTB) from Stochastic Take-The-Best (STTB), we use a set of closely spaced validities. TTB strictly orders features by validity and will always base its decision on the single most valid discriminating cue, regardless of how many lower-validity cues point the other way. STTB, however, selects the cue to examine probabilistically based on a softmax over validities. By creating trials where the most valid discriminating cue favors Option A, but several slightly less valid cues favor Option B, TTB will deterministically choose Option A, whereas STTB will predict a non-trivial probability of choosing Option B (since it might sample one of the many cues favoring B first). We also include trials with ties on the most valid cues to test the secondary search probabilities.

## RESPONSE FORMAT
Design one new experiment that maximally distinguishes THEORY 1 from THEORY 2.
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To maximally distinguish deterministic Take The Best (TTB) from Stochastic Take-The-Best (STTB), we use a set of closely spaced validities. TTB strictly orders features by validity and will always base its decision on the single most valid discriminating cue, regardless of how many lower-validity cues point the other way. STTB, however, selects the cue to examine probabilistically based on a softmax over validities. By creating trials where the most valid discriminating cue favors Option A, but several slightly less valid cues favor Option B, TTB will deterministically choose Option A, whereas STTB will predict a non-trivial probability of choosing Option B (since it might sample one of the many cues favoring B first). We also include trials with ties on the most valid cues to test the secondary search probabilities.",
  "validities": [
    0.95,
    0.92,
    0.89,
    0.86,
    0.83,
    0.8
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
      0,
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
      0,
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
      0,
      0
    ],
    [
      1,
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
      1,
      1
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
      1,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      1,
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
      0,
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
      0,
      0,
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
      1,
      0,
      0,
      0
    ],
    [
      1,
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
  "prompt_token_count": 3064,
  "candidates_token_count": 652,
  "total_token_count": 5222
}
```
