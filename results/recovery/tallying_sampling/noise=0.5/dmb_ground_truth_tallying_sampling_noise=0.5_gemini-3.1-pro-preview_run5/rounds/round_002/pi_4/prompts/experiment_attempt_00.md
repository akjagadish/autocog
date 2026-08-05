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
**Description:** Noisy-Validity Take-The-Best (NV-TTB) Model: Decision makers use a strict lexicographic search (Take-The-Best), consulting cues in descending order of their validity and stopping at the first cue that discriminates between options. However, subjects have noisy internal representations of cue validities. On each choice, Gaussian noise is added to the objective validities, and cues are sorted based on these noisy values. This probabilistic cue ordering allows the model to capture aggregate deviations from pure TTB (and approach Tallying-like behavior when noise is high) while preserving the non-compensatory, one-reason decision mechanism at the single-trial level.

**Parameters:**
- sigma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    epsilon = float(parameters["epsilon"])
    
    n_features = stim.shape[1]
    
    # Marginalize over noisy validities via sampling to produce choice probabilities
    n_samples = 200
    noise = np.random.normal(0, sigma + 1e-9, size=(n_samples, n_features))
    noisy_validities = validities + noise
    
    # Sort cues for each sample (descending order of noisy validity)
    cue_orders = np.argsort(-noisy_validities, axis=1)
    
    a, b = stim[0], stim[1]
    wins = np.zeros(2)
    
    for i in range(n_samples):
        winner = None
        for j in cue_orders[i]:
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            wins += 0.5
        else:
            wins[winner] += 1.0
            
    p_core = wins / n_samples
    
    # Incorporate lapse rate
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


## COMPETING THEORY
**Description:** Tallying (Equal-Weight) Theory: People compare two options by simply counting the number of positive cues (features) for each option and choosing the one with the higher count. This theory posits that decision-makers ignore the varying validities of the cues, treating all features as equally important. It is a compensatory strategy because multiple cues can outweigh a single cue, but it is more frugal than a Weighted Additive (WADD) rule because it avoids multiplying by or storing cardinal validities. Response noise is modeled via a softmax function over the tally scores, along with an independent lapse rate.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.1, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: count the number of positive cues for each option
    # Since cues are binary (0 or 1), we can just sum them.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design dissociates the non-compensatory 'Take The Best' (TTB) heuristic from the fully compensatory Weighted Additive (WADD) rule. By establishing a set of validities with a clear descending order but relatively close values, we can construct 'compensatory' trials where the single best discriminating cue favors Option A, but a coalition of lower-validity cues strongly favors Option B. TTB will exclusively follow the highest-validity discriminating cue and choose A, ignoring the lower-validity cues. In contrast, WADD integrates all cues weighted by their validities, allowing the sum of the lower-validity cues to eclipse the single best cue, thus choosing B. We also include 'control' trials where both models agree, ensuring that differences in model fit are driven by the specific cue-conflict structure rather than an overall failure to capture behavior.

[1] This design tests the compensatory nature of the Weighted Additive (WADD) rule against the non-compensatory Take The Best (TTB) heuristic by presenting choices where the single most valid discriminating cue is pitted against a coalition of multiple less valid cues. Validities are spread such that the highest validity cue is very strong (0.95), but the sum of the remaining cues is substantially larger. In the critical 'compensatory' trials, TTB will always select the option favored by the single highest-validity discriminating cue, ignoring the rest. WADD, evaluating the weighted sum of all cues, will choose the option favored by the coalition of lower-validity cues. We also include 'control' trials where both models agree (e.g., the highest validity cue and the sum of lower validity cues point to the same option) to ensure that overall task engagement is captured.

[2] To dissociate the Tallying (Equal-Weight) theory from the Weighted Additive (WADD) theory, we pit a small number of highly valid cues against a larger number of less valid cues. Tallying ignores cue validities and simply counts the number of positive features, thereby favoring the option with the higher feature count. WADD, on the other hand, computes a validity-weighted sum, allowing a few highly predictive features to outweigh a larger number of weak features. By carefully selecting validities such that the sum of two high-validity cues exceeds the sum of three low-validity cues, we create critical trials where Tallying and WADD make diametrically opposite predictions. We also include trials where both models agree, and trials where feature counts are tied (so Tallying guesses) but WADD has a strong preference.

[3] To firmly dissociate the Weighted Additive (WADD) theory from the Tallying (Equal-Weight) theory, we pit a small number of high-validity cues against a larger number of low-validity cues. Tallying ignores cue validities and simply counts the number of positive features, thus favoring the option with the higher feature count. WADD computes a validity-weighted sum, allowing fewer, highly predictive features to outweigh a larger number of weak features. By using validities where the sum of the two best cues (0.95 + 0.85 = 1.80) exceeds the sum of the three worst (0.55 + 0.55 + 0.50 = 1.60), we create critical trials where Tallying and WADD make diametrically opposite predictions. We also include trials where feature counts are tied (Tallying guesses, but WADD has a strong preference) and control trials where both models agree.

[4] To dissociate Tallying from Noisy-Validity Take-The-Best (NV-TTB), we pit a single highly valid cue against a coalition of multiple less valid cues. Tallying entirely ignores cue validities, simply summing the number of positive features. NV-TTB, on the other hand, lexicographically searches cues based on their (noisy) validities, meaning it will typically stop at the most valid discriminating cue. We include trials where Tallying strongly favors one option (e.g., it has 3 or 4 positive cues) but NV-TTB favors the other option because it possesses the single most valid discriminating cue. We also include trials where Tallying is indifferent (equal number of positive cues) but NV-TTB exhibits a strong preference, ensuring clear quantitative separation between the two theories.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "To cleanly dissociate Noisy-Validity Take-The-Best (NV-TTB) from Tallying, we construct trials that pit the highest-validity cue against a larger quantity of lower-validity cues. NV-TTB searches lexicographically and will stop at the most valid discriminating cue, thus choosing the option favored by the single best cue regardless of the overall count of positive features. Tallying ignores validity entirely and simply counts the number of positive features, choosing the option with the higher total. We include critical trials where one option has fewer positive cues but possesses the highest-validity cue (NV-TTB and Tallying make opposite predictions), trials where the feature counts are tied but one has higher-validity cues (Tallying guesses, NV-TTB has a strong preference), and control trials where both models agree.",
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
      0,
      0,
      1,
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
      0
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
      1,
      0,
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
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3143,
  "candidates_token_count": 458,
  "total_token_count": 4471
}
```
