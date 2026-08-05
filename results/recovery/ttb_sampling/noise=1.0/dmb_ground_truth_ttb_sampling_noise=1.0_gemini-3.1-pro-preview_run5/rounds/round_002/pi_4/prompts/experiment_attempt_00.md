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
**Description:** Stochastic Take-The-Best (Lexicographic): Decision-makers search through features sequentially in descending order of their objective validities. The process is stochastic: at each step, the decision-maker may prematurely terminate the search and guess. If they evaluate a cue and it discriminates between the options, they choose the superior option with high probability, but may choose the inferior option due to execution error or misperception. If the cue does not discriminate, the search continues to the next most valid cue. This algorithmic mechanism generates high stochasticity and intermediate choice proportions without relying on compensatory weighted sums.

**Parameters:**
- p_stop: [0.0, 1.0]
- p_error: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    p_stop = float(parameters["p_stop"])
    p_error = float(parameters["p_error"])
    
    # Sort features by validity in descending order
    # Using mergesort for stable sorting in case of tied validities
    order = np.argsort(validities, kind='mergesort')[::-1]
    
    p_A = 0.0
    p_B = 0.0
    p_guess = 0.0
    
    reach_prob = 1.0
    
    for idx in order:
        # At each step, there is a probability of prematurely terminating the search and guessing
        prob_stop_here = reach_prob * p_stop
        p_guess += prob_stop_here
        reach_prob -= prob_stop_here
        
        a_val = stim[0, idx]
        b_val = stim[1, idx]
        
        if a_val > b_val:
            # Cue discriminates in favor of A
            p_A += reach_prob * (1.0 - p_error)
            p_B += reach_prob * p_error
            reach_prob = 0.0
            break
        elif b_val > a_val:
            # Cue discriminates in favor of B
            p_B += reach_prob * (1.0 - p_error)
            p_A += reach_prob * p_error
            reach_prob = 0.0
            break
        else:
            # Cue does not discriminate (tie), continue search
            pass
            
    # If search completes and no cue discriminated, guess
    p_guess += reach_prob
    
    # Distribute guessing probability equally
    p_A += p_guess * 0.5
    p_B += p_guess * 0.5
    
    return np.array([p_A, p_B])
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
**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Parameters:**
- gamma: [0.0, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
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
[0] To dissociate Tallying from WADD, we must present choices where one option has more positive features (favored by Tallying) but the alternative option's fewer positive features have much higher validities (favored by WADD). By using a set of 5 features with a steep drop-off in validities (e.g., two very high, three relatively low), we can create trials where the sheer count of features points to one option while the weighted sum points to the other. The design includes both congruent and incongruent trials to isolate the specific contribution of feature weighting versus simple counting.

[1] To quantitatively dissociate WADD from Tallying, we use a 5-feature design with a steep drop-off in validities (two high validities, three low validities). This allows us to construct 'incongruent' trials where one option wins on the sheer count of features (favored by Tallying) while the other option wins on the sum of validities due to possessing fewer but much more predictive features (favored by WADD). We also include trials where Tallying sees a tie but WADD has a strong preference, as well as congruent trials where both models agree. This variety ensures that the distinct response patterns of simple counting versus weighted integration can be identified.

[2] To quantitatively dissociate Probabilistic Cue Sampling (PCS) from the Weighted Additive rule (WADD), this design exploits the core difference in their integration mechanisms: PCS uses a softmax over validities to sample a single feature, amplifying the influence of the highest-validity cue, whereas WADD linearly integrates all features. We use four cues with linearly spaced validities. We include 'compensatory' trials where one option possesses the single highest-validity cue but the other possesses all remaining cues (PCS prefers the former, WADD strongly prefers the latter). We also include 'tie-breaker' trials where the sum of validities is equal across both options (yielding a tie in WADD), but the convexity of the softmax in PCS leads to a strong preference for the option with the most valid cue.

[3] This design exploits the fundamental difference between linear integration (WADD) and softmax-based probabilistic sampling (PCS). We use five cues with linearly spaced validities: [0.9, 0.8, 0.7, 0.6, 0.5]. We construct pairs where the difference in the sum of validities is identical (e.g., +0.2, -0.2, or exactly 0). Because WADD is a linear model, it predicts identical choice probabilities across trials with the same validity difference. In contrast, PCS uses a softmax function over validities to determine sampling probabilities. Due to the strict convexity of the exponential function, the difference in sampling probabilities for two cues with a fixed validity difference (e.g., 0.9 vs 0.7 compared to 0.7 vs 0.5) is much larger at the high end of the validity scale. Moreover, for trials where the sum of validities is perfectly tied (e.g., 0.9 + 0.5 vs 0.8 + 0.6), WADD predicts exactly 50/50 guessing, whereas the convexity of the PCS softmax assigns a higher sampling probability to the extreme cues, leading PCS to systematically prefer the option with the single highest validity cue. This provides a crisp, parameter-free quantitative dissociation between the two theories.

[4] This design dissociates Probabilistic Cue Sampling (PCS) from Stochastic Take-The-Best (STTB) by exploiting their different responses to tied top-validity cues and multiple lower-validity cues. STTB searches sequentially in order of validity, terminating as soon as a cue discriminates. Thus, if the highest-validity cues tie, STTB simply moves to the next highest cue and bases its decision entirely on that single cue. In contrast, PCS samples a single cue proportionally to the softmax of validities. If the top cues tie, sampling them leads to guessing, allowing the accumulation of multiple lower-validity cues to dominate the overall choice probability. By presenting trials where the most valid discriminating cue favors option A, but a larger number of lower-validity cues favor option B, STTB will systematically choose A, whereas PCS will assign a higher aggregate probability to B.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design dissociates Stochastic Take-The-Best (STTB) from Probabilistic Cue Sampling (PCS) by exploiting their divergent handling of tied cues and evidence accumulation. STTB evaluates cues strictly in descending order of validity; if the highest validity cue(s) tie, it skips them without accumulating evidence and bases its decision entirely on the first discriminating cue. In contrast, PCS samples cues based on a softmax distribution over validities. When top cues tie, sampling them leads to a 50/50 guess, effectively diluting the choice probability and allowing a larger number of lower-validity cues to dominate the aggregate decision. By presenting trials where the top cue ties and the next highest cue favors Option A, but all remaining lower-validity cues favor Option B, STTB will systematically choose A, whereas PCS will assign a higher overall probability to B.",
  "validities": [
    0.9,
    0.85,
    0.8,
    0.75,
    0.7
  ],
  "trial_a_ratings": [
    [
      1,
      1,
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
      1,
      0,
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
      0,
      1
    ]
  ],
  "trial_b_ratings": [
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
      1,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3389,
  "candidates_token_count": 456,
  "total_token_count": 5754
}
```
