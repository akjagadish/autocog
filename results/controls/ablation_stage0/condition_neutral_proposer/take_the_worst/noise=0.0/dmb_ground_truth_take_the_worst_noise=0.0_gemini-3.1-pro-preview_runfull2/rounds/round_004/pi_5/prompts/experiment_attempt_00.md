# experiment_attempt_00

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
**Description:** Tallying with Take-The-Best Tiebreaker: Decision makers primarily use a simple Tallying heuristic, counting the number of features that favor each option. If one option has a higher tally, it is chosen. However, if the tallies are tied, they do not simply guess; instead, they fall back to the Take-The-Best (TTB) heuristic, breaking the tie by choosing the option favored by the single most valid differentiating feature. This tie-breaking influence can be parameterized to allow for both positive reinforcement or penalty depending on the specific cue structures.

**Parameters:**
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- tau: [-2.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary heuristic: Tallying (counting strict wins)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Tie-breaker: Take-The-Best
    if score_a == score_b:
        tau = float(parameters["tau"])
        val = np.asarray(parameters["validities"], dtype=float)
        # Sort indices by descending validity
        order = np.argsort(-val, kind="stable")
        for idx in order:
            if a[idx] > b[idx]:
                score_a += tau
                break
            elif b[idx] > a[idx]:
                score_b += tau
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


## THEORY 2
**Description:** Weighted Additive (WADD) with Subjective Weighting: Decision-makers evaluate options using a fully compensatory strategy where each option's value is the weighted sum of its features. Instead of using the raw validities as weights, they apply a subjective transformation modeled as a power law (validity^gamma). This parameterization allows the model to capture a spectrum of strategies: when gamma is 0, it reduces to Equal-Weight (Tallying); when gamma is 1, it is standard WADD. By constraining gamma to be very small, the model predominantly relies on tallying-like behavior while allowing slight compensatory deviations to capture nuances in specific experiments.

**Parameters:**
- beta: [0.1, 20.0]
- gamma: [0.0, 0.8]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights via a power law
    weights = val ** gamma
    
    # Calculate the weighted sum for each option
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To maximally dissociate Take The Best (TTB) from Tallying, this design uses four features with strictly decreasing validities. The trials are constructed such that the highest-validity discriminating cue frequently favors one option, while the sheer number of discriminating cues favors the other. In some trials, TTB makes a definitive choice while Tallying results in a tie. This creates sharp opposing predictions and distinct patterns of guessing between the two models.

[1] To maximally dissociate Take The Best (TTB) and Tallying, we use 5 features with strictly decreasing validities. The trials are designed such that the single highest-validity discriminating cue often strongly favors one option (driving TTB's choice), while the sheer number of remaining cues favors the other option (driving Tallying's choice). We also include trials where Tallying results in an exact tie (e.g., 2 wins for A, 2 wins for B) while TTB makes a definitive choice based on the first discriminating cue. This creates a combination of sharp opposing predictions and distinct patterns of guessing between the two models.

[2] To maximally dissociate WADD from Tallying, we use 5 features with a steep drop-off in validities. WADD computes a weighted sum, meaning it can strongly prefer an option endorsed by a small number of high-validity features. Tallying discards validities and simply counts the number of winning features. By pitting 2 high-validity features against 3 low-validity features, we create strong dissociations where WADD prefers the option with 2 features while Tallying prefers the option with 3. We also include pairs with an equal number of winning features (which Tallying considers a tie) but where WADD has a clear preference due to the different validities, further pulling apart their predictions.

[3] To maximally dissociate Tallying from WADD, we use 5 features with a steep drop-off in validities. WADD computes a weighted sum and can strongly prefer an option endorsed by a small number of high-validity features. Tallying discards validities entirely, simply counting the number of winning features for each option. By pitting 2 high-validity features against 3 low-validity features, we create strong dissociations where WADD prefers the option with 2 features while Tallying prefers the option with 3. We also include pairs with an equal number of winning features (which Tallying considers a tie) but where WADD has a clear preference due to the different validities.

[4] To dissociate Top-N Tallying (Theory 1) from standard Tallying (Theory 2), we exploit the fact that Theory 1 ignores the lowest validity features (with k_prop in [0.8, 1.0], a 5-feature design means Theory 1 often ignores the 5th feature). By constructing trials where the top 4 features result in a tie (or a 1-point difference) but the 5th feature breaks the tie (or creates one), we generate sharp qualitative differences. Theory 2 will reliably use the 5th feature to break ties or form ties, whereas Theory 1 will guess when the top 4 tie, or stick with its top-4 preference despite the 5th feature.

[5] To effectively dissociate Tallying (Theory 1) from Top-N Tallying (Theory 2), we use a 6-feature design. Theory 2 uses a parameter k_prop in [0.8, 1.0], which means it will typically truncate its consideration to the top 5 features (since round(0.8 * 6) = 5). By manipulating the 6th (lowest validity) feature, we can create sharp divergences. We include trials where the top 5 features result in a tie, but the 6th feature breaks it (Theory 1 will have a strict preference, while Theory 2 will guess). We also include trials where the top 5 features slightly favor one option, but the 6th feature favors the other, resulting in a tie across all 6 features (Theory 1 will guess, while Theory 2 will have a strict preference).

[6] To maximally dissociate Theory 1 (Tallying with TTB Tiebreaker) from Theory 2 (Pure Tallying), we must focus on trials where Tallying results in a tie. Theory 2 will always predict uniform guessing (50/50) on these trials, regardless of the features' validities. Theory 1, however, will invoke the Take-The-Best (TTB) tiebreaker, searching for the highest-validity discriminating feature to shift its preference. By including pairs that tie in Tallying but have clear TTB winners on the 1st, 2nd, or 3rd highest validity features, we can cleanly observe whether subjects use the validity-based tiebreaker or simply guess. We also include non-tie trials to anchor the estimation of the base Tallying parameters (beta and epsilon).

[7] To maximally distinguish Pure Tallying (Theory 1) from Tallying with a Take-The-Best Tiebreaker (Theory 2), the experiment focuses on trials where the two options tie in the number of winning features. On these tie trials, Theory 1 always predicts uniform guessing (50/50), regardless of feature validities. In contrast, Theory 2 breaks the tie by looking at the highest-validity discriminating feature (TTB). By including tie trials where the TTB tiebreaker favors Option A or Option B on different validity levels, we can cleanly observe whether subjects use the validity-based tiebreaker or simply guess. We also include 'unequal tally' trials where both models agree to anchor the estimation of the base Tallying parameters (beta and epsilon).

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
  "rationale": "To maximally distinguish Theory 1 (Tallying with TTB Tiebreaker) from Theory 2 (WADD with Subjective Weighting), we use a 5-feature design with a sharp drop-off in validities between the top two and the bottom three cues. This allows us to create two critical types of dissociation trials. First, in 'Tally Tie' trials, both options have an equal number of winning features. Theory 1 relies on the single highest-validity discriminating cue (TTB) to break the tie, whereas Theory 2 evaluates the subjective weighted sum, which can favor the option with the second and third highest validities combined. Second, in 'Unequal Tally' trials, one option has more winning features (favored by Theory 1's Tallying), but the other option has the two highest-validity features. If Theory 2's gamma parameter is sufficiently high, the subjective weights of the top two features will overpower the three lower-validity features, leading to opposing predictions.",
  "validities": [
    0.98,
    0.95,
    0.55,
    0.52,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      0,
      0,
      0,
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
      1
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
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
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
      1,
      0,
      0,
      1
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
      0,
      1,
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
  "prompt_token_count": 3570,
  "candidates_token_count": 633,
  "total_token_count": 9580
}
```
