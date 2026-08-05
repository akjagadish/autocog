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
**Description:** Constant-Mixture Strategy Selection with dampened Tallying: Decision-makers evaluate options using a probabilistic mixture of a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying). Individuals have a fixed, trait-like probability of using TTB versus Tallying on any given trial, independent of cue depth. To account for zero covariance between tally score differences and choices in certain environments, the temperature parameter for the Tallying component can reach zero, dampening its linear tracking of score differences.

**Parameters:**
- alpha: [0.0, 1.0]
- tau: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine Take-The-Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine Tallying prediction using softmax
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    tau = float(parameters["tau"])
    
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)
        
    # Probabilistic strategy switch (constant mixture, independent of depth)
    alpha = float(parameters["alpha"])
    p_core = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
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


## THEORY 2
**Description:** People evaluate options using a non-linearly Weighted Additive (WADD) strategy. Each option's value is the sum of its features weighted by their perceived importance, which is a non-linear power function of the objective cue validities. This allows a single high-validity cue to balance out multiple lower-validity cues, resulting in compensatory trade-offs and choice probabilities near 0.5 on conflict trials.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To distinguish Take The Best (TTB) from Tallying, we design trials where the option favored by the single most valid cue is opposed by the simple majority of features. TTB stops at the first discriminating cue (highest validity), completely ignoring how many subsequent cues favor the other option. Tallying simply counts the number of feature-wise wins for each option, ignoring cue validities. For instance, if Option A has a positive rating only on the most valid cue and Option B has positive ratings on all other cues, TTB will choose A while Tallying will strongly prefer B. We also include trials where the most valid cues are tied to test the fallback mechanisms of both theories.

[1] This design specifically targets trials where Tallying results in a tie (and thus guesses) while TTB has a strong deterministic preference, as well as trials where the two models have completely opposing preferences. By varying which cue is the highest discriminating cue (sometimes the 1st, sometimes the 2nd or 3rd), we can also observe whether subjects follow the TTB cue cascade or simply count the total number of wins. Comparing trials where Tallying guesses vs. strongly prefers an option allows us to precisely measure the influence of the sheer number of winning features against the single highest-validity discriminating cue.

[2] To distinguish the strictly non-compensatory Take The Best (TTB) from the Non-linearly Weighted Additive (WADD) model, we use a 5-feature design with a linear spread of validities. TTB relies entirely on the single highest-validity discriminating cue, ignoring all lower-validity cues. WADD, depending on its gamma parameter, can exhibit compensatory behavior where multiple lower-validity cues can outweigh a single higher-validity cue. By pitting the most valid discriminating cue against varying numbers of lower-validity cues, we can observe whether choices flip (indicating low-gamma WADD) or remain steadfast (indicating TTB or high-gamma WADD). Furthermore, because WADD computes a continuous difference in weighted sums, its choice probabilities (via softmax) will vary across trials depending on the margin of the weighted sum, whereas TTB's choice probabilities depend solely on the binary output of the single discriminating cue, yielding constant confidence across all trials where a cue discriminates.

[3] This design tests a core quantitative divergence between Take The Best (TTB) and the Non-linearly Weighted Additive (WADD) model: sensitivity to the margin of evidence. Across all trials, Option A is favored by the single most valid cue (validity 0.95), while Option B is favored by an increasing number of lower-validity cues, or Option A is favored by an increasing number of supporting cues. Because TTB stops at the first discriminating cue, it predicts identical choice probabilities for Option A across all these trials (a flat line, as the binary TTB score difference is always 1). In contrast, WADD computes a continuous weighted sum, so it predicts a monotonically increasing probability of choosing Option A as the net evidence shifts from strongly favoring B (despite losing the top cue) to strongly favoring A. This provides a stark quantitative dissociation: flat vs. graded choice probabilities.

[4] This experiment specifically targets the 'depth of the first discriminating cue' mechanism unique to Theory 1. Theory 1 posits that subjects probabilistically switch from Take-The-Best (TTB) to Tallying as the first discriminating cue is found deeper in the cue hierarchy (via the gamma decay parameter). To test this, we design trial pairs where TTB and Tallying consistently oppose each other (e.g., TTB favors one option based on the highest available discriminating cue, while Tallying favors the other option based on a simple majority of lower-validity cues). Across trials, we systematically increase the depth of the first discriminating cue (from depth 0 to depth 3) by tying the top N cues. Theory 1 predicts a sharp, depth-dependent decay in the probability of choosing the TTB-favored option as the decision strategy falls back to Tallying. Theory 2 (WADD) evaluates options based on a continuous weighted sum; while the absolute difference in weighted sums may vary slightly across these trials, WADD does not contain a discrete depth-based penalty, leading to a fundamentally different pattern of choice probabilities across the depth gradient.

[5] This design systematically varies the depth of the first discriminating cue while maintaining a consistent conflict between the highest-validity discriminating cue and the simple majority of lower-validity cues. Theory 2 (Context-Dependent Strategy Selection) predicts a discrete, sharp drop in the probability of choosing the TTB-favored option as the depth of the first discriminating cue increases, because it explicitly falls back to Tallying (which favors the other option). Theory 1 (WADD) computes a continuous non-linear weighted sum; it lacks a discrete depth penalty and will instead show choice probabilities that track the continuous differences in weighted evidence, providing a clear quantitative dissociation from Theory 2's depth-decay mechanism.

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
  "rationale": "By holding both the Take-The-Best (TTB) prediction and the Tallying score difference strictly constant across a sequence of trials, we fix the predicted choice probability of Theory 1 (Constant-Mixture TTB + Tallying) to be perfectly flat. However, by systematically swapping which specific lower-validity cues support each option, we smoothly vary the continuous weighted sum difference. Theory 2 (WADD) is sensitive to the exact validities of the supporting cues and thus predicts a monotonically changing choice probability across the sequence. This provides a stark quantitative dissociation: constant vs. graded choice probabilities.",
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
      0,
      1,
      0,
      1,
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
      1,
      1,
      0,
      1,
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
      1,
      1,
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
      1,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3386,
  "candidates_token_count": 613,
  "total_token_count": 6340
}
```
