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
**Description:** Hybrid Heuristic Theory: Decision-makers integrate evidence from multiple strategies before making a choice, rather than probabilistically sampling between distinct strategies. Specifically, individuals compute a combined subjective value for each option by taking a weighted average of normalized compensatory (validity-weighted) and non-compensatory (tallying) evidence. A single stochastic decision process then operates on these integrated values.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Normalized WADD scores
    sum_val = np.sum(val)
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    # Normalized Tallying scores
    score_tal_a = float(np.sum(a > b)) / n_features
    score_tal_b = float(np.sum(b > a)) / n_features
    
    beta = float(parameters["beta"])
    w_mix = float(parameters["w_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Combined integrated scores
    score_combined_a = w_mix * score_wadd_a + (1.0 - w_mix) * score_tal_a
    score_combined_b = w_mix * score_wadd_b + (1.0 - w_mix) * score_tal_b
    scores = np.array([score_combined_a, score_combined_b])
    
    # Single Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Probabilistic Strategy Selection Theory: Decision-makers maintain a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive/WADD). On each trial, they probabilistically sample one strategy from this repertoire according to base-rate strategy weights, and execute it purely. This discrete mixture of strategies accounts for intermediate aggregate choice proportions while naturally predicting exact ties when the sampled strategy itself encounters a tie.

**Parameters:**
- w_ttb: [0.0, 1.0]
- w_tal: [0.0, 1.0]
- w_wadd: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB)
    diff = a - b
    order = np.argsort(val)[::-1]
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if diff[idx] > 0:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif diff[idx] < 0:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Tallying
    tal_diff = np.sum(a) - np.sum(b)
    if tal_diff > 0:
        tal_pred = np.array([1.0, 0.0])
    elif tal_diff < 0:
        tal_pred = np.array([0.0, 1.0])
    else:
        tal_pred = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD)
    wadd_diff = np.sum(val * a) - np.sum(val * b)
    if wadd_diff > 0:
        wadd_pred = np.array([1.0, 0.0])
    elif wadd_diff < 0:
        wadd_pred = np.array([0.0, 1.0])
    else:
        wadd_pred = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    w_arr = np.array([w_ttb, w_tal, w_wadd])
    sum_w = np.sum(w_arr)
    if sum_w == 0:
        p_strat = np.array([1/3, 1/3, 1/3])
    else:
        p_strat = w_arr / sum_w
        
    p_core = p_strat[0] * ttb_pred + p_strat[1] * tal_pred + p_strat[2] * wadd_pred
    
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
[0] To strongly dissociate Take The Best (TTB) from Tallying, we use 5 features with distinct validities. The trials are designed such that the highest-validity discriminating cue points to one option (which TTB will choose), while the majority of the remaining lower-validity cues point to the other option (which Tallying will choose). This creates a direct opposition between the 'one-reason' logic of TTB and the unweighted feature-counting of Tallying.

[1] To uniquely isolate Tallying from Take The Best (TTB), we use 4 features with strictly decreasing validities. The trials are constructed to yield three types of dissociations: (1) Direct contradictions where the single highest-validity discriminating cue favors one option (TTB's choice) but the sheer number of lower-validity cues favors the other (Tallying's choice); (2) Partial dissociations where TTB has a strong preference based on the first cue, but Tallying predicts a perfect tie (guessing) because both options have an equal number of positive features; (3) Cases where the top feature is tied, forcing TTB to rely on the second feature, while Tallying counts the remaining features. This combination of oppositions and ties robustly separates the unweighted counting mechanism of Tallying from the sequential, one-reason stopping rule of TTB.

[2] To cleanly dissociate WADD (Weighted Additive with non-linear scaling) from Tallying, we use 5 features with a wide spread of validities. Tallying purely counts the number of winning features, completely ignoring the validities (treating all features equally). WADD computes a weighted sum, where weights are validities scaled by a non-linear parameter gamma. We design trials that fall into three categories: (1) Ties in Tallying where WADD has a strong preference because it respects the validities of the winning features; (2) Contradictions where Tallying prefers an option with a higher quantity of low-validity features, while WADD prefers the option with fewer but much higher-validity features (especially for gamma > 1); (3) Alignments with different confidence levels. This thoroughly isolates the feature-counting mechanism of Tallying from the validity-weighted compensatory mechanism of WADD.

[3] To effectively dissociate Tallying from WADD, we must exploit Tallying's strict unweighted counting mechanism against WADD's validity-weighted (and non-linearly scalable) compensatory mechanism. We use 5 features with a wide spread of validities. The trials are crafted to produce three types of dissociations: (1) 'Tallying Ties' where both options have an equal number of positive features (Tallying predicts guessing), but WADD strongly prefers the option with higher-validity features; (2) 'Quantity vs. Quality' contradictions where Tallying prefers the option with more positive features (e.g., 3 low-validity vs. 2 high-validity), while WADD prefers the option with fewer but more predictive features; (3) 'Non-linear WADD shifts' where Tallying firmly predicts one option, but WADD's preference flips depending on the gamma parameter (e.g., heavily upweighting the single best cue).

[4] The design sharply dissociates the Advocated Weighted Additive (WADD) theory with Non-Linear Cue Scaling from the Competing Hybrid Heuristic Theory. The Hybrid Theory assumes a linear combination of normalized WADD (linear) and Tallying scores. Thus, if both linear WADD and Tallying agree on an option, the Hybrid model is mathematically constrained to strongly prefer that option, regardless of its mixing parameter. However, the Advocated WADD theory uses a non-linear scaling parameter (gamma) on validities. For gamma > 1, it heavily upweights the most valid cues, and for gamma < 1, it approaches equal weighting. We exploit this by designing 'Trap Trials' where linear WADD and Tallying both point to Option B (so Hybrid must choose B), but the non-linear scaling of WADD points to Option A (because Option A possesses the single most valid cue, overcoming the sum of lower validities). We also include 'Contradiction Trials' where linear WADD and Tallying oppose each other, ensuring the Hybrid model's mixing parameter can be properly identified, making the test strictly fair and highly diagnostic.

[5] To cleanly dissociate the Advocated Hybrid Heuristic Theory from the Competing Weighted Additive (WADD) Theory with Non-Linear Cue Scaling, we exploit the functional form of their effective cue weights. The Hybrid theory assumes a linear combination of normalized WADD (which is linear in validity) and Tallying (which applies a uniform weight to all features). Consequently, the Hybrid model's effective weights are always strictly linear with respect to cue validities. In contrast, the Competing WADD theory scales validities by an exponent (gamma), producing weights that are convex (gamma > 1) or concave (gamma < 1) with respect to validity. We use 5 features with linearly spaced validities to test this. For example, in trials where the sum of validities is equal (e.g., cues 1 and 4 vs. cues 2 and 3), the Hybrid model perfectly ties regardless of its mixing parameter, whereas the Non-Linear WADD model will predictably diverge from a tie depending on its gamma parameter. By including a mix of these 'linear-tie' trials alongside classical Tallying vs. linear WADD contradictions, we can perfectly isolate the linear interpolation of the Hybrid model from the non-linear scaling of the Competing model.

[6] This design strictly dissociates the Advocated Probabilistic Strategy Selection Theory from the Competing Hybrid Heuristic Theory by exploiting the absence of 'Take-The-Best' (TTB) in the Hybrid model. The Hybrid model strictly integrates Weighted Additive (WADD) and Tallying scores. In trials where WADD and Tallying perfectly agree on an option, the Hybrid theory is forced to predict a strong preference for that option (bounded only by the lapse rate). However, the Advocated theory assumes TTB is a discrete strategy in the repertoire. By creating 'Trap Trials' where WADD and Tallying strongly prefer Option B, but the single highest-validity cue points to Option A (triggering TTB), the Advocated model can predict a substantial proportion of Option A choices (proportional to the base-rate weight of TTB). These qualitative divergences, combined with varying degrees of opposition between TTB, WADD, and Tallying across trials, allow for a robust quantitative dissociation of the two theories.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "The design sharply dissociates the Advocated Hybrid Heuristic Theory from the Competing Probabilistic Strategy Selection Theory by exploiting the fundamental difference in their strategy repertoires. The Advocated theory posits that decisions are made by integrating evidence from Weighted Additive (WADD) and Tallying strategies into a single subjective value, without an independent 'Take-The-Best' (TTB) component. Consequently, in trials where both WADD and Tallying strongly favor Option B, the Hybrid model is mathematically forced to predict Option B (subject only to lapse). In contrast, the Competing theory assumes decision-makers probabilistically sample from a repertoire that explicitly includes TTB. By designing 'Trap Trials' where the single highest-validity cue points to Option A but all other cues point to Option B (making both Tallying and WADD favor B), the Competing theory can predict a substantial proportion of Option A choices (reflecting the base-rate of TTB), whereas the Advocated theory cannot. Additional trials orthogonally pit Tallying against WADD to ensure all mixing parameters can be identified.",
  "validities": [
    0.95,
    0.85,
    0.65,
    0.55,
    0.5
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
      1
    ],
    [
      1,
      0,
      1,
      0,
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
      0,
      1,
      1,
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
      1,
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
      0,
      1,
      0,
      1,
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
  "prompt_token_count": 4032,
  "candidates_token_count": 575,
  "total_token_count": 7284
}
```
