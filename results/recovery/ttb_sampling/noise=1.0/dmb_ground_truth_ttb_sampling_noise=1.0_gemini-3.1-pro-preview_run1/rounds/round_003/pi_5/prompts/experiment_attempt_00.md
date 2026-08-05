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
**Description:** Sequential Evidence Accumulation Theory: Decision-makers sample cues sequentially in order of validity and accumulate evidence (counts of features favoring each option). If the difference in accumulated evidence reaches an internal threshold, the search is terminated and a choice is made based on the evidence collected up to that point. If all cues are exhausted without reaching the threshold, a choice is made based on the final accumulated tallies. This single mechanistic process naturally interpolates between Take-The-Best behavior (low threshold) and Tallying behavior (high threshold), while a softmax over the naturally accumulated evidence accounts for the observed stochasticity.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- threshold: [1.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sample cues in order of validity
    order = np.argsort(validities)[::-1]
    
    threshold = float(parameters["threshold"])
    
    ev_A = 0.0
    ev_B = 0.0
    
    for idx in order:
        if a[idx] > b[idx]:
            ev_A += 1.0
        elif b[idx] > a[idx]:
            ev_B += 1.0
            
        # Check if the evidence difference reached the threshold
        if ev_A - ev_B >= threshold:
            break
        elif ev_B - ev_A >= threshold:
            break
            
    scores = np.array([ev_A, ev_B])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over accumulated evidence
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.4, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design quantitatively dissociates Tallying (which simply counts the number of features favoring each option) from Weighted Additive (WADD) (which integrates the cardinal validity of each feature). The validities are chosen such that the sum of the three least valid features (0.6 + 0.55 + 0.5 = 1.65) is less than the sum of the two most valid features (0.9 + 0.85 = 1.75). This creates a critical dissociation: Tallying will strongly favor an option that wins on the three minor features (3 wins vs 2 wins), whereas WADD will favor the option that wins on the two major features. Additional trials include cases where Tallying predicts a tie but WADD has a strong preference, and cases where both models agree, ensuring a comprehensive test space.

[1] This design strictly dissociates WADD from Tallying by manipulating the sum of validities versus the raw count of features. The validities are heavily skewed: the top two features have very high validities (0.95, 0.90), while the remaining three have low validities (0.55, 0.52, 0.50). In critical trials, one option wins on the two highly valid features (thus preferred by WADD, sum = 1.85), while the other option wins on the three less valid features (thus preferred by Tallying, 3 wins vs 2 wins). Additional trials include cases where both heuristics predict a tie or agree, providing a comprehensive test of whether subjects integrate cardinal weights or simply count winning features.

[2] To quantitatively dissociate Take-The-Best (TTB) from Weighted Additive (WADD), we exploit TTB's strict non-compensatory lexicographic stopping rule. By configuring the validities such that the most valid cue is outweighed by the sum of multiple lower-validity cues, we create critical trials where the two models make opposite predictions. For instance, if Option A wins on the highest validity cue but Option B wins on all remaining cues, TTB will deterministically choose A (ignoring the rest of the features), whereas WADD will integrate all features and strongly prefer B. The trials include a mix of these compensatory vs. non-compensatory conflicts at various cue validities, as well as control trials where both models agree.

[3] To quantitatively dissociate the Weighted Additive (WADD) rule from the Take-The-Best (TTB) heuristic, we exploit TTB's strict non-compensatory lexicographic stopping rule. We configure the validities such that the most valid cues can be outweighed by the sum of several lower-validity cues. In critical trials, one option wins on the highest-validity cue (thus TTB deterministically chooses it, ignoring all other features), whereas the other option wins on multiple remaining cues (causing WADD to integrate the features and prefer the latter option). By including trials where the conflict happens at the first cue, the second cue, and the third cue, we can robustly distinguish whether subjects use a lexicographic search or a compensatory weighted integration.

[4] This design quantitatively dissociates the Strategy Mixture Theory (which posits intra-individual trial-by-trial mixing of Take-The-Best and Tallying) from the Weighted Additive (WADD) rule. The validities are structured with two highly valid cues and three less valid cues. We construct critical trials where TTB and Tallying point in opposite directions (e.g., Option A wins on the highest validity cue, but Option B wins on a greater number of less valid cues). Because Strategy Mixture Theory assumes individuals mix TTB and Tallying (with p_ttb ~ 0.5), it predicts a choice proportion near 0.5 for these conflict trials. In contrast, WADD integrates all cardinal validities and will strongly favor one option depending on the exact sum of weights, driving choice probabilities toward 1 or 0. By varying which option WADD favors while keeping the TTB/Tallying conflict constant, we can perfectly dissociate the two models.

[5] This design specifically targets the Strategy Mixture model, which posits that subjects randomly alternate between Take-The-Best (TTB) and Tallying. By pitting TTB against Tallying across a set of trials, the Strategy Mixture model predicts choice proportions near 0.50 (assuming a typical mixing probability around 0.50). However, the Weighted Additive (WADD) model integrates all cues according to their validities. By carefully selecting the validities and feature configurations, we can force WADD to strongly prefer Option A in some of these conflict trials and Option B in others, whereas Strategy Mixture remains stuck at predicting ~0.50 across all of them.

[6] This design quantitatively dissociates Strategy Mixture Theory from Sequential Evidence Accumulation (SEA) Theory by exploiting their different mechanisms for handling conflict between the most valid cue and the sheer number of cues. Strategy Mixture predicts a trial-by-trial coin flip (choice probability ~0.5) whenever Take-The-Best (TTB) and Tallying disagree, regardless of the size of the tallying majority. SEA, however, deterministically follows a threshold: a low threshold mimics TTB (always choosing the option favored by the best cue), while a higher threshold allows evidence to accumulate, mimicking Tallying. By varying the point at which the tallying majority overcomes the best cue across trials, SEA is forced to strongly favor one option or the other depending on its fitted threshold, whereas Strategy Mixture consistently predicts ~0.5. Control trials where both heuristics agree ensure that baseline accuracy is maintained, preventing SEA from fitting a near-zero beta to mimic the 0.5 predictions.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Sequential Evidence Accumulation (SEA) Theory from Strategy Mixture Theory. Strategy Mixture Theory predicts a choice proportion near 0.50 whenever Take-The-Best (TTB) and Tallying conflict, because it mixes these two strategies trial-by-trial. SEA, on the other hand, accumulates evidence sequentially and stops when a threshold is reached. By varying the sequence in which evidence arrives across different conflict trials (e.g., Option A wins the first cue, but Option B wins the next three vs. Option A wins the first two cues, but Option B wins the next three), we can manipulate whether SEA's threshold is triggered early (favoring A) or late/never (favoring B). This causes SEA to make deterministic, varying predictions across different conflict trials depending on the threshold parameter, while Strategy Mixture remains rigidly stuck at predicting ~0.50 for all of them.",
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
      1,
      1,
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
      0,
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
      1,
      0,
      1,
      1,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 3790,
  "candidates_token_count": 472,
  "total_token_count": 5410
}
```
