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
**Description:** Probabilistic Cue Integration with Attention Lapses and Choice Inertia: Subjects integrate evidence from cues by probabilistically attending to them based on validity. To account for behavioral overdispersion and sequence effects, the final choice probability is a mixture of the evidence-based decision, random lapses, and explicit choice inertia (stickiness) from the previous trial.

**Parameters:**
- attention_base: [0.1, 1.0]
- attention_gamma: [-5.0, 10.0]
- history_bias: [-10.0, 10.0]
- temperature: [0.01, 10.0]
- epsilon: [0.0, 1.0]
- stickiness: [0.0, 0.99]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    attention_base = float(parameters["attention_base"])
    attention_gamma = float(parameters["attention_gamma"])
    history_bias = float(parameters["history_bias"])
    temperature = float(parameters["temperature"])
    epsilon = float(parameters["epsilon"])
    stickiness = float(parameters["stickiness"])
    
    # Probability of attending to each cue depends on base rate and its validity
    p_attend = np.clip(attention_base * (val ** attention_gamma), 0.0, 1.0)
    
    prev_resp = None
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
    
    bias = 0.0
    p_prev = np.array([0.5, 0.5])
    if prev_resp == 0:
        bias = history_bias
        p_prev = np.array([1.0, 0.0])
    elif prev_resp == 1:
        bias = -history_bias
        p_prev = np.array([0.0, 1.0])
        
    a, b = stim[0], stim[1]
    n_features = len(val)
    
    # Monte Carlo simulation of probabilistic attention
    n_sims = 2000
    attend_mask = np.random.rand(n_sims, n_features) < p_attend
    evidence = weights * (a - b)
    
    # Sum evidence only for attended cues and add history bias
    total_evidence = np.sum(attend_mask * evidence, axis=1) + bias
    
    # Softmax conversion to probability of choosing A
    z = total_evidence / temperature
    z = np.clip(z, -100, 100)
    p_A_sims = 1.0 / (1.0 + np.exp(-z))
    
    p_A = np.mean(p_A_sims)
    p_B = 1.0 - p_A
    
    p_core = np.array([p_A, p_B])
    
    # Normalize weights to ensure valid probabilities even if stickiness + epsilon > 1
    w_core = max(0.0, 1.0 - stickiness - epsilon)
    total_w = w_core + stickiness + epsilon
    
    return (w_core * p_core + stickiness * p_prev + epsilon * np.array([0.5, 0.5])) / total_w
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
**Description:** Noisy Evidence Accumulation with History Bias and Leakage: Decision making is a noisy sequential sampling process where evidence from cues is accumulated in order of validity. The accumulation is subject to moment-to-moment noise and evidence decay (leakage), producing trial-to-trial variability and primacy/recency effects. The starting point of evidence accumulation is biased by the previous choice, naturally capturing sequence-aware divergence. Wide individual differences in noise, threshold, leak, and history bias account for the high variance in behavior across subjects.

**Parameters:**
- threshold: [0.01, 10.0]
- noise_std: [0.1, 5.0]
- history_bias: [-5.0, 5.0]
- epsilon: [0.0, 0.5]
- leak: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    noise_std = float(parameters["noise_std"])
    history_bias = float(parameters["history_bias"])
    epsilon = float(parameters["epsilon"])
    leak = float(parameters["leak"])
    
    # Determine previous response to set accumulation starting bias
    prev_resp = None
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
    
    bias = 0.0
    if prev_resp == 0:
        bias = history_bias
    elif prev_resp == 1:
        bias = -history_bias
        
    a, b = stim[0], stim[1]
    
    # Vectorized Monte Carlo simulation of the noisy accumulation process
    n_sims = 1000
    accumulated = np.full(n_sims, bias)
    decided = np.zeros(n_sims, dtype=bool)
    choices = np.zeros(n_sims, dtype=int)
    
    for j in cue_order:
        mu = weights[j] * (a[j] - b[j])
        # Moment-to-moment noise in evidence accumulation
        step = np.random.normal(mu, noise_std, size=n_sims)
        
        # Apply leakage to previously accumulated evidence before adding the new step
        accumulated = np.where(decided, accumulated, accumulated * (1.0 - leak) + step)
        
        hit_A = (accumulated >= threshold) & ~decided
        hit_B = (accumulated <= -threshold) & ~decided
        
        choices[hit_A] = 0
        choices[hit_B] = 1
        
        decided = decided | hit_A | hit_B
        if np.all(decided):
            break
            
    # For simulations that didn't cross the threshold, decide based on final accumulated evidence
    if not np.all(decided):
        undecided = ~decided
        undecided_A = undecided & (accumulated > 0)
        undecided_B = undecided & (accumulated < 0)
        undecided_tie = undecided & (accumulated == 0)
        
        choices[undecided_A] = 0
        choices[undecided_B] = 1
        
        ties = np.sum(undecided_tie)
        if ties > 0:
            choices[undecided_tie] = np.random.choice([0, 1], size=ties)
            
    p_A = np.mean(choices == 0)
    p_B = 1.0 - p_A
    
    p_core = np.array([p_A, p_B])
    n_opts = len(p_core)
    
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To quantitatively dissociate Take The Best (TTB) from Tallying, we use 5 features with strictly descending validities. The trials are designed such that the option favored by the highest-validity discriminating cue (the TTB choice) is often opposed by the option that wins on the simple count of cues (the Tallying choice). Additionally, we include trials where Tallying predicts a perfect tie (guessing) while TTB makes a deterministic prediction based on the single highest-validity discriminating feature.

[1] To strongly dissociate Tallying from Take The Best (TTB), we use 4 features with descending validities. We design trials where the highest-validity discriminating cue favors one option, while the simple count of winning features (Tallying) favors the other option. We also include trials where Tallying predicts a perfect tie (guessing) because both options win on an equal number of features, whereas TTB makes a deterministic prediction based on the single highest-validity discriminating feature. This provides a robust quantitative dissociation between the two models.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use 5 features with validities chosen such that the log-odds weight of any single feature is smaller than the combined log-odds weights of the features ranked below it. In the critical trials, the option favored by the highest-validity discriminating cue (the TTB choice) is opposed by an option that has a greater number of lower-validity cues in its favor. Because WADD integrates all evidence via a weighted sum (compensatory), it predicts that the accumulation of weaker cues will override the single best cue. TTB, being non-compensatory, strictly follows the single highest-validity discriminating cue. We also include agreement trials where the two models make the same choice but with different confidence margins.

[3] To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we configure 5 features with validities such that the log-odds weight of the most valid cue is smaller than the combined log-odds weights of multiple lower-validity cues. In our critical trials, the option favored by the single highest-validity discriminating cue (the TTB choice) is opposed by the option that possesses a greater number of lower-validity cues. Because WADD integrates all evidence via a compensatory weighted sum, it predicts that the accumulation of weaker cues will override the single best cue. TTB, being strictly non-compensatory, will always choose the option favored by the first discriminating cue regardless of the lower-validity cues.

[4] To quantitatively dissociate the sequential evidence accumulation model from the Weighted Additive (WADD) model, we use 5 features with validities designed such that the log-odds weight of the best cue is smaller than the sum of the remaining cues' weights, and similarly for subsequent cues. Critical trials contrast an option favored by the single best available discriminating cue against an option favored by a combination of multiple lower-validity cues. Because WADD strictly integrates all cues, it predicts the option with the higher weighted sum of all features. The advocated sequential model, however, can stop accumulating evidence once an internal threshold is reached. By fitting a threshold, the sequential model can predict choices aligned with early stopping (non-compensatory behavior) that WADD inherently cannot capture, as well as varying degrees of partial integration. The inclusion of agreement trials ensures robust estimation of baseline noise and threshold parameters.

[5] To demonstrate that decision-making relies on compensatory weighted addition (WADD) rather than sequential early-stopping, we construct trial pairs where the difference in evidence from the most valid cues strongly favors Option A, but the evidence from lower-validity cues systematically varies in support of Option B. A sequential model with a finite threshold predicts that once the evidence difference threshold is crossed by the top cues, the remaining lower-validity cues are ignored, resulting in identical choice probabilities across these variations (a step-function response). WADD, however, predicts that every cue continuously contributes to the final weighted sum, producing a smooth, graded shift in choice probabilities. By observing sensitivity to the lowest-validity cues even when the top cues provide strong evidence, we can rule out the threshold-based stopping rule and support continuous integration.

[6] To quantitatively dissociate the deterministic sequential threshold model (Advocated) from the noisy accumulation model with leakage (Competing), we configure 6 features with descending validities. We design critical trials that pit early (high-validity) evidence against late (lower-validity) evidence. In the Advocated model, evidence is perfectly retained; thus, a single top cue can either trigger an early decision (if the threshold is low) or be evaluated against the exact mathematical sum of late cues (if the threshold is high). In the Competing model, 'leakage' causes early evidence to decay over time, creating a recency effect where a cluster of late cues will disproportionately overpower a single early cue. By systematically varying the temporal position of the opposing cues while keeping the total log-odds evidence roughly balanced, we can observe whether choices exhibit the step-function/exact-sum behavior predicted by the Advocated model or the recency bias predicted by the Competing model's leakage.

[7] To quantitatively dissociate the Noisy Evidence Accumulation model with Leakage from the deterministic Threshold model, we use 5 features with descending validities. We design critical trials that pit early (high-validity) evidence against late (lower-validity) evidence. In the Competing model, evidence is perfectly retained; thus, a single top cue can either trigger an early decision (if the threshold is low) or be evaluated against the exact mathematical sum of late cues (if the threshold is high). In the Advocated model, 'leakage' causes early evidence to decay over time, creating a recency effect where a cluster of late cues will disproportionately overpower a single early cue. By systematically varying the temporal position of the opposing cues while keeping the total log-odds evidence roughly balanced, we can observe whether choices exhibit the step-function/exact-sum behavior predicted by the Competing model or the recency bias predicted by the Advocated model's leakage.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "The Advocated theory posits probabilistic, independent attention to cues based on their validity, without a strict processing order or early stopping. The Competing theory posits sequential processing strictly ordered by validity, with early stopping (thresholds) and evidence leakage. By pitting a single very high-validity cue against multiple lower-validity cues, the Competing theory predicts either early stopping (always choosing the high-validity cue) or strong recency effects due to leakage (if the threshold is high, the early high-validity cue decays and the later lower-validity cues dominate). The Advocated theory predicts a smoother integration where the high-validity cue is attended with high probability, but occasionally missed, producing a specific mixture of choices that does not depend on sequential decay but rather on independent attention lapses.",
  "validities": [
    0.9,
    0.8,
    0.7,
    0.6,
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
      0,
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
      1,
      0
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
      1,
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
      0
    ],
    [
      1,
      0,
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
  "prompt_token_count": 4520,
  "candidates_token_count": 510,
  "total_token_count": 6530
}
```
