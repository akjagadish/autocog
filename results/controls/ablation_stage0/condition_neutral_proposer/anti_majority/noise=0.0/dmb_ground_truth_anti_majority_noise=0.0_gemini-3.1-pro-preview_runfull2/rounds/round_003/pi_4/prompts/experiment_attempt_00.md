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
**Description:** Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

**Parameters:**
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## THEORY 2
**Description:** Cue Difference Threshold Theory: Decision-makers evaluate options lexicographically but demand that the best discriminating cue provides a decisive advantage. A cue is deemed decisive if its validity exceeds the best opposing cue by a sufficient threshold, or if the sheer number of opposing cues is small enough (below a tallying deficit limit). If the top cue's advantage is challenged by a concentrated block of moderately high opposing cues (failing both conditions), the decision-maker abandons the non-compensatory heuristic and falls back to a compensatory Weighted Additive (WADD) process to resolve the choice.

**Parameters:**
- threshold: [0.0, 1.0]
- deficit_limit: {0, 1, 2, 3, 4, 5}
- beta: [0.1, 25.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    deficit_limit = int(parameters["deficit_limit"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues favor which option
    favor_a = (a > b)
    favor_b = (b > a)
    
    max_val_a = np.max(val[favor_a]) if np.any(favor_a) else 0.0
    max_val_b = np.max(val[favor_b]) if np.any(favor_b) else 0.0
    
    winner_ttb = 0 if max_val_a > max_val_b else (1 if max_val_b > max_val_a else None)
    
    if winner_ttb is not None:
        # Number of cues favoring the TTB loser
        num_opposing = np.sum(favor_b) if winner_ttb == 0 else np.sum(favor_a)
        
        # Two conditions for TTB to be considered decisive:
        # 1. The validity difference between the best cues of each option is large enough.
        # 2. The number of opposing cues is within the acceptable deficit limit.
        cue_diff_decisive = abs(max_val_a - max_val_b) >= threshold
        deficit_ok = num_opposing <= deficit_limit
        
        if cue_diff_decisive or deficit_ok:
            # Decisive advantage: stick to TTB
            scores = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        else:
            # Not decisive: fall back to compensatory WADD
            val_clipped = np.clip(val, 0.5001, 0.9999)
            log_odds = np.log(val_clipped / (1.0 - val_clipped))
            score_a = np.sum(log_odds * a)
            score_b = np.sum(log_odds * b)
            
            # Normalize compensatory scores by total weight to keep the scale 
            # comparable to the [0, 1] TTB scores for the softmax temperature.
            total_w = np.sum(log_odds)
            scores = np.array([score_a, score_b]) / total_w if total_w > 0 else np.array([0.5, 0.5])
    else:
        scores = np.array([0.5, 0.5])
        
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
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
[0] To cleanly dissociate Take The Best (TTB) from Tallying, we use 5 features with strictly descending validities. In each trial, one option is favored by the single most valid discriminating cue (which TTB relies on exclusively), while the other option is favored by a greater number of less valid cues (which Tallying simply counts). Thus, TTB and Tallying make opposite predictions on every trial.

[1] To cleanly dissociate Tallying from Take The Best (TTB), we use 5 features with strictly descending validities. In each trial, one option is favored by the single most valid discriminating cue (which TTB relies on exclusively), while the other option is favored by a greater number of less valid cues (which Tallying simply counts). We also include variations where the first cue is tied, forcing TTB to rely on the second cue, while Tallying still favors the opposite option based on the remaining cues. Thus, TTB and Tallying make opposite predictions on every trial.

[2] To distinguish Take The Best (TTB) from Weighted Additive (WADD) with non-linear weight scaling, we use four features with varying validities. The trials are designed to contrast the single-cue reliance of TTB against the compensatory integration of WADD. In several trials, the option favored by the single most valid discriminating cue is opposed by multiple less valid cues favoring the other option. By varying the number and validity of the opposing cues, we can identify whether participants evaluate cues sequentially and stop at the first discriminator (TTB) or integrate them compensatorily (WADD). Furthermore, the varying margins of compensation provide a gradient that can estimate the non-linear scaling parameter gamma in the WADD model.

[3] To distinguish the Weighted Additive (WADD) model with non-linear weight scaling from the lexicographic Take The Best (TTB) model, we use a 5-feature design with a steep drop-off in validities. TTB evaluates cues strictly sequentially and stops at the first discriminator, ignoring all other cues. WADD integrates all cues, but its non-linear scaling parameter (gamma) determines whether it behaves more like TTB (gamma > 1) or more like Tallying (gamma < 1). By pitting the single highest-validity cue against varying numbers of lower-validity cues, we create compensatory trials where WADD (with low-to-moderate gamma) will choose the option favored by the sum of lower-validity cues, while TTB will stubbornly choose the option favored by the single best cue. Varying the threshold of compensation across trials provides a fine-grained measure to estimate WADD's gamma parameter and clearly separate it from TTB's strict one-reason decision making.

[4] To maximally distinguish Theory 1 (Mixture of TTB and linear WADD) from Theory 2 (Non-linear WADD), we exploit a fundamental difference in how they weight cues. Theory 1's WADD component uses raw validities as linear weights, while Theory 2 transforms validities into log-odds before applying a non-linear scaling parameter (gamma). Because raw validities are bounded between 0.5 and 1.0, their sum grows quickly, making linear WADD strongly favor options with more positive cues (Tallying-like behavior). In contrast, log-odds can be heavily skewed (e.g., a 0.92 validity has a much higher log-odds than a 0.72 validity), meaning Theory 2 can easily favor the single best cue even without a strict TTB mechanism, especially when gamma > 1. By pitting the highest-validity cue against various combinations of lower-validity cues, we create trials where the raw validity sum strongly opposes the highest cue, but the log-odds sum (depending on gamma) may still favor it. The pattern of choices across these specific trade-offs will differentiate a probabilistic mixture of step-function TTB and linear WADD from a deterministic, non-linearly scaled log-odds WADD.

[5] To maximally distinguish Theory 1 (Non-linear log-odds WADD) from Theory 2 (Mixture of TTB and Linear WADD), we exploit the fundamental difference in how they weight cues. Theory 2's WADD component uses raw validities as linear weights, which sums up quickly and strongly favors options with more positive cues (similar to Tallying). Theory 1 transforms validities into log-odds before applying a non-linear scaling parameter (gamma). Because log-odds grow asymptotically as validity approaches 1.0, Theory 1 can strongly favor a single highly valid cue over multiple moderately valid cues, even without a strict TTB mechanism. By creating trials where the linear sum of raw validities strongly opposes the highest cue, but the log-odds sum supports it (or vice-versa depending on the exact combination), we can cleanly separate the probabilistic mixture of step-function TTB and linear WADD from the deterministic, non-linearly scaled log-odds WADD.

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
  "rationale": "To distinguish Theory 1 (Strategy Mixture: TTB + Linear WADD) from Theory 2 (Cue Difference Threshold: TTB with a fallback to Log-Odds WADD), we exploit the non-linear nature of log-odds compared to raw validities. We use a set of validities with one extreme high value (0.99) and several moderate values. Because log-odds grow asymptotically as validity approaches 1.0, the single 0.99 cue has a massive log-odds weight (approx 4.60) compared to the moderate cues, but a small linear weight (0.99). In critical trials, we pit the 0.99 cue against multiple moderate cues such that the linear sum strongly favors the moderate cues, but the log-odds sum still favors the single 0.99 cue. Theory 2 will either stick to TTB or fall back to log-odds WADD\u2014both of which favor the 0.99 cue, leading to a deterministic choice. Theory 1, however, will mix TTB (favoring the 0.99 cue) and Linear WADD (favoring the moderate cues), predicting a mixed response rate. We also vary the top cue difference and the number of opposing cues to trigger Theory 2's threshold and deficit limit fallback conditions differentially.",
  "validities": [
    0.99,
    0.85,
    0.8,
    0.75,
    0.6
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
      1
    ],
    [
      0,
      0,
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
      0,
      0,
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
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4108,
  "candidates_token_count": 699,
  "total_token_count": 7443
}
```
