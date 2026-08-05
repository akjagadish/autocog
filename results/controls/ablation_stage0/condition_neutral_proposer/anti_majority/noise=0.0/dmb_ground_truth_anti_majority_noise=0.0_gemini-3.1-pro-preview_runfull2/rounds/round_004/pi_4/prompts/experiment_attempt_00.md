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
**Description:** Rank-Dependent Tallying: Decision-makers primarily evaluate options based on the sheer number of positive features (a tallying-like process), but the weight of each feature is subject to rank-based discounting. Rather than scaling exponentially with log-odds, a feature's weight decays as a power law of its validity rank. This ensures a strong compensatory mechanism where multiple moderate cues can easily overpower a single high-validity cue, and tallying differences dominate choice probabilities unless the validity rank differences are extreme.

**Parameters:**
- gamma: [2.0, 8.0]
- beta: [0.1, 20.0]
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
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Assign ranks to features based on validity (1 = highest validity)
    n_features = len(val)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros(n_features)
    ranks[order] = np.arange(1, n_features + 1)
    
    # Rank-based discounting: weight decays as an inverse power of rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted tally for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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

[6] To distinguish Theory 1 (Strategy Mixture: TTB + Linear WADD) from Theory 2 (Cue Difference Threshold: TTB with a fallback to Log-Odds WADD), we exploit the non-linear nature of log-odds compared to raw validities. We use a set of validities with one extreme high value (0.99) and several moderate values. Because log-odds grow asymptotically as validity approaches 1.0, the single 0.99 cue has a massive log-odds weight (approx 4.60) compared to the moderate cues, but a small linear weight (0.99). In critical trials, we pit the 0.99 cue against multiple moderate cues such that the linear sum strongly favors the moderate cues, but the log-odds sum still favors the single 0.99 cue. Theory 2 will either stick to TTB or fall back to log-odds WADD—both of which favor the 0.99 cue, leading to a deterministic choice. Theory 1, however, will mix TTB (favoring the 0.99 cue) and Linear WADD (favoring the moderate cues), predicting a mixed response rate. We also vary the top cue difference and the number of opposing cues to trigger Theory 2's threshold and deficit limit fallback conditions differentially.

[7] To cleanly dissociate Theory 1 (Cue Difference Threshold with log-odds WADD fallback) from Theory 2 (Mixture of TTB and linear WADD), we exploit the difference between log-odds weighting and linear weighting, as well as the threshold-triggered fallback mechanism. We use five features with a wide spread of validities. Because log-odds grow non-linearly, the highest validity cue (0.95) has a massive log-odds weight compared to moderate cues, but a relatively small linear weight advantage. In trials where the 0.95 cue is pitted against several moderate cues (e.g., 0.75, 0.65, 0.55), the linear sum strongly favors the moderate cues (Linear WADD), while the log-odds sum still favors the 0.95 cue. Theory 1 will either stick to TTB or fall back to log-odds WADD—both of which favor the single highest cue, resulting in a deterministic choice. Theory 2, however, mixes TTB (favoring the top cue) and Linear WADD (favoring the multiple moderate cues), predicting a mixed response. By varying the difference between the top cues and the number of opposing cues, we can also map out the threshold and deficit limit parameters of Theory 1.

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
  "rationale": "Theory 1 predicts choices as a probabilistic mixture of Take-The-Best (TTB) and a Linear Weighted Additive rule (WADD). Because Linear WADD uses raw validities as weights, it is strongly compensatory: the sum of several moderate validities easily exceeds a single high validity. Theory 2, on the other hand, uses rank-dependent tallying where the weight of a feature decays as an inverse power of its rank (gamma >= 2.0). Mathematically, for any gamma >= 2.0, the weight of the highest-ranked feature (1.0) is strictly greater than the sum of all subsequent feature weights combined (which maxes out at ~0.645). Thus, Theory 2 acts as a deterministic, strictly non-compensatory lexicographic model (essentially a pure TTB rule). By pitting the highest-validity discriminating cue against multiple lower-validity cues, we can cleanly separate the two theories: Theory 2 will uniformly choose the option favored by the single best cue, while Theory 1 will exhibit a mixed response profile reflecting its compensatory WADD component.",
  "validities": [
    0.95,
    0.85,
    0.8,
    0.75,
    0.7
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
      1
    ],
    [
      0,
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
      1,
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
      1,
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
    ]
  ]
}
```

## Usage

```json
{
  "prompt_token_count": 4221,
  "candidates_token_count": 650,
  "total_token_count": 6995
}
```
