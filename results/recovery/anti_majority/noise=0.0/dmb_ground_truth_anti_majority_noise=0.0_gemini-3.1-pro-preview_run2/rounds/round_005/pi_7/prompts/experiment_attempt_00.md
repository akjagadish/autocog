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
**Description:** Sequential Evidence Accumulation with Lateral Inhibition, Leak, and a Collapsing Decision Threshold: Decision-makers sample features sequentially with probability proportional to their validities. Options accumulate evidence subject to leak and lateral inhibition. To prevent indecision when evidence is mixed or tied, the decision threshold collapses over time (urgency signal). A highly valid cue can quickly cross the initially high threshold (mimicking Take-The-Best), but if early cues conflict, the threshold gradually lowers, forcing a decision based on the integrated evidence of later sampled cues (mimicking Tallying or WADD).

**Parameters:**
- gamma: [0.1, 20.0]
- theta: [1.0, 15.0]
- inhibition: [0.0, 1.0]
- leak: [0.0, 1.0]
- collapse_rate: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    inhibition = float(parameters["inhibition"])
    leak = float(parameters["leak"])
    collapse_rate = float(parameters["collapse_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Sampling probability based on validity
    w = val ** gamma
    if np.sum(w) == 0:
        p_sample = np.ones_like(w) / len(w)
    else:
        p_sample = w / np.sum(w)
        
    n_sims = 2000
    max_steps = 100
    
    # Pre-sample all cues for all sims and steps for speed
    samples = np.random.choice(len(val), size=(n_sims, max_steps), p=p_sample)
    
    inc_a = a[samples]
    inc_b = b[samples]
    
    ea = np.zeros(n_sims)
    eb = np.zeros(n_sims)
    
    decided = np.zeros(n_sims, dtype=bool)
    winner = np.full(n_sims, -1)
    
    for step in range(max_steps):
        mask = ~decided
        if not np.any(mask):
            break
            
        curr_inc_a = inc_a[mask, step]
        curr_inc_b = inc_b[mask, step]
        
        # LCA update rule with lateral inhibition and leak
        new_ea = np.maximum(0.0, ea[mask] * (1.0 - leak) + curr_inc_a - inhibition * eb[mask])
        new_eb = np.maximum(0.0, eb[mask] * (1.0 - leak) + curr_inc_b - inhibition * ea[mask])
        
        ea[mask] = new_ea
        eb[mask] = new_eb
        
        # Collapsing threshold
        curr_theta = max(0.01, theta - step * collapse_rate)
        cross_a = new_ea >= curr_theta
        cross_b = new_eb >= curr_theta
        
        just_decided = cross_a | cross_b
        
        if np.any(just_decided):
            jd_indices = np.where(just_decided)[0]
            
            for idx in jd_indices:
                if cross_a[idx] and not cross_b[idx]:
                    winner_val = 0
                elif cross_b[idx] and not cross_a[idx]:
                    winner_val = 1
                else:
                    if new_ea[idx] > new_eb[idx]:
                        winner_val = 0
                    elif new_eb[idx] > new_ea[idx]:
                        winner_val = 1
                    else:
                        winner_val = np.random.choice([0, 1])
                
                orig_idx = np.where(mask)[0][idx]
                winner[orig_idx] = winner_val
                decided[orig_idx] = True

    undecided = ~decided
    if np.any(undecided):
        ea_un = ea[undecided]
        eb_un = eb[undecided]
        
        # For those that haven't crossed threshold, highest evidence wins
        ties = ea_un == eb_un
        win_un = np.where(ea_un > eb_un, 0, 1)
        if np.any(ties):
            win_un[ties] = np.random.choice([0, 1], size=ties.sum())
            
        winner[undecided] = win_un
        
    p_A = np.sum(winner == 0) / n_sims
    p_B = np.sum(winner == 1) / n_sims
    
    p = np.array([p_A, p_B])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] To dissociate Take The Best (TTB) from Tallying, we use a 4-feature design where the most valid cue often points to one option, while the sheer number of winning cues points to the other. TTB will choose the option favored by the single highest-validity discriminating cue, whereas Tallying will choose the option favored by the majority of discriminating cues, regardless of their validities. We also include trials where Tallying predicts a tie but TTB strongly prefers one option.

[1] To robustly dissociate Tallying from Take The Best (TTB), we use 5 features with descending validities. TTB will always choose the option favored by the first discriminating feature (the one with the highest validity). Tallying, however, simply counts the number of features favoring each option, regardless of validity. By pitting a single high-validity feature against multiple lower-validity features, we create clear adversarial trials: TTB strongly prefers the option with the single best feature, while Tallying prefers the option with the majority of winning features. We also include trials where Tallying predicts a tie (equal number of winning features) but TTB makes a deterministic choice, further differentiating the two models.

[2] To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 4-feature design with a steep drop-off in validities after the first cue, but closely spaced validities for the remaining cues (e.g., 0.95, 0.75, 0.65, 0.55). TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, on the other hand, integrates all cues. If WADD's gamma parameter is low, it will behave compensatorily and favor options with multiple lower-validity cues over an option with a single high-validity cue. We construct trials that pit the highest-validity discriminating cue against a coalition of multiple lower-validity cues. For instance, an option favored by only the top cue will be chosen by TTB, whereas WADD (with low-to-moderate gamma) will choose the alternative favored by the three remaining cues. We also include trials where the highest cue is tied, shifting the TTB decision to the second cue, which is then pitted against the third and fourth cues to provide a robust test across different levels of the cue hierarchy.

[3] To robustly dissociate the Weighted Additive (WADD) theory from Take The Best (TTB), we use a 5-feature design with one highly valid cue and several moderately valid cues. TTB relies strictly on the highest-validity discriminating cue, ignoring all others. WADD, however, integrates all cues, allowing a coalition of lower-validity cues to compensate for the lack of the highest-validity cue. We construct a series of adversarial trials where the top discriminating cue points to Option A, but the majority of remaining cues point to Option B. Across these trials, TTB consistently chooses Option A, while WADD (assuming a moderate gamma parameter) will consistently choose Option B due to the overwhelming additive evidence of the lesser cues. By varying which cues tie and which discriminate, we test the compensatory nature of WADD against the non-compensatory lexicographic stopping rule of TTB at different levels of the cue hierarchy.

[4] To dissociate Strategy Mixture Theory from the Generalized WADD theory, we exploit how each model produces intermediate choice probabilities. Strategy Mixture Theory assumes a probabilistic mixture of pure TTB and pure WADD, predicting relatively constant intermediate choice proportions (e.g., matching the mixing parameter alpha) across any trial where TTB and WADD strongly conflict, regardless of the exact margin of the WADD sum. In contrast, Generalized WADD achieves intermediate probabilities either via high noise (low beta) or by precisely tuning the gamma parameter to make the exponentiated sums of the two options nearly equal. By presenting multiple distinct conflict trials with varying differences in raw feature sums, Generalized WADD cannot simultaneously equate the sums for all trials with a single gamma parameter. Consequently, Generalized WADD will predict extreme probabilities on some conflict trials and intermediate on others, whereas Strategy Mixture Theory predicts consistent intermediate probabilities across all such conflict trials.

[5] To quantitatively dissociate Generalized WADD from Strategy Mixture Theory, we manipulate the margin by which the compensatory WADD strategy opposes the non-compensatory TTB strategy. Strategy Mixture Theory assumes choices are a stable probabilistic coin-flip between pure TTB and pure WADD. Therefore, on any trial where TTB strongly favors Option A and WADD strongly favors Option B, Strategy Mixture predicts a relatively constant intermediate choice probability driven by the mixing parameter 'alpha'. In contrast, Generalized WADD integrates all features non-linearly. A single 'gamma' parameter cannot flatten the varying evidence margins across different trials. By presenting a spectrum of conflicts—ranging from the top cue being opposed by all four remaining cues, to being opposed by only two—Generalized WADD is forced to predict a graded shift in probabilities, whereas Strategy Mixture predicts a step-function or constant mixture across these conflicts.

[6] This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting their fundamentally different mechanisms for scaling evidence. WADD scales validities via exponentiation (gamma), which relies on the ratio between validities. Because the ratio between adjacent validities increases as validities get smaller (e.g., 0.75/0.65 > 0.85/0.75), WADD naturally becomes *more* non-compensatory (TTB-like) at lower levels of the cue hierarchy. In contrast, SEA uses an absolute difference threshold (theta) on unscaled validities. Because the absolute values of the validities decrease down the hierarchy, SEA naturally becomes *less* non-compensatory at lower levels, as single cues are no longer large enough to cross the fixed threshold. By pitting a single discriminating cue against the remaining lower cues at different levels of the hierarchy (e.g., Cue 2 vs Cues 3-5, and Cue 3 vs Cues 4-5), SEA predicts non-compensatory choices at the top of the hierarchy and compensatory choices at the bottom. WADD predicts the exact opposite pattern, making them highly distinguishable.

[7] This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting the bounded nature of SEA's stopping threshold (theta max = 1.25). In Trials 1 and 2, one option is favored by the top two most valid cues, while the other is favored by the remaining four. Because SEA accumulates evidence sequentially, the first two cues generate a running difference (0.85 + 0.80 = 1.65) that strictly exceeds the maximum possible threshold. Consequently, SEA is forced to stop and choose the option with the top two cues across 100% of valid parameter combinations, completely ignoring the remaining four cues. In contrast, WADD integrates all available information. Because the linear sum of the bottom four cues (2.70) is vastly greater than the top two (1.65), WADD strongly favors the option with the bottom four cues when its non-linear scaling parameter (gamma) is low, and smoothly shifts to favoring the top two cues as gamma increases. Trials 3 and 4 serve as controls where the running difference fluctuates, allowing SEA to also predict a shift in choices across its parameter range.

[8] This design quantitatively dissociates Dual-Process Strategy Selection Theory from Generalized WADD by holding the top discriminating cue constant across a sequence of trials while systematically increasing the number of opposing lower-validity cues. Dual-Process assumes the probability of employing the non-compensatory Take-The-Best (TTB) strategy depends solely on the absolute validity of the top discriminating cue. Because the top cue is identical across the first four trials, the probability of executing TTB remains fixed. As the compensatory WADD fallback strategy increasingly favors the opposing option, the overall choice probability for the top-cue option will drop but then plateau at the exact probability of TTB selection (e.g., an intermediate asymptote like 60%). In contrast, Generalized WADD integrates all features; adding more opposing cues continuously increases the evidence sum for the alternative option, driving the probability of choosing the top-cue option progressively toward zero. A single gamma scaling parameter in WADD cannot produce a stable intermediate plateau across these escalating conflicts.

[9] This design quantitatively dissociates Generalized WADD from Dual-Process Strategy Selection by pitting constant absolute WADD evidence differences against decreasing top-cue validities. Across trials, the top discriminating cue shifts down the hierarchy (from Cue 1 to Cue 2 to Cue 3), while the linear WADD sum difference favoring the opposing option remains strictly constant. Dual-Process Strategy Selection relies on the absolute validity of the top discriminating cue to probabilistically trigger the non-compensatory Take-The-Best (TTB) strategy; therefore, it predicts choices will become less TTB-like (more compensatory) as the top cue moves down the hierarchy and its validity drops. In contrast, Generalized WADD scales validities via exponentiation, which depends on the ratio between validities. Because the ratio between adjacent validities increases down the hierarchy (e.g., 0.75/0.65 > 0.95/0.85), Generalized WADD predicts decisions will inherently become MORE TTB-like at lower levels of the hierarchy. These diametrically opposed predictions provide a strong quantitative dissociation.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — heuristics that rely on validity-ordering re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) collapse WADD into a scaled Equal-Weight rule and make them indistinguishable by any decision.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended heuristics (TTB, EQW, Tallying, WADD) be dissociated; avoid trials where every heuristic agrees.


## Response

```json
{
  "rationale": "This design quantitatively dissociates Sequential Evidence Accumulation (SEA) from the Weighted Additive (WADD) theory by exploiting SEA's lateral inhibition and collapsing threshold. In SEA, if the most valid cues agree (Trial 1), evidence accumulates rapidly, crossing the threshold early and producing a non-compensatory choice that ignores lower-validity cues. However, if the top cues conflict (Trials 2 and 3), they mutually inhibit each other. This prevents early threshold crossing, forces the threshold to collapse over time, and causes the decision to be driven compensatorily by the remaining lower-validity cues. WADD, lacking a sequential stopping rule or dynamic inhibition, applies a single non-linear scaling parameter (gamma) across all trials. WADD cannot simultaneously behave in a highly non-compensatory manner when top cues agree and a highly compensatory manner when top cues conflict, whereas SEA naturally predicts this sudden strategy shift.",
  "validities": [
    0.95,
    0.9,
    0.7,
    0.65,
    0.6
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
      1,
      1,
      0
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
  "prompt_token_count": 5024,
  "candidates_token_count": 404,
  "total_token_count": 7212
}
```
