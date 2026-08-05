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
**Description:** Strategy Selection (Mixed Heuristics): Decision-makers probabilistically sample between a non-compensatory heuristic (Take The Best) and a simple compensatory heuristic (Tallying) on a trial-by-trial basis. The probability of using Tallying increases with the ease of the choice, defined by the absolute difference in the number of features favoring each option. By restricting the sensitivity parameter of this mixture, decision-makers preserve a baseline probability of using TTB even when Tallying discriminates, matching empirical reliance on dominant cues while pulling highly conflicting trials toward chance.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) score
    ttb_score = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying (Equal-Weights) score
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    if a_wins > b_wins:
        tally_score = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_score = np.array([0.0, 1.0])
    else:
        tally_score = np.array([0.5, 0.5])
        
    # Difficulty defined by tally difference
    diff = abs(a_wins - b_wins)
    
    # Probability of using Tallying over TTB
    gamma = float(parameters["gamma"])
    w_tally = 1.0 - np.exp(-gamma * diff)
    w_ttb = 1.0 - w_tally
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * ttb_score
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * tally_score
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of probabilities
    p_core = w_ttb * p_ttb + w_tally * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 1.0]
- threshold: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] To distinguish Take The Best (TTB) from Tallying, we need trials where the option with the highest-validity discriminating cue is different from the option with the greatest total number of winning cues. In trials 1 and 3, TTB strongly prefers the option that wins on the most valid cue, whereas Tallying strongly prefers the other option because it wins on three lower-validity cues. In trials 2 and 4, Tallying predicts a tie (guessing) because both options win on an equal number of cues, whereas TTB has a strong preference based on the single most valid cue. Trials 5 and 6 test similar principles but with the highest validity cue tied, forcing TTB to look at the second-best cue.

[1] This design pits Take The Best (TTB) against Tallying using 5 features with strictly decreasing validities. In trials 1-6, the option that wins on the single highest-validity discriminating cue (favored by TTB) loses on the majority of the remaining lower-validity cues (causing Tallying to prefer the opposite option). In trials 7-8, the two options tie in the total number of winning cues (leading Tallying to guess uniformly), but one option has the highest-validity discriminating cue (leading TTB to have a strong preference). This ensures a strong quantitative dissociation across multiple trial types.

[2] To quantitatively dissociate Take The Best (TTB) from the Weighted Additive Model (WADD), this design exploits the compensatory nature of WADD versus the non-compensatory nature of TTB. We use five features with validities set such that the log-odds weight of the highest-validity cue is less than the sum of the log-odds weights of several lower-validity cues. In most of these trials, one option possesses the single highest discriminating validity cue (leading TTB to strictly prefer it), while the other option possesses a combination of lower-validity cues whose combined log-odds evidence outweighs the single best cue (leading WADD to strictly prefer the opposite option). We also include trials where the top cue is tied, forcing TTB to the second cue while WADD integrates the remaining evidence.

[3] To maximally distinguish the Weighted Additive Model (WADD) from Take The Best (TTB), this design uses a 5-cue structure where the validities are set such that the highest-validity cue is highly predictive but its log-odds weight is outweighed by the sum of several lower-validity cues. In all trials, Option A possesses the single best discriminating cue, causing TTB to strictly prefer Option A. However, Option B possesses a combination of lower-validity cues that collectively provide stronger evidence in terms of log-odds, causing WADD to strictly prefer Option B. By varying which cues are tied and which lower-validity cues Option B has, we ensure robust quantitative dissociation across different levels of cue conflict.

[4] To maximally distinguish Theory 1 (Strategy Selection: mixture of Take The Best and Tallying) from Theory 2 (Weighted Additive Model), this design uses a specific set of validities where the top cue is strong, but the next two cues combined strongly outweigh it in log-odds, while the bottom two cues have very low log-odds weights. In key trials (e.g., Trial 1), Option A possesses the single best cue and the two worst cues, giving it a Tallying advantage (3 winning cues to 2) and the TTB advantage. Theory 1 will thus unequivocally prefer Option A, regardless of its mixture parameter. However, Option B possesses the second and third best cues, whose combined log-odds weight heavily exceeds Option A's total, causing Theory 2 (WADD) to unequivocally prefer Option B. Other trials vary the tally difference and cue distribution to further map out the mixture dynamics of Theory 1 against WADD's compensatory integration.

[5] To maximally distinguish Theory 1 (Weighted Additive Model) from Theory 2 (Strategy Selection: mixture of TTB and Tallying), this experiment uses a 5-feature design with a specific validity profile: one highly valid cue, two moderately high ones, and two low ones. In Trial 1, Option A possesses the highest valid cue and the two lowest valid cues, winning on TTB and winning on Tallying (3 to 2). Theory 2 will unequivocally prefer Option A. However, Option B possesses the second and third best cues, whose combined log-odds weight strongly outweighs Option A's cues, causing Theory 1 (WADD) to unequivocally prefer Option B. Other trials systematically vary the tally difference (which controls Theory 2's mixture probability) to create further dissociations where WADD's compensatory integration diverges from the TTB/Tallying mixture.

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
  "rationale": "This design quantitatively dissociates Theory 1 (Strategy Selection: mixture of TTB and Tallying) from Theory 2 (Sequential Evidence Accumulation). By manipulating the absolute difference in tally scores (diff) alongside the evidence for the Weighted Additive rule (WADD), we create unique divergence points. For example, when diff=0, Theory 1 reduces strictly to Take The Best (TTB). In Trial 1, diff=0 so Theory 1 unequivocally chooses Option B (which has the highest valid cue). However, the remaining cues are structured such that Option A wins on WADD. Thus, Theory 2 will choose Option B if its threshold is low, but will switch to Option A if its threshold is high. Other trials systematically vary 'diff' to map the mixture probability in Theory 1 against the threshold-dependent stopping in Theory 2.",
  "validities": [
    0.85,
    0.8,
    0.75,
    0.6,
    0.55
  ],
  "trial_a_ratings": [
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
      1,
      1,
      1,
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
      1
    ],
    [
      1,
      0,
      1,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
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
      0
    ],
    [
      0,
      1,
      1,
      0,
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
  "prompt_token_count": 3623,
  "candidates_token_count": 460,
  "total_token_count": 9072
}
```
