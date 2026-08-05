# feedback_iter_08

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real (human / ground-truth) data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_12`
- The recommendation below acts on THEORY 2 (= `pi_12`).

Propose a 'Weighted Additive with Non-linear Validity Scaling' (WADD-NL) theory. Instead of categorical strategy switching (e.g., TTB vs Tallying vs Reverse TTB), decision-makers integrate all available cues but weight them non-linearly based on their stated validities. For instance, the subjective weight of a cue could be an exponential or power function of its validity, allowing the model to naturally transition between TTB-like behavior (when the non-linearity is high, heavily favoring the best cue) and Tallying-like behavior (when the non-linearity is low), without hard thresholds or tied-sum triggers. This continuous integration approach will likely provide more robust and gradated choice probabilities across all experimental designs.


## CANDIDATE THEORY
Weighted Additive with Power-Function Validity Scaling (WADD-PWR)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


`parameters`:
- gamma: [-10.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's advice, we return to the unnormalized power function base (`w = val ** gamma`) from iteration 6, which successfully captured both TTB and Reverse TTB behaviors. To fix the issue of extreme deterministic predictions without resorting to failed normalizations or temperature expansions, we narrow the range of gamma to [-10.0, 10.0]. This prevents the absolute scale of the evidence from exploding when gamma is large and negative (e.g., 0.5 ** -20 = 1,048,576), ensuring the beta scaling remains calibrated across all experimental designs while preserving the continuous WADD-NL integration mechanism.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.5013 -> ACCEPTED
- iter 2: loss=0.4408 -> ACCEPTED
- iter 3: loss=0.5058 -> REJECTED
- iter 4: loss=0.4807 -> REJECTED
- iter 5: loss=0.5055 -> REJECTED
- iter 6: loss=0.4312 -> ACCEPTED
- iter 7: loss=0.4677 -> REJECTED
- iter 8: loss=0.4623 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.4646 -> REJECTED
Running-best (last accepted) base: iter 6 at loss=0.4312.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        
        tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else -1)
        
        # Only consider trials where TTB and Tallying make opposing deterministic predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            ttb_matches.append(1 if row['response'] == ttb_winner else 0)
            
    if not ttb_matches:
        return 0.5
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.6508 (var=0.0505)
**Candidate trajectory (this loop):**
  - iter 1: 0.7154 (var=0.0395) (Δ vs real +0.0646)
  - iter 2: 0.4100 (var=0.0862) (Δ vs real -0.2408)
  - iter 3: 0.3237 (var=0.0801) (Δ vs real -0.3271)
  - iter 4: 0.3513 (var=0.1107) (Δ vs real -0.2996)
  - iter 5: 0.3775 (var=0.0748) (Δ vs real -0.2733)
  - iter 6: 0.3425 (var=0.0822) (Δ vs real -0.3083)
  - iter 7: 0.3867 (var=0.0903) (Δ vs real -0.2642)
  - iter 8: 0.4475 (var=0.1284) (Δ vs real -0.2033)
  - iter 9 (current): 0.2617 (var=0.0695) (Δ vs real -0.3892)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8471 (var=0.0094)
- pi_2: 0.1842 (var=0.0150)
- pi_3: 0.6208 (var=0.0346)
- pi_4: 0.6571 (var=0.0250)
- pi_5: 0.6525 (var=0.0167)
- pi_6: 0.5054 (var=0.0097)
- pi_7: 0.7004 (var=0.0332)
- pi_8: 0.7383 (var=0.0261)
- pi_9: 0.4642 (var=0.0870)
- pi_10: 0.6300 (var=0.0278)
- pi_11: 0.3767 (var=0.0225)
- pi_12: 0.7571 (var=0.0819)

### Experiment 2
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    matches = (tally_preds[valid_mask] == responses[valid_mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3011 (var=0.0238)
**Candidate trajectory (this loop):**
  - iter 1: 0.2708 (var=0.0161) (Δ vs real -0.0303)
  - iter 2: 0.5094 (var=0.0883) (Δ vs real +0.2083)
  - iter 3: 0.6086 (var=0.0973) (Δ vs real +0.3075)
  - iter 4: 0.6086 (var=0.0676) (Δ vs real +0.3075)
  - iter 5: 0.6500 (var=0.0576) (Δ vs real +0.3489)
  - iter 6: 0.5642 (var=0.0617) (Δ vs real +0.2631)
  - iter 7: 0.5503 (var=0.0648) (Δ vs real +0.2492)
  - iter 8: 0.6439 (var=0.0716) (Δ vs real +0.3428)
  - iter 9 (current): 0.5942 (var=0.0578) (Δ vs real +0.2931)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8328 (var=0.0095)
- pi_1: 0.1311 (var=0.0070)
- pi_3: 0.3628 (var=0.0340)
- pi_4: 0.3628 (var=0.0203)
- pi_5: 0.3386 (var=0.0210)
- pi_6: 0.4697 (var=0.0089)
- pi_7: 0.2425 (var=0.0259)
- pi_8: 0.2467 (var=0.0310)
- pi_9: 0.5097 (var=0.0656)
- pi_10: 0.3900 (var=0.0317)
- pi_11: 0.6008 (var=0.0199)
- pi_12: 0.2317 (var=0.0698)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        pred = None
        # The validities are [0.95, 0.93, 0.91, 0.89, 0.5], so the cue order is simply 0 to 4.
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
                
        if pred is not None:
            agreements.append(1 if resp == pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.6100 (var=0.0044)
**Candidate trajectory (this loop):**
  - iter 1: 0.5183 (var=0.0472) (Δ vs real -0.0917)
  - iter 2: 0.3981 (var=0.0088) (Δ vs real -0.2119)
  - iter 3: 0.3458 (var=0.0114) (Δ vs real -0.2642)
  - iter 4: 0.3856 (var=0.0057) (Δ vs real -0.2244)
  - iter 5: 0.3506 (var=0.0121) (Δ vs real -0.2594)
  - iter 6: 0.3679 (var=0.0070) (Δ vs real -0.2421)
  - iter 7: 0.4110 (var=0.0098) (Δ vs real -0.1990)
  - iter 8: 0.3633 (var=0.0056) (Δ vs real -0.2467)
  - iter 9 (current): 0.3327 (var=0.0059) (Δ vs real -0.2773)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8738 (var=0.0091)
- pi_3: 0.3508 (var=0.0099)
- pi_2: 0.3267 (var=0.0043)
- pi_4: 0.6810 (var=0.0113)
- pi_5: 0.4873 (var=0.0031)
- pi_6: 0.5006 (var=0.0039)
- pi_7: 0.7798 (var=0.0289)
- pi_8: 0.6758 (var=0.0197)
- pi_9: 0.5813 (var=0.0331)
- pi_10: 0.5865 (var=0.0337)
- pi_11: 0.3890 (var=0.0102)
- pi_12: 0.8083 (var=0.0358)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    # TTB consults cues in order of validity (which corresponds to the feature index 0 to 4).
    # We can find the TTB choice by weighting the differences such that earlier features strictly dominate.
    diff = A - B
    weights = 10 ** np.arange(A.shape[1])[::-1]
    ttb_score = diff.dot(weights)
    
    # If ttb_score > 0, A is favored on the first discriminating cue (predict 0).
    # If ttb_score < 0, B is favored (predict 1).
    ttb_pred = (ttb_score < 0).astype(int)
    
    return float(np.mean(data['response'].values == ttb_pred))
```

**Observed (real) value:** 0.6383 (var=0.0300)
**Candidate trajectory (this loop):**
  - iter 1: 0.6594 (var=0.0069) (Δ vs real +0.0210)
  - iter 2: 0.3921 (var=0.0753) (Δ vs real -0.2462)
  - iter 3: 0.3117 (var=0.0919) (Δ vs real -0.3267)
  - iter 4: 0.3187 (var=0.0624) (Δ vs real -0.3196)
  - iter 5: 0.2852 (var=0.0458) (Δ vs real -0.3531)
  - iter 6: 0.3558 (var=0.0783) (Δ vs real -0.2825)
  - iter 7: 0.3873 (var=0.0518) (Δ vs real -0.2510)
  - iter 8: 0.3829 (var=0.0990) (Δ vs real -0.2554)
  - iter 9 (current): 0.3115 (var=0.0713) (Δ vs real -0.3269)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6185 (var=0.0157)
- pi_1: 0.8521 (var=0.0087)
- pi_2: 0.1623 (var=0.0094)
- pi_4: 0.7048 (var=0.0188)
- pi_5: 0.6348 (var=0.0233)
- pi_6: 0.5340 (var=0.0043)
- pi_7: 0.6963 (var=0.0213)
- pi_8: 0.7631 (var=0.0297)
- pi_9: 0.4179 (var=0.0528)
- pi_10: 0.5962 (var=0.0228)
- pi_11: 0.4073 (var=0.0196)
- pi_12: 0.7473 (var=0.0796)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (data['A_tuple'] == (1,0,1,0,0)) & (data['B_tuple'] == (0,1,0,1,1))
    t3_mask = (data['A_tuple'] == (1,0,0,0,1)) & (data['B_tuple'] == (0,1,1,1,0))
    t4_mask = (data['A_tuple'] == (1,1,0,0,0)) & (data['B_tuple'] == (1,0,1,1,0))
    t5_mask = (data['A_tuple'] == (1,1,0,0,0)) & (data['B_tuple'] == (1,0,0,1,1))
    
    p_A_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_A_t3 = 1.0 - data[t3_mask]['response'].mean()
    p_A_t4 = 1.0 - data[t4_mask]['response'].mean()
    p_A_t5 = 1.0 - data[t5_mask]['response'].mean()
    
    val = (p_A_t1 - p_A_t3) + (p_A_t5 - p_A_t4)
    
    if pd.isna(val):
        return 0.0
    return float(val)
```

**Observed (real) value:** 0.0825 (var=0.1837)
**Candidate trajectory (this loop):**
  - iter 1: 0.0900 (var=0.1106) (Δ vs real +0.0075)
  - iter 2: -0.1675 (var=0.3132) (Δ vs real -0.2500)
  - iter 3: 0.0388 (var=0.2136) (Δ vs real -0.0438)
  - iter 4: -0.1838 (var=0.2483) (Δ vs real -0.2663)
  - iter 5: 0.1400 (var=0.2923) (Δ vs real +0.0575)
  - iter 6: -0.1425 (var=0.2913) (Δ vs real -0.2250)
  - iter 7: -0.4025 (var=0.2925) (Δ vs real -0.4850)
  - iter 8: -0.1338 (var=0.3636) (Δ vs real -0.2163)
  - iter 9 (current): 0.0050 (var=0.4029) (Δ vs real -0.0775)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0050 (var=0.0408)
- pi_3: 0.2650 (var=0.1685)
- pi_1: 0.0012 (var=0.0293)
- pi_2: -0.0100 (var=0.0237)
- pi_5: 0.1613 (var=0.0413)
- pi_6: 0.0163 (var=0.0737)
- pi_7: 0.0187 (var=0.0499)
- pi_8: 0.0313 (var=0.0402)
- pi_9: -0.2188 (var=0.2266)
- pi_10: 0.1050 (var=0.0954)
- pi_11: 0.0825 (var=0.0662)
- pi_12: -0.1137 (var=0.1323)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'A_str': a_str,
        'response': data['response']
    })
    
    # Strategy Mixture strictly predicts identical probabilities for Trials 1 & 7, and Trials 2 & 8.
    # T1 & T7: TTB predicts Option A, Tallying predicts Option A.
    # T2 & T8: TTB predicts Option B, Tallying predicts Option B.
    # WADD with non-linear scaling strongly differentiates these pairs based on specific cue validities.
    pairs = [
        ('1000111', '1010101'), # T1 vs T7
        ('0101010', '0111000')  # T8 vs T2
    ]
    
    scores = []
    for subj, grp in df.groupby('subject_id'):
        subj_score = 0
        for s_a, s_b in pairs:
            ra = grp[grp['A_str'] == s_a]['response'].values
            rb = grp[grp['A_str'] == s_b]['response'].values
            if len(ra) >= 2 and len(rb) >= 2:
                # Split-half cross-product provides an unbiased estimator of the squared difference
                # in true choice probabilities. Under Strategy Mixture, expected value is exactly 0.
                # Under WADD, the expected value is strictly positive.
                ra_even, ra_odd = ra[::2].mean(), ra[1::2].mean()
                rb_even, rb_odd = rb[::2].mean(), rb[1::2].mean()
                subj_score += (ra_even - rb_even) * (ra_odd - rb_odd)
        scores.append(subj_score)
        
    return float(np.mean(scores))
```

**Observed (real) value:** -0.0167 (var=0.0028)
**Candidate trajectory (this loop):**
  - iter 1: 0.5039 (var=0.4113) (Δ vs real +0.5206)
  - iter 2: 0.4500 (var=0.3520) (Δ vs real +0.4667)
  - iter 3: 0.5406 (var=0.5163) (Δ vs real +0.5572)
  - iter 4: 0.4450 (var=0.3048) (Δ vs real +0.4617)
  - iter 5: 0.4094 (var=0.3568) (Δ vs real +0.4261)
  - iter 6: 0.3111 (var=0.2476) (Δ vs real +0.3278)
  - iter 7: 0.3989 (var=0.2973) (Δ vs real +0.4156)
  - iter 8: 0.6278 (var=0.4031) (Δ vs real +0.6444)
  - iter 9 (current): 0.3089 (var=0.2650) (Δ vs real +0.3256)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9578 (var=0.3256)
- pi_4: 0.0056 (var=0.0017)
- pi_1: -0.0100 (var=0.0035)
- pi_2: 0.0028 (var=0.0039)
- pi_5: 0.0578 (var=0.0224)
- pi_6: 0.0361 (var=0.0279)
- pi_7: 0.0539 (var=0.0772)
- pi_8: -0.0100 (var=0.0058)
- pi_9: 0.0172 (var=0.0087)
- pi_10: 0.1400 (var=0.1147)
- pi_11: 0.0306 (var=0.0254)
- pi_12: 0.0089 (var=0.0012)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_trial_type(a):
        a_tuple = tuple(a)
        if a_tuple == (1, 1, 0, 0, 1): return 1
        if a_tuple == (1, 0, 0, 1, 1): return 3
        if a_tuple == (1, 0, 0, 0, 1): return 4
        if a_tuple == (1, 0, 1, 0, 0): return 6
        return 0
        
    trial_types = data['option_a_ratings'].apply(get_trial_type)
    
    p_A = {}
    for t in [1, 3, 4, 6]:
        mask = trial_types == t
        if mask.sum() > 0:
            p_A[t] = np.mean(data.loc[mask, 'response'] == 0)
        else:
            p_A[t] = 0.5
            
    return float((p_A[1] - p_A[3]) + (p_A[6] - p_A[4]))
```

**Observed (real) value:** -0.2050 (var=0.2002)
**Candidate trajectory (this loop):**
  - iter 1: 0.1225 (var=0.1255) (Δ vs real +0.3275)
  - iter 2: -0.1988 (var=0.2307) (Δ vs real +0.0062)
  - iter 3: -0.0888 (var=0.2277) (Δ vs real +0.1162)
  - iter 4: -0.2087 (var=0.1957) (Δ vs real -0.0037)
  - iter 5: -0.0263 (var=0.1945) (Δ vs real +0.1787)
  - iter 6: -0.3187 (var=0.2500) (Δ vs real -0.1137)
  - iter 7: -0.2838 (var=0.2669) (Δ vs real -0.0788)
  - iter 8: -0.2213 (var=0.2233) (Δ vs real -0.0163)
  - iter 9 (current): -0.0325 (var=0.3099) (Δ vs real +0.1725)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0162 (var=0.0422)
- pi_5: 0.1437 (var=0.0461)
- pi_1: -0.0125 (var=0.0342)
- pi_2: 0.0000 (var=0.0255)
- pi_3: 0.2900 (var=0.1053)
- pi_6: 0.0575 (var=0.0679)
- pi_7: 0.0563 (var=0.0375)
- pi_8: -0.0062 (var=0.0454)
- pi_9: -0.1388 (var=0.1600)
- pi_10: 0.1387 (var=0.0677)
- pi_11: 0.0763 (var=0.0632)
- pi_12: -0.0225 (var=0.0382)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    # Identify trials by their sum of cues (Tallying score proxy)
    sumA = A.sum(axis=1)
    sumB = B.sum(axis=1)
    
    # 1. Trials where one option has strictly more cues (Trials 4, 5, 8)
    mask_more_B = (sumB > sumA)
    mask_more_A = (sumA > sumB)
    
    more_cues_chosen = 0
    more_cues_total = 0
    if np.any(mask_more_B):
        more_cues_chosen += np.sum(resp[mask_more_B] == 1)
        more_cues_total += np.sum(mask_more_B)
    if np.any(mask_more_A):
        more_cues_chosen += np.sum(resp[mask_more_A] == 0)
        more_cues_total += np.sum(mask_more_A)
        
    p_more_cues = float(more_cues_chosen) / more_cues_total if more_cues_total > 0 else 0.5
    
    # 2. Trials where options have an equal number of cues (Trials 1, 2, 3, 6, 7)
    mask_equal = (sumA == sumB)
    
    ttb_winner_chosen = 0
    ttb_total = 0
    if np.any(mask_equal):
        # Cue 0 is the highest validity cue. In equal cue trials, 
        # the option with Cue 0 is always the TTB winner.
        mask_ttb_A = mask_equal & (A[:, 0] == 1)
        ttb_winner_chosen += np.sum(resp[mask_ttb_A] == 0)
        ttb_total += np.sum(mask_ttb_A)
        
        mask_ttb_B = mask_equal & (B[:, 0] == 1)
        ttb_winner_chosen += np.sum(resp[mask_ttb_B] == 1)
        ttb_total += np.sum(mask_ttb_B)
        
    p_ttb_winner = float(ttb_winner_chosen) / ttb_total if ttb_total > 0 else 0.5
    
    # The metric is a linear combination designed to cancel out the p_ttb parameter in the Mixture model
    return float(p_more_cues + 2.0 * p_ttb_winner)

```

**Observed (real) value:** 0.9324 (var=0.1377)
**Candidate trajectory (this loop):**
  - iter 1: 1.8696 (var=0.0322) (Δ vs real +0.9371)
  - iter 2: 1.7688 (var=0.1072) (Δ vs real +0.8363)
  - iter 3: 1.8687 (var=0.1106) (Δ vs real +0.9362)
  - iter 4: 1.7830 (var=0.0644) (Δ vs real +0.8506)
  - iter 5: 1.7392 (var=0.0766) (Δ vs real +0.8068)
  - iter 6: 1.4457 (var=0.1679) (Δ vs real +0.5132)
  - iter 7: 1.8079 (var=0.0595) (Δ vs real +0.8754)
  - iter 8: 1.7130 (var=0.2744) (Δ vs real +0.7806)
  - iter 9 (current): 1.5436 (var=0.2414) (Δ vs real +0.6111)
**Other theories' values on this metric (for reference):**
- pi_5: 1.7010 (var=0.0202)
- pi_4: 1.8784 (var=0.0226)
- pi_1: 1.8606 (var=0.0150)
- pi_2: 1.9030 (var=0.0303)
- pi_3: 1.9046 (var=0.0565)
- pi_6: 1.5333 (var=0.0268)
- pi_7: 1.7891 (var=0.0225)
- pi_8: 1.3323 (var=0.0562)
- pi_9: 1.6521 (var=0.0553)
- pi_10: 1.6938 (var=0.0342)
- pi_11: 1.7516 (var=0.1198)
- pi_12: 1.3804 (var=0.0691)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    
    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t3_mask = a_ratings == (1, 1, 0, 0, 0)
    # Trial 4: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 1]
    t4_mask = a_ratings == (0, 1, 1, 0, 0)
    
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t3) or pd.isna(p_a_t4):
        return 0.0
        
    return float(p_a_t3 + p_a_t4)
```

**Observed (real) value:** 1.6547 (var=0.1361)
**Candidate trajectory (this loop):**
  - iter 1: 1.0168 (var=0.0335) (Δ vs real -0.6379)
  - iter 2: 0.7053 (var=0.1162) (Δ vs real -0.9495)
  - iter 3: 0.6232 (var=0.1496) (Δ vs real -1.0316)
  - iter 4: 0.5747 (var=0.1691) (Δ vs real -1.0800)
  - iter 5: 0.5505 (var=0.1697) (Δ vs real -1.1042)
  - iter 6: 0.7168 (var=0.1727) (Δ vs real -0.9379)
  - iter 7: 0.6495 (var=0.1757) (Δ vs real -1.0053)
  - iter 8: 0.5684 (var=0.1474) (Δ vs real -1.0863)
  - iter 9 (current): 0.5263 (var=0.1689) (Δ vs real -1.1284)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7937 (var=0.0431)
- pi_6: 1.0674 (var=0.0256)
- pi_1: 0.9800 (var=0.0117)
- pi_2: 0.2621 (var=0.0461)
- pi_3: 1.0484 (var=0.0172)
- pi_5: 1.1011 (var=0.0213)
- pi_7: 0.9853 (var=0.0247)
- pi_8: 0.9579 (var=0.0176)
- pi_9: 0.7442 (var=0.1582)
- pi_10: 1.0505 (var=0.0208)
- pi_11: 0.8495 (var=0.1024)
- pi_12: 0.8695 (var=0.0941)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    A_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    B_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    chose_A = 1.0 - data['response']
    
    m12 = ((A_str == '100100') & (B_str == '011000')) | ((A_str == '100110') & (B_str == '011001'))
    m34 = ((A_str == '011000') & (B_str == '100000')) | ((A_str == '011010') & (B_str == '100001'))
    m56 = ((A_str == '100000') & (B_str == '011100')) | ((A_str == '100010') & (B_str == '011101'))
    m78 = ((A_str == '001100') & (B_str == '100000')) | ((A_str == '001110') & (B_str == '100001'))
    
    def get_lo(mask):
        n = mask.sum()
        if n == 0:
            return 0.0
        x = chose_A[mask].sum()
        # Laplace smoothing to avoid log(0)
        p = (x + 0.5) / (n + 1.0)
        return np.log(p / (1.0 - p))
        
    lo12 = get_lo(m12)
    lo34 = get_lo(m34)
    lo56 = get_lo(m56)
    lo78 = get_lo(m78)
    
    # Numerator: Contrast where Mixture is exactly 0, WADD-DR is strictly positive
    num = lo34 - lo78
    # Denominator: Contrast that is positive for both and scales identically with beta
    denom = lo12 - lo56
    
    # Bounded normalized ratio to cancel out the beta variance
    return float(num / (abs(num) + abs(denom) + 0.1))
```

**Observed (real) value:** 0.0885 (var=0.0487)
**Candidate trajectory (this loop):**
  - iter 1: 0.3804 (var=0.1904) (Δ vs real +0.2919)
  - iter 2: -0.0272 (var=0.1011) (Δ vs real -0.1157)
  - iter 3: 0.3163 (var=0.1110) (Δ vs real +0.2278)
  - iter 4: 0.1348 (var=0.1077) (Δ vs real +0.0463)
  - iter 5: 0.1603 (var=0.1334) (Δ vs real +0.0718)
  - iter 6: 0.0892 (var=0.1069) (Δ vs real +0.0007)
  - iter 7: 0.2169 (var=0.1372) (Δ vs real +0.1284)
  - iter 8: 0.0364 (var=0.1440) (Δ vs real -0.0521)
  - iter 9 (current): 0.1764 (var=0.0933) (Δ vs real +0.0879)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3311 (var=0.1888)
- pi_4: -0.0236 (var=0.2122)
- pi_1: 0.2907 (var=0.3186)
- pi_2: -0.0310 (var=0.0885)
- pi_3: 0.3089 (var=0.1732)
- pi_5: 0.5910 (var=0.1844)
- pi_7: -0.4383 (var=0.2361)
- pi_8: -0.0265 (var=0.0856)
- pi_9: 0.1615 (var=0.2406)
- pi_10: 0.1827 (var=0.2096)
- pi_11: 0.1176 (var=0.1914)
- pi_12: -0.1220 (var=0.2496)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    # Conflict trials: TTB prefers A (cue 1), but Tally prefers B (2 vs 3)
    t1 = (1, 0, 0, 0, 1)
    t2 = (1, 0, 0, 1, 0)
    t3 = (1, 0, 1, 0, 0)
    conflict_trials = {t1, t2, t3}
    
    # Agreement trial: TTB prefers A (cue 1), and Tally prefers A (2 vs 1)
    t6 = (1, 1, 0, 0, 0)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        df_conflict = subj_df[subj_df['A_tuple'].isin(conflict_trials)]
        df_agree = subj_df[subj_df['A_tuple'] == t6]
        
        if len(df_conflict) == 0 or len(df_agree) == 0:
            continue
            
        # response = 0 means option A was chosen
        p_a_conflict = 1.0 - df_conflict['response'].mean()
        p_a_agree = 1.0 - df_agree['response'].mean()
        
        subj_diffs.append(p_a_agree - p_a_conflict)
        
    if not subj_diffs:
        return 0.0
        
    return float(np.mean(subj_diffs))
```

**Observed (real) value:** -0.4292 (var=0.0555)
**Candidate trajectory (this loop):**
  - iter 1: 0.0400 (var=0.0227) (Δ vs real +0.4692)
  - iter 2: 0.1658 (var=0.0570) (Δ vs real +0.5950)
  - iter 3: 0.3173 (var=0.0746) (Δ vs real +0.7465)
  - iter 4: 0.1781 (var=0.0563) (Δ vs real +0.6073)
  - iter 5: 0.2838 (var=0.0976) (Δ vs real +0.7131)
  - iter 6: 0.0315 (var=0.1120) (Δ vs real +0.4608)
  - iter 7: 0.1804 (var=0.0527) (Δ vs real +0.6096)
  - iter 8: 0.0154 (var=0.0848) (Δ vs real +0.4446)
  - iter 9 (current): 0.1831 (var=0.1229) (Δ vs real +0.6123)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1927 (var=0.0269)
- pi_7: 0.0235 (var=0.0242)
- pi_1: 0.0077 (var=0.0203)
- pi_2: 0.6223 (var=0.0421)
- pi_3: 0.0885 (var=0.0402)
- pi_5: 0.1638 (var=0.0160)
- pi_6: 0.0323 (var=0.0289)
- pi_8: 0.1088 (var=0.0248)
- pi_9: 0.0923 (var=0.0418)
- pi_10: 0.1050 (var=0.0177)
- pi_11: 0.1827 (var=0.0528)
- pi_12: 0.2415 (var=0.0382)

### Experiment 12
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    t9_mask = (data['A_str'] == '11100') & (data['B_str'] == '00011')
    t10_mask = (data['A_str'] == '11010') & (data['B_str'] == '00101')
    t7_mask = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    t8_mask = (data['A_str'] == '00111') & (data['B_str'] == '10000')
    
    ttb_t9 = 1.0 - data.loc[t9_mask, 'response'].mean() if t9_mask.sum() > 0 else 0.5
    ttb_t10 = 1.0 - data.loc[t10_mask, 'response'].mean() if t10_mask.sum() > 0 else 0.5
    ttb_t7 = 1.0 - data.loc[t7_mask, 'response'].mean() if t7_mask.sum() > 0 else 0.5
    ttb_t8 = data.loc[t8_mask, 'response'].mean() if t8_mask.sum() > 0 else 0.5
    
    agree = (ttb_t9 + ttb_t10) / 2.0
    disagree = (ttb_t7 + ttb_t8) / 2.0
    
    return float(agree - disagree)
```

**Observed (real) value:** -0.6711 (var=0.0499)
**Candidate trajectory (this loop):**
  - iter 1: 0.0333 (var=0.0191) (Δ vs real +0.7044)
  - iter 2: 0.1911 (var=0.0943) (Δ vs real +0.8622)
  - iter 3: 0.3144 (var=0.0855) (Δ vs real +0.9856)
  - iter 4: 0.1544 (var=0.0841) (Δ vs real +0.8256)
  - iter 5: 0.2844 (var=0.0996) (Δ vs real +0.9556)
  - iter 6: 0.1611 (var=0.0951) (Δ vs real +0.8322)
  - iter 7: 0.0978 (var=0.0520) (Δ vs real +0.7689)
  - iter 8: 0.0778 (var=0.0704) (Δ vs real +0.7489)
  - iter 9 (current): 0.2389 (var=0.0868) (Δ vs real +0.9100)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0444 (var=0.0356)
- pi_4: 0.2378 (var=0.0388)
- pi_1: -0.0222 (var=0.0131)
- pi_2: 0.6789 (var=0.0431)
- pi_3: 0.1478 (var=0.0759)
- pi_5: 0.1711 (var=0.0236)
- pi_6: 0.0478 (var=0.0311)
- pi_8: 0.0000 (var=0.0405)
- pi_9: 0.0789 (var=0.0387)
- pi_10: 0.0589 (var=0.0557)
- pi_11: 0.3389 (var=0.0800)
- pi_12: 0.0544 (var=0.0251)

### Experiment 13
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the total number of positive cues is tied
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum == b_sum
    filtered = data[mask]
    
    if len(filtered) == 0:
        return 0.5
    
    # In these tied trials, check if the subject chose the option with the highest-validity cue (cue 0)
    a_cue0 = filtered['option_a_ratings'].apply(lambda x: x[0])
    chose_cue0 = ((a_cue0 == 1) & (filtered['response'] == 0)) | ((a_cue0 == 0) & (filtered['response'] == 1))
    
    return float(chose_cue0.mean())
```

**Observed (real) value:** 0.2644 (var=0.0112)
**Candidate trajectory (this loop):**
  - iter 1: 0.8575 (var=0.0077) (Δ vs real +0.5931)
  - iter 2: 0.5544 (var=0.0975) (Δ vs real +0.2900)
  - iter 3: 0.5181 (var=0.1034) (Δ vs real +0.2536)
  - iter 4: 0.4508 (var=0.0929) (Δ vs real +0.1864)
  - iter 5: 0.5725 (var=0.1094) (Δ vs real +0.3081)
  - iter 6: 0.5019 (var=0.1174) (Δ vs real +0.2375)
  - iter 7: 0.5172 (var=0.0945) (Δ vs real +0.2528)
  - iter 8: 0.5186 (var=0.1560) (Δ vs real +0.2542)
  - iter 9 (current): 0.4853 (var=0.1173) (Δ vs real +0.2208)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4861 (var=0.0031)
- pi_7: 0.7722 (var=0.0129)
- pi_1: 0.8689 (var=0.0074)
- pi_2: 0.5056 (var=0.0030)
- pi_3: 0.8119 (var=0.0117)
- pi_4: 0.7708 (var=0.0129)
- pi_5: 0.7458 (var=0.0159)
- pi_6: 0.5544 (var=0.0097)
- pi_9: 0.5758 (var=0.0727)
- pi_10: 0.7389 (var=0.0217)
- pi_11: 0.6389 (var=0.0203)
- pi_12: 0.2139 (var=0.0556)

### Experiment 14
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive cues for options A and B
    a_sums = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sums = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Isolate trials where both options have the same number of positive cues (Trials 1 and 2)
    mask = a_sums == b_sums
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    
    # In these trials, option A always possesses the most valid cue (cue 0)
    # We calculate the proportion of times the subject chose option A (response == 0)
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.1350 (var=0.0065)
**Candidate trajectory (this loop):**
  - iter 1: 0.8237 (var=0.0178) (Δ vs real +0.6887)
  - iter 2: 0.4608 (var=0.1119) (Δ vs real +0.3258)
  - iter 3: 0.5271 (var=0.1153) (Δ vs real +0.3921)
  - iter 4: 0.6112 (var=0.1127) (Δ vs real +0.4762)
  - iter 5: 0.5292 (var=0.1017) (Δ vs real +0.3942)
  - iter 6: 0.4642 (var=0.1158) (Δ vs real +0.3292)
  - iter 7: 0.5537 (var=0.0884) (Δ vs real +0.4187)
  - iter 8: 0.5208 (var=0.1483) (Δ vs real +0.3858)
  - iter 9 (current): 0.4467 (var=0.1191) (Δ vs real +0.3117)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7721 (var=0.0156)
- pi_8: 0.4925 (var=0.0046)
- pi_1: 0.8529 (var=0.0106)
- pi_2: 0.4979 (var=0.0064)
- pi_3: 0.8146 (var=0.0159)
- pi_4: 0.7725 (var=0.0116)
- pi_5: 0.7383 (var=0.0113)
- pi_6: 0.5767 (var=0.0130)
- pi_9: 0.5746 (var=0.0900)
- pi_10: 0.7417 (var=0.0261)
- pi_11: 0.6408 (var=0.0132)
- pi_12: 0.1625 (var=0.0471)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a2 = data['option_a_ratings'].apply(lambda x: x[2])
    b2 = data['option_b_ratings'].apply(lambda x: x[2])
    
    mask = a2 != b2
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    a2_sub = a2[mask]
    b2_sub = b2[mask]
    
    ttb_pred = (b2_sub > a2_sub).astype(int)
    return float((subset['response'] == ttb_pred).mean())
```

**Observed (real) value:** 0.8031 (var=0.0244)
**Candidate trajectory (this loop):**
  - iter 1: 0.7738 (var=0.0523) (Δ vs real -0.0292)
  - iter 2: 0.4500 (var=0.1176) (Δ vs real -0.3531)
  - iter 3: 0.2969 (var=0.0986) (Δ vs real -0.5062)
  - iter 4: 0.3477 (var=0.1263) (Δ vs real -0.4554)
  - iter 5: 0.2377 (var=0.0687) (Δ vs real -0.5654)
  - iter 6: 0.2769 (var=0.0640) (Δ vs real -0.5262)
  - iter 7: 0.3615 (var=0.1213) (Δ vs real -0.4415)
  - iter 8: 0.3215 (var=0.1222) (Δ vs real -0.4815)
  - iter 9 (current): 0.3092 (var=0.0711) (Δ vs real -0.4938)
**Other theories' values on this metric (for reference):**
- pi_8: 0.7508 (var=0.0329)
- pi_9: 0.1408 (var=0.0120)
- pi_1: 0.8531 (var=0.0125)
- pi_2: 0.1223 (var=0.0100)
- pi_3: 0.6200 (var=0.0461)
- pi_4: 0.6008 (var=0.0357)
- pi_5: 0.6192 (var=0.0356)
- pi_6: 0.5285 (var=0.0127)
- pi_7: 0.7077 (var=0.0356)
- pi_10: 0.5869 (var=0.0497)
- pi_11: 0.3600 (var=0.0224)
- pi_12: 0.7992 (var=0.0527)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Identify trials where the total number of cues is perfectly tied (diff_cues == 0) 
    # and the options are spatially symmetric (outer vs inner cues).
    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 0, 0, 0, 1]
    # Trial 4: A=[1, 1, 0, 0, 0], B=[0, 0, 0, 1, 1]
    mask = data['option_a_ratings'].apply(tuple).isin([(1, 0, 0, 0, 0), (1, 1, 0, 0, 0)])
    df_trial = data[mask]
    if len(df_trial) == 0:
        return 0.0
    
    # For the Competing model, diff_cues == 0 means 100% reliance on Tallying. 
    # Since the sum of cues is equal, Tallying predicts exactly 50/50, so subject means will be ~0.5.
    # For the Advocated model, extreme primacy or recency will drive choices deterministically 
    # towards A or B, so subject means will be near 0.0 or 1.0.
    # Measuring the absolute deviation from 0.5 captures this structural divergence.
    subj_means = df_trial.groupby('subject_id')['response'].mean()
    return float(np.mean(np.abs(subj_means - 0.5)))
```

**Observed (real) value:** 0.2611 (var=0.0294)
**Candidate trajectory (this loop):**
  - iter 1: 0.1168 (var=0.0126) (Δ vs real -0.1442)
  - iter 2: 0.2653 (var=0.0232) (Δ vs real +0.0042)
  - iter 3: 0.2753 (var=0.0162) (Δ vs real +0.0142)
  - iter 4: 0.2358 (var=0.0176) (Δ vs real -0.0253)
  - iter 5: 0.2079 (var=0.0179) (Δ vs real -0.0532)
  - iter 6: 0.2300 (var=0.0245) (Δ vs real -0.0311)
  - iter 7: 0.2495 (var=0.0215) (Δ vs real -0.0116)
  - iter 8: 0.3084 (var=0.0211) (Δ vs real +0.0474)
  - iter 9 (current): 0.2321 (var=0.0184) (Δ vs real -0.0289)
**Other theories' values on this metric (for reference):**
- pi_9: 0.3216 (var=0.0178)
- pi_8: 0.0658 (var=0.0028)
- pi_1: 0.3547 (var=0.0127)
- pi_2: 0.0679 (var=0.0037)
- pi_3: 0.1137 (var=0.0087)
- pi_4: 0.2447 (var=0.0108)
- pi_5: 0.1584 (var=0.0115)
- pi_6: 0.0684 (var=0.0033)
- pi_7: 0.1747 (var=0.0131)
- pi_10: 0.1621 (var=0.0206)
- pi_11: 0.0863 (var=0.0054)
- pi_12: 0.3395 (var=0.0186)

### Experiment 17
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    correct = 0
    total = 0
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if sum(a) == sum(b):
            for i in range(len(a)):
                if a[i] != b[i]:
                    expected = 0 if a[i] > b[i] else 1
                    if resp == expected:
                        correct += 1
                    break
            total += 1
            
    return float(correct / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.2825 (var=0.0159)
**Candidate trajectory (this loop):**
  - iter 1: 0.8158 (var=0.0157) (Δ vs real +0.5333)
  - iter 2: 0.6154 (var=0.0312) (Δ vs real +0.3329)
  - iter 3: 0.6262 (var=0.0355) (Δ vs real +0.3438)
  - iter 4: 0.6458 (var=0.0339) (Δ vs real +0.3633)
  - iter 5: 0.6171 (var=0.0401) (Δ vs real +0.3346)
  - iter 6: 0.5158 (var=0.0443) (Δ vs real +0.2333)
  - iter 7: 0.6446 (var=0.0275) (Δ vs real +0.3621)
  - iter 8: 0.6796 (var=0.0597) (Δ vs real +0.3971)
  - iter 9 (current): 0.5938 (var=0.0419) (Δ vs real +0.3113)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4938 (var=0.0038)
- pi_10: 0.7017 (var=0.0218)
- pi_1: 0.8517 (var=0.0136)
- pi_2: 0.5033 (var=0.0039)
- pi_3: 0.7708 (var=0.0174)
- pi_4: 0.7521 (var=0.0140)
- pi_5: 0.6846 (var=0.0134)
- pi_6: 0.5333 (var=0.0069)
- pi_7: 0.7550 (var=0.0149)
- pi_9: 0.6763 (var=0.0302)
- pi_11: 0.5300 (var=0.0085)
- pi_12: 0.4238 (var=0.0184)

### Experiment 18
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where both options have the same total number of positive cues (zero conflict)
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    zero_diff = (sum_a == sum_b)
    
    subset = data[zero_diff]
    if len(subset) == 0:
        return 0.5
        
    # Identify which option possesses the highest-validity cue (index 0)
    a_has_cue1 = subset['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_has_cue1 = subset['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    # Calculate how often the subject chose the option with the highest-validity cue
    chose_a = (subset['response'] == 0)
    chose_b = (subset['response'] == 1)
    
    chose_highest_validity = (chose_a & a_has_cue1) | (chose_b & b_has_cue1)
    
    return float(chose_highest_validity.mean())
```

**Observed (real) value:** 0.3458 (var=0.0444)
**Candidate trajectory (this loop):**
  - iter 1: 0.8363 (var=0.0164) (Δ vs real +0.4904)
  - iter 2: 0.6362 (var=0.0350) (Δ vs real +0.2904)
  - iter 3: 0.6071 (var=0.0434) (Δ vs real +0.2612)
  - iter 4: 0.7087 (var=0.0343) (Δ vs real +0.3629)
  - iter 5: 0.6450 (var=0.0372) (Δ vs real +0.2992)
  - iter 6: 0.6171 (var=0.0246) (Δ vs real +0.2712)
  - iter 7: 0.6950 (var=0.0289) (Δ vs real +0.3492)
  - iter 8: 0.6088 (var=0.0278) (Δ vs real +0.2629)
  - iter 9 (current): 0.6129 (var=0.0307) (Δ vs real +0.2671)
**Other theories' values on this metric (for reference):**
- pi_10: 0.7100 (var=0.0283)
- pi_8: 0.4883 (var=0.0056)
- pi_1: 0.8488 (var=0.0116)
- pi_2: 0.4983 (var=0.0061)
- pi_3: 0.7408 (var=0.0154)
- pi_4: 0.7654 (var=0.0104)
- pi_5: 0.7238 (var=0.0132)
- pi_6: 0.5546 (var=0.0090)
- pi_7: 0.7325 (var=0.0216)
- pi_9: 0.6946 (var=0.0320)
- pi_11: 0.5208 (var=0.0060)
- pi_12: 0.5196 (var=0.0047)

### Experiment 19
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the total number of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for trials where both options have the same number of positive cues
    # (i.e., diff_cues == 0)
    mask = sum_a == sum_b
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(subset['response'] == 0))

```

**Observed (real) value:** 0.1758 (var=0.0110)
**Candidate trajectory (this loop):**
  - iter 1: 0.8247 (var=0.0152) (Δ vs real +0.6489)
  - iter 2: 0.5979 (var=0.0665) (Δ vs real +0.4221)
  - iter 3: 0.6005 (var=0.0741) (Δ vs real +0.4247)
  - iter 4: 0.5542 (var=0.0780) (Δ vs real +0.3784)
  - iter 5: 0.5268 (var=0.0769) (Δ vs real +0.3511)
  - iter 6: 0.5442 (var=0.1227) (Δ vs real +0.3684)
  - iter 7: 0.5779 (var=0.0629) (Δ vs real +0.4021)
  - iter 8: 0.4405 (var=0.1455) (Δ vs real +0.2647)
  - iter 9 (current): 0.5353 (var=0.1138) (Δ vs real +0.3595)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4742 (var=0.0058)
- pi_11: 0.5853 (var=0.0135)
- pi_1: 0.8453 (var=0.0113)
- pi_2: 0.5105 (var=0.0066)
- pi_3: 0.7842 (var=0.0210)
- pi_4: 0.7505 (var=0.0134)
- pi_5: 0.6884 (var=0.0161)
- pi_6: 0.5205 (var=0.0062)
- pi_7: 0.7732 (var=0.0187)
- pi_9: 0.6700 (var=0.0544)
- pi_10: 0.6905 (var=0.0233)
- pi_12: 0.2053 (var=0.0502)

### Experiment 20
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    # Trial 3: A=[1,0,0,0,0] (sum=1), B=[0,1,1,1,1] (sum=4)
    mask_t3 = (sum_a == 1) & (sum_b == 4)
    # Trial 5: A=[1,1,0,0,0] (sum=2), B=[0,0,1,1,1] (sum=3)
    mask_t5 = (sum_a == 2) & (sum_b == 3)
    
    if not np.any(mask_t3) or not np.any(mask_t5):
        return 0.0
        
    responses = data['response'].values
    
    # Probability of choosing Option B in Trial 3 and Trial 5
    p_b_t3 = np.mean(responses[mask_t3] == 1)
    p_b_t5 = np.mean(responses[mask_t5] == 1)
    
    # Return the difference in probability of choosing B between Trial 5 and Trial 3
    return float(p_b_t5 - p_b_t3)
```

**Observed (real) value:** 0.2025 (var=0.0829)
**Candidate trajectory (this loop):**
  - iter 1: -0.0900 (var=0.0505) (Δ vs real -0.2925)
  - iter 2: -0.0787 (var=0.0643) (Δ vs real -0.2812)
  - iter 3: -0.2212 (var=0.1346) (Δ vs real -0.4237)
  - iter 4: -0.0887 (var=0.0652) (Δ vs real -0.2913)
  - iter 5: -0.1450 (var=0.0854) (Δ vs real -0.3475)
  - iter 6: -0.1388 (var=0.0493) (Δ vs real -0.3413)
  - iter 7: -0.0337 (var=0.0314) (Δ vs real -0.2362)
  - iter 8: -0.1012 (var=0.0900) (Δ vs real -0.3037)
  - iter 9 (current): -0.1875 (var=0.0894) (Δ vs real -0.3900)
**Other theories' values on this metric (for reference):**
- pi_11: -0.1675 (var=0.0405)
- pi_8: 0.0612 (var=0.0277)
- pi_1: -0.0125 (var=0.0145)
- pi_2: -0.0387 (var=0.0200)
- pi_3: -0.2275 (var=0.0912)
- pi_4: -0.0463 (var=0.0270)
- pi_5: -0.1750 (var=0.0239)
- pi_6: -0.0600 (var=0.0261)
- pi_7: -0.0337 (var=0.0453)
- pi_9: -0.0887 (var=0.0481)
- pi_10: -0.1213 (var=0.0588)
- pi_12: -0.0038 (var=0.0152)

### Experiment 21
**Design**
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of positive cues for A and B
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Filter for zero-conflict (tied sum) trials
    tied_mask = sum_a == sum_b
    
    if not tied_mask.any():
        return 0.5
        
    tied_data = data[tied_mask]
    
    # The Competing theory predicts a boost to Reverse TTB on tied trials.
    # In the experimental design, Option A always wins the lowest-validity 
    # cue on the tied trials (Trial 1 and Trial 2).
    # The Advocated theory predicts exactly 50/50 on these trials.
    # We return the proportion of times Option A is chosen (response == 0).
    return float(np.mean(tied_data['response'] == 0))
```

**Observed (real) value:** 0.4050 (var=0.0313)
**Candidate trajectory (this loop):**
  - iter 1: 0.4913 (var=0.0044) (Δ vs real +0.0862)
  - iter 2: 0.5950 (var=0.0484) (Δ vs real +0.1900)
  - iter 3: 0.5719 (var=0.0504) (Δ vs real +0.1669)
  - iter 4: 0.6256 (var=0.0413) (Δ vs real +0.2206)
  - iter 5: 0.6325 (var=0.0451) (Δ vs real +0.2275)
  - iter 6: 0.6294 (var=0.0582) (Δ vs real +0.2244)
  - iter 7: 0.6081 (var=0.0585) (Δ vs real +0.2031)
  - iter 8: 0.6863 (var=0.0572) (Δ vs real +0.2812)
  - iter 9 (current): 0.6144 (var=0.0575) (Δ vs real +0.2094)
**Other theories' values on this metric (for reference):**
- pi_8: 0.5000 (var=0.0067)
- pi_12: 0.8000 (var=0.0345)
- pi_1: 0.4969 (var=0.0031)
- pi_2: 0.4894 (var=0.0056)
- pi_3: 0.4581 (var=0.0092)
- pi_4: 0.4844 (var=0.0058)
- pi_5: 0.4219 (var=0.0071)
- pi_6: 0.4906 (var=0.0089)
- pi_7: 0.4894 (var=0.0055)
- pi_9: 0.5731 (var=0.0369)
- pi_10: 0.4575 (var=0.0085)
- pi_11: 0.4062 (var=0.0154)

### Experiment 22
**Design**
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sums_a = data['option_a_ratings'].apply(sum)
    sums_b = data['option_b_ratings'].apply(sum)
    tied = data[sums_a == sums_b]
    if len(tied) == 0:
        return 0.5
    return float((tied['response'] == 0).mean())
```

**Observed (real) value:** 0.5684 (var=0.0814)
**Candidate trajectory (this loop):**
  - iter 1: 0.1442 (var=0.0140) (Δ vs real -0.4242)
  - iter 2: 0.4995 (var=0.1264) (Δ vs real -0.0689)
  - iter 3: 0.4768 (var=0.1332) (Δ vs real -0.0916)
  - iter 4: 0.5174 (var=0.1353) (Δ vs real -0.0511)
  - iter 5: 0.4605 (var=0.1181) (Δ vs real -0.1079)
  - iter 6: 0.5426 (var=0.1181) (Δ vs real -0.0258)
  - iter 7: 0.5600 (var=0.1170) (Δ vs real -0.0084)
  - iter 8: 0.5553 (var=0.1395) (Δ vs real -0.0132)
  - iter 9 (current): 0.5774 (var=0.1240) (Δ vs real +0.0089)
**Other theories' values on this metric (for reference):**
- pi_12: 0.8284 (var=0.0469)
- pi_8: 0.5084 (var=0.0076)
- pi_1: 0.1584 (var=0.0089)
- pi_2: 0.5116 (var=0.0059)
- pi_3: 0.1737 (var=0.0151)
- pi_4: 0.2558 (var=0.0164)
- pi_5: 0.2500 (var=0.0142)
- pi_6: 0.4611 (var=0.0132)
- pi_7: 0.2084 (var=0.0201)
- pi_9: 0.4111 (var=0.1285)
- pi_10: 0.2816 (var=0.0362)
- pi_11: 0.3716 (var=0.0228)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The WADD-NL candidate was accepted and establishes a solid baseline, but it struggles significantly on experiments designed to detect Reverse TTB-like behavior or negative weighting of high-validity cues (e.g., Exp 13, 14, 18, 19, where observed values are well below 0.5, but the candidate predicts > 0.8). The current power-function parameterization with `gamma >= 0` only allows interpolation between Tallying (gamma=0) and TTB (gamma>0). It cannot capture the empirical tendency of some subjects to down-weight or anti-weight the highest validity cues in tied-sum scenarios.
Rationale: To maintain the WADD-NL family while capturing the full empirical spectrum (TTB, Tallying, and Reverse TTB-like behaviors), revise the cue weighting function. Instead of a power function on shifted validities, use an exponential weighting scheme: `w = np.exp(gamma * val)`. Expand the range of `gamma` to allow negative values (e.g., `[-10.0, 10.0]`). When `gamma > 0`, the model will approximate TTB; when `gamma == 0`, it approximates Tallying; and critically, when `gamma < 0`, it will upweight lower-validity cues, naturally capturing the Reverse TTB-like patterns seen in Exps 13, 14, 18, and 19 without leaving the continuous-integration WADD family.

**Outcome of your advice:** iter 1 candidate loss=0.5013 -> iter 2 candidate loss=0.4408 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The introduction of the exponential weighting scheme (WADD-EXP) successfully improved the aggregate loss by allowing the model to capture Reverse TTB-like patterns (e.g., Exps 13, 14, 18, 19) through negative gamma values. However, the model still exhibits significant deviations on several key experiments, notably Exps 1, 4, 8, and 9. The current formulation `w = np.exp(gamma * val)` means that because validities are bounded between 0.5 and 1.0, even non-predictive cues (val=0.5) are scaled dramatically when gamma is large. This lacks a principled baseline and likely contributes to the misfit and high variance in certain tasks.
Rationale: To improve the WADD-EXP model while staying strictly within the prescribed family, center the validities at chance level before applying the exponential scaling: use `w = np.exp(gamma * (val - 0.5))`. This ensures that a completely non-predictive cue (validity 0.5) always receives a baseline weight of 1 (since exp(0)=1), while predictive cues scale exponentially up or down relative to chance depending on gamma. Additionally, narrow the range of `gamma` to `[-10.0, 10.0]` to prevent the model from collapsing into overly extreme weight distributions, which should stabilize fits on Exps 8 and 9.

**Outcome of your advice:** iter 2 candidate loss=0.4408 -> iter 3 candidate loss=0.5058 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The current candidate (iter 3) was rejected by the accept gate, meaning the aggregate loss worsened compared to the iter 2 base. The attempt to center validities at 0.5 and remove max-normalization likely caused the absolute scale of the accumulated evidence to vary wildly depending on the value of gamma. Because the scores were no longer bounded, the beta parameter (temperature) could not stably map evidence differences to choice probabilities across the diverse experimental designs, leading to poor fits on experiments like 1, 4, 8, 9, and 15.
Rationale: Since the unnormalized centered exponential was rejected, we should return to the iter 2 base but implement a more robust normalization scheme. Specifically, apply a softmax function over the scaled validities: `w = np.exp(gamma * val) / np.sum(np.exp(gamma * val))`. This elegantly bounds the weights so they always sum to 1, providing a perfectly stable scale for the accumulated scores regardless of whether gamma is large, zero, or negative. Positive gamma will smoothly approach TTB, gamma=0 yields pure Tallying, and negative gamma captures Reverse TTB. Because the maximum possible score difference is now bounded by 1.0, you should expand the upper bound of the `beta` parameter (e.g., `[0.1, 50.0]`) so the model can still make deterministic predictions when needed.

**Outcome of your advice:** iter 3 candidate loss=0.5058 -> iter 4 candidate loss=0.4807 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate (iter 4) was rejected by the accept gate, meaning the aggregate loss worsened compared to the iter 2 base. The attempt to use a softmax normalization over the cue weights squashed the evidence differences into a simplex, which likely distorted the relative distances between cues when gamma is moderate. This caused further degradation on experiments that rely on delicate evidence accumulation (Exps 1, 4, 9, 15), where the model's predictions drifted further away from human data compared to iter 2.
Rationale: Since the softmax weight normalization was rejected, we should return to the iter 2 base but implement a simpler, non-sum-constrained max-normalization to prevent overflow while preserving the raw relative distances of the exponential function. Specifically, compute the unnormalized log-weights `z = gamma * val`, and then simply anchor the maximum weight to 1.0 using `w = np.exp(z - np.max(z))`. Do NOT divide by the sum. This ensures the most favored cue (whether highest validity if gamma > 0, or lowest if gamma < 0) always has a weight of exactly 1.0, while the others decay exponentially. Keep the beta parameter's upper bound around 20.0 to allow for smooth probabilistic mapping without the evidence scale being artificially constrained by a sum-to-1 requirement.

**Outcome of your advice:** iter 4 candidate loss=0.4807 -> iter 5 candidate loss=0.5055 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The current candidate (iter 5) was rejected by the gate, meaning the aggregate loss worsened compared to the iter 2 base. Over the last three iterations, attempts to stabilize the exponential weighting scheme (centering validities, softmax normalization, and max-normalization) have all failed to improve upon the unnormalized exponential from iter 2. The core issue is that `exp(gamma * val)` creates massive absolute scale differences depending on `gamma`, making it difficult for a single `beta` parameter to map evidence differences to choice probabilities consistently across all experimental designs.
Rationale: Since all exponential normalization schemes have been rejected, we should return to the power-function parameterization (`w = val ** gamma`) from the very first iteration, but crucially, keep the expanded range for gamma (`[-10.0, 10.0]` or even `[-20.0, 20.0]`). A power function naturally anchors a validity of 1.0 to a weight of 1.0 (since `1.0 ** gamma = 1.0`), providing built-in numerical stability without needing any artificial normalization steps. When `gamma > 0`, it approximates TTB; when `gamma == 0`, it yields pure Tallying; and critically, when `gamma < 0`, lower validities (e.g., 0.6) will correctly receive much higher weights than higher validities (e.g., 0.9), naturally capturing the Reverse TTB behaviors seen in Exps 13, 14, 18, and 19. This fulfills the arbiter's WADD-NL prescription while fixing the scaling instabilities of the exponential.

**Outcome of your advice:** iter 5 candidate loss=0.5055 -> iter 6 candidate loss=0.4312 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The return to the power-function parameterization (WADD-PWR) was highly successful. As predicted, allowing `gamma` to span `[-20.0, 20.0]` natively captured both TTB (gamma > 0) and Reverse TTB (gamma < 0) behaviors without the catastrophic scaling instabilities of the exponential functions, leading to the lowest aggregate loss yet (0.4312). However, the model still slightly underpredicts human choice consistency on tasks that strongly favor TTB (e.g., Exps 1, 4, 9, and 15). This happens because when `gamma` is large and positive, `val ** gamma` shrinks the absolute magnitude of all weights (e.g., `0.9 ** 20 ≈ 0.12`), which compresses the accumulated evidence differences and limits the sharpness of the softmax even at maximum `beta`.
Rationale: To fix the remaining underprediction on TTB-sensitive experiments while keeping the highly successful power-function family intact, apply a simple max-normalization to the weights before evidence accumulation: `w = val ** gamma` followed by `w = w / np.max(w)`. This guarantees that the most heavily weighted cue (whether it's the highest validity cue when gamma > 0, or the lowest when gamma < 0) always receives an absolute weight of exactly 1.0. This completely decouples the absolute scale of the evidence from the value of gamma, allowing the `beta` parameter to fully and reliably control choice determinism across the entire strategy spectrum. Keep the parameter ranges as they are.

**Outcome of your advice:** iter 6 candidate loss=0.4312 -> iter 7 candidate loss=0.4677 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The attempt to apply max-normalization to the power-function weights (Iter 7) was rejected by the accept gate, meaning the aggregate loss worsened compared to the Iter 6 base. While max-normalization mathematically decoupled the evidence scale from gamma, it likely destroyed the implicit regularization that the unnormalized power function provided, forcing the model into overly deterministic predictions across the board and worsening the fit on experiments that require delicate probabilistic scaling (e.g., Exps 1, 4, 9, 15). The unnormalized power function from Iter 6 remains the best approach.
Rationale: Since max-normalization was rejected, the loop will automatically revert to the Iter 6 base (`w = val ** gamma`). To address the original issue of evidence compression when `gamma` is large (which limits determinism on TTB-sensitive tasks), we should simply expand the upper bound of the `beta` parameter instead of altering the weighting equation. Increase the `beta` range from `[0.1, 20.0]` to `[0.1, 100.0]`. This allows the model to naturally learn a higher inverse temperature to compensate for the shrunken absolute weights when necessary, without artificially distorting the evidence scale or losing the implicit regularization of the unnormalized power function. Keep `gamma` at `[-20.0, 20.0]`.

**Outcome of your advice:** iter 7 candidate loss=0.4677 -> iter 8 candidate loss=0.4623 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The current candidate (Iter 8) was rejected by the accept gate, meaning the aggregate loss worsened compared to the Iter 6 base. The attempt to address evidence compression by expanding the upper bound of the `beta` parameter to 100.0 backfired. While it allowed for sharper predictions, it likely caused the model to overfit or become overly deterministic on tasks with noisy human choice patterns (e.g., Exps 6, 8, and 9), significantly degrading the fit compared to Iter 6. Both max-normalization (Iter 7) and expanding the temperature bounds (Iter 8) have failed to improve upon the unnormalized power function from Iter 6.
Rationale: Since expanding `beta` was rejected, the loop will automatically revert to the Iter 6 base (`w = val ** gamma`). The core unnormalized power function is structurally sound, but the instability likely stems from extreme negative values of `gamma`. When `gamma` is large and negative (e.g., -20), cues with validities near 0.5 receive massively explosive weights (e.g., 0.5 ** -20 = 1,048,576), which completely breaks the `beta` scaling and forces extreme deterministic choices. To fix this while staying strictly in the WADD-NL family, keep the Iter 6 equation but narrow the range of `gamma` to `[-10.0, 10.0]`. This still allows for robust Reverse TTB behavior (down-weighting high validities) without causing numerical explosions that destroy the softmax calibration. Return the `beta` range to its standard `[0.1, 20.0]`.

**Outcome of your advice:** iter 8 candidate loss=0.4623 -> CURRENT candidate loss=0.4646 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The current candidate (Iter 9) was rejected by the accept gate, meaning the aggregate loss worsened compared to the Iter 6 base. The attempt to prevent numerical explosions by narrowing the gamma bounds to `[-10.0, 10.0]` restricted the model's ability to capture strong TTB and Reverse TTB behaviors, significantly degrading performance on Exps 1, 4, 9, and 15. The core issue with the Iter 6 base remains: when using an unnormalized power function `val ** gamma`, large positive gammas shrink all absolute weights to near-zero (compressing evidence and causing under-confidence), while large negative gammas cause the weights to explode (causing overflow and broken beta scaling). Previous attempts to fix this via raw max-normalization (Iter 7) or expanding the beta range (Iter 8) both failed.",
  "verdict": "regenerate",
  "rationale": "To preserve the highly successful power-function weighting from Iter 6 while definitively fixing the numerical scaling issues without distorting the model, implement a sum-normalized power function using the numerically stable log-max trick. Specifically, compute the log-weights: `log_w = gamma * np.log(val)`. Then apply the softmax transformation: `w = np.exp(log_w - np.max(log_w))` followed by `w = w / np.sum(w)`. This is mathematically identical to `(val ** gamma) / np.sum(val ** gamma)`, but it is completely immune to overflow or underflow regardless of how extreme `gamma` gets. This ensures the total weight always sums to 1.0, providing a perfectly stable and bounded scale for the `beta` parameter. Restore `gamma` to `[-20.0, 20.0]` to allow the full strategic spectrum, and keep `beta` in `[0.1, 20.0]`."
}
```

## Usage

```json
{
  "prompt_token_count": 32658,
  "candidates_token_count": 420,
  "total_token_count": 35940
}
```
