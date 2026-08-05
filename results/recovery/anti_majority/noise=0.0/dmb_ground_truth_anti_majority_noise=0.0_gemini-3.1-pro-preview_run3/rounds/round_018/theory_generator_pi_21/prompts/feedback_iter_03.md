# feedback_iter_03

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
- THEORY 1 = `pi_17`
- THEORY 2 = `pi_20`
- The recommendation below acts on THEORY 2 (= `pi_20`).

Propose a 'Strategy Mixture' theory. Instead of a single complex evidence accumulator, assume subjects probabilistically sample between distinct decision strategies on each trial: a non-compensatory heuristic (like Take-The-Best, which stops at the first discriminating cue) and a compensatory heuristic (like Tallying or simple Unit-Weight Additive, which counts the number of supporting cues). The probability of using each strategy could vary across subjects or be influenced by the dispersion of cue validities. This mixture approach can naturally account for both strong primacy effects (via TTB) and sensitivity to secondary/shared cues (via Tallying) without introducing the overly aggressive self-dilution seen in the Lateral Inhibition model.


## CANDIDATE THEORY
Strategy Mixture with Validity-Scaled TTB and Logistic Dispersion Modulation: Decision-makers probabilistically sample between a non-compensatory heuristic (Take-The-Best) and a compensatory heuristic (Unit-Weight Tallying). To better capture confidence in TTB, the evidence vector for TTB is scaled by the validity of the first discriminating cue, rather than being a rigid binary prediction. Furthermore, the probability of employing TTB is determined by a logistic function of the environmental dispersion of cue validities, allowing for a smooth, thresholded transition between compensatory and non-compensatory strategies based on context.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy - scaled by the validity of the discriminating cue
    ev_ttb = np.array([0.0, 0.0])
    for i in cue_order:
        if a[i] > b[i]:
            ev_ttb = np.array([val[i], 0.0])
            break
        elif b[i] > a[i]:
            ev_ttb = np.array([0.0, val[i]])
            break
            
    # Tallying Strategy (Unit-Weight Additive)
    ev_tally = np.array([np.sum(a), np.sum(b)])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB probabilities
    z_ttb = beta_ttb * ev_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb = p_ttb / np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta_tally * ev_tally
    z_tally = z_tally - np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally = p_tally / np.sum(p_tally)
    
    # Strategy Mixture Weight (Logistic modulation based on dispersion)
    w_base = float(parameters["w_base"])
    gamma = float(parameters["gamma"])
    dispersion = np.std(val) if len(val) > 1 else 0.0
    
    logit_w = w_base + gamma * dispersion
    w = 1.0 / (1.0 + np.exp(-logit_w))
    
    p_mix = w * p_ttb + (1.0 - w) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

`parameters`:
- w_base: [-5.0, 5.0]
- gamma: [-20.0, 20.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

`rationale`:
Following the critic's advice, this minimal edit builds directly on the successful Iteration 1 base. We maintain the Strategy Mixture of TTB and Unit-Weight Tallying. To address the failures in conflict scenarios, we refine the TTB evidence vector by scaling it by the validity of the discriminating cue, allowing TTB's confidence to naturally scale with cue reliability. We also reinstate the validity-dispersion modulation for the mixture weight, but instead of a simple linear clip, we use a logistic function parameterized by a base logit (`w_base`) and a dispersion sensitivity (`gamma`). This provides the flexibility to capture thresholded strategy shifts while keeping the core mechanism intact.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.3958 -> ACCEPTED
- iter 2: loss=0.4079 -> REJECTED
- iter 3: loss=0.4078 -> REJECTED
- iter 4 (current candidate you are grading): loss=0.3823 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.3823.

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
  - iter 1: 0.5487 (var=0.0525) (Δ vs real -0.1021)
  - iter 2: 0.4896 (var=0.0484) (Δ vs real -0.1613)
  - iter 3: 0.4775 (var=0.0546) (Δ vs real -0.1733)
  - iter 4 (current): 0.5733 (var=0.0864) (Δ vs real -0.0775)
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
- pi_13: 0.4533 (var=0.0873)
- pi_14: 0.3396 (var=0.0945)
- pi_15: 0.5775 (var=0.0644)
- pi_16: 0.4779 (var=0.1411)
- pi_17: 0.7825 (var=0.0205)
- pi_18: 0.5608 (var=0.0162)
- pi_19: 0.4725 (var=0.0286)
- pi_20: 0.4983 (var=0.0110)

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
  - iter 1: 0.4919 (var=0.0522) (Δ vs real +0.1908)
  - iter 2: 0.4256 (var=0.0497) (Δ vs real +0.1244)
  - iter 3: 0.5575 (var=0.0427) (Δ vs real +0.2564)
  - iter 4 (current): 0.5069 (var=0.0629) (Δ vs real +0.2058)
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
- pi_13: 0.6303 (var=0.0479)
- pi_14: 0.5461 (var=0.1141)
- pi_15: 0.4664 (var=0.0528)
- pi_16: 0.5619 (var=0.1384)
- pi_17: 0.3364 (var=0.0082)
- pi_18: 0.4344 (var=0.0119)
- pi_19: 0.4992 (var=0.0140)
- pi_20: 0.5147 (var=0.0068)

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
  - iter 1: 0.5619 (var=0.0435) (Δ vs real -0.0481)
  - iter 2: 0.5406 (var=0.0279) (Δ vs real -0.0694)
  - iter 3: 0.5642 (var=0.0243) (Δ vs real -0.0458)
  - iter 4 (current): 0.5825 (var=0.0437) (Δ vs real -0.0275)
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
- pi_13: 0.3767 (var=0.0065)
- pi_14: 0.4292 (var=0.0756)
- pi_15: 0.5729 (var=0.0506)
- pi_16: 0.5181 (var=0.0904)
- pi_17: 0.6192 (var=0.0061)
- pi_18: 0.5400 (var=0.0053)
- pi_19: 0.4054 (var=0.0084)
- pi_20: 0.4827 (var=0.0046)

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
  - iter 1: 0.4696 (var=0.0702) (Δ vs real -0.1687)
  - iter 2: 0.4562 (var=0.0514) (Δ vs real -0.1821)
  - iter 3: 0.4585 (var=0.0402) (Δ vs real -0.1798)
  - iter 4 (current): 0.5369 (var=0.0755) (Δ vs real -0.1015)
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
- pi_13: 0.3390 (var=0.0671)
- pi_14: 0.4158 (var=0.0970)
- pi_15: 0.5404 (var=0.0509)
- pi_16: 0.4894 (var=0.1279)
- pi_17: 0.6208 (var=0.0059)
- pi_18: 0.5577 (var=0.0117)
- pi_19: 0.4940 (var=0.0135)
- pi_20: 0.4494 (var=0.0099)

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
  - iter 1: -0.0250 (var=0.0489) (Δ vs real -0.1075)
  - iter 2: 0.0512 (var=0.0512) (Δ vs real -0.0313)
  - iter 3: -0.0187 (var=0.0593) (Δ vs real -0.1013)
  - iter 4 (current): -0.0012 (var=0.0430) (Δ vs real -0.0838)
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
- pi_13: -0.3000 (var=0.3262)
- pi_14: -0.0113 (var=0.0484)
- pi_15: -0.0050 (var=0.0500)
- pi_16: 0.0025 (var=0.0366)
- pi_17: -0.3162 (var=0.0979)
- pi_18: 0.0413 (var=0.0653)
- pi_19: 0.1913 (var=0.0838)
- pi_20: 0.0175 (var=0.0498)

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
  - iter 1: -0.0050 (var=0.0045) (Δ vs real +0.0117)
  - iter 2: 0.1656 (var=0.0681) (Δ vs real +0.1822)
  - iter 3: 0.0100 (var=0.0080) (Δ vs real +0.0267)
  - iter 4 (current): -0.0050 (var=0.0035) (Δ vs real +0.0117)
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
- pi_13: 0.3944 (var=0.2993)
- pi_14: -0.0078 (var=0.0028)
- pi_15: -0.0094 (var=0.0009)
- pi_16: 0.0017 (var=0.0020)
- pi_17: 0.0783 (var=0.0222)
- pi_18: 0.0294 (var=0.0087)
- pi_19: 0.2039 (var=0.1917)
- pi_20: 0.0261 (var=0.0180)

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
  - iter 1: 0.0087 (var=0.0405) (Δ vs real +0.2137)
  - iter 2: 0.0650 (var=0.0433) (Δ vs real +0.2700)
  - iter 3: -0.0112 (var=0.0476) (Δ vs real +0.1938)
  - iter 4 (current): 0.0875 (var=0.0392) (Δ vs real +0.2925)
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
- pi_13: -0.2688 (var=0.1822)
- pi_14: -0.0200 (var=0.0346)
- pi_15: -0.0075 (var=0.0342)
- pi_16: 0.0162 (var=0.0381)
- pi_17: -0.4150 (var=0.1487)
- pi_18: 0.0375 (var=0.0747)
- pi_19: 0.2613 (var=0.1406)
- pi_20: 0.0563 (var=0.0688)

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
  - iter 1: 1.8526 (var=0.0233) (Δ vs real +0.9201)
  - iter 2: 1.8491 (var=0.0364) (Δ vs real +0.9167)
  - iter 3: 1.8194 (var=0.0278) (Δ vs real +0.8870)
  - iter 4 (current): 1.8707 (var=0.0190) (Δ vs real +0.9382)
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
- pi_13: 1.4948 (var=0.1568)
- pi_14: 1.5803 (var=0.1489)
- pi_15: 1.8798 (var=0.0296)
- pi_16: 1.5639 (var=0.2574)
- pi_17: 1.5772 (var=0.0251)
- pi_18: 1.5602 (var=0.0568)
- pi_19: 1.7018 (var=0.0824)
- pi_20: 1.5384 (var=0.0239)

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
  - iter 1: 0.7053 (var=0.0977) (Δ vs real -0.9495)
  - iter 2: 0.8179 (var=0.0481) (Δ vs real -0.8368)
  - iter 3: 0.6305 (var=0.0736) (Δ vs real -1.0242)
  - iter 4 (current): 0.7926 (var=0.0736) (Δ vs real -0.8621)
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
- pi_13: 0.7000 (var=0.1668)
- pi_14: 0.6095 (var=0.0947)
- pi_15: 0.5842 (var=0.0802)
- pi_16: 0.9474 (var=0.1545)
- pi_17: 0.8947 (var=0.0400)
- pi_18: 0.9884 (var=0.0315)
- pi_19: 1.0316 (var=0.0790)
- pi_20: 0.9526 (var=0.0278)

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
  - iter 1: 0.1008 (var=0.1860) (Δ vs real +0.0123)
  - iter 2: -0.0215 (var=0.1709) (Δ vs real -0.1100)
  - iter 3: -0.0711 (var=0.1714) (Δ vs real -0.1596)
  - iter 4 (current): 0.0428 (var=0.2002) (Δ vs real -0.0458)
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
- pi_13: 0.1309 (var=0.1218)
- pi_14: 0.0758 (var=0.1555)
- pi_15: 0.0052 (var=0.0841)
- pi_16: 0.0843 (var=0.2106)
- pi_17: 0.2047 (var=0.1812)
- pi_18: 0.3135 (var=0.2388)
- pi_19: 0.2625 (var=0.1702)
- pi_20: 0.1998 (var=0.2196)

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
  - iter 1: 0.3477 (var=0.0784) (Δ vs real +0.7769)
  - iter 2: 0.3112 (var=0.0546) (Δ vs real +0.7404)
  - iter 3: 0.3277 (var=0.0583) (Δ vs real +0.7569)
  - iter 4 (current): 0.2858 (var=0.0711) (Δ vs real +0.7150)
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
- pi_13: 0.0292 (var=0.0884)
- pi_14: 0.4273 (var=0.0675)
- pi_15: 0.3115 (var=0.0416)
- pi_16: 0.0592 (var=0.0859)
- pi_17: -0.2081 (var=0.0388)
- pi_18: 0.0127 (var=0.0236)
- pi_19: 0.2077 (var=0.0389)
- pi_20: 0.0508 (var=0.0312)

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
  - iter 1: 0.2978 (var=0.0985) (Δ vs real +0.9689)
  - iter 2: 0.2478 (var=0.0755) (Δ vs real +0.9189)
  - iter 3: 0.2900 (var=0.0621) (Δ vs real +0.9611)
  - iter 4 (current): 0.3489 (var=0.1054) (Δ vs real +1.0200)
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
- pi_13: 0.1067 (var=0.0601)
- pi_14: 0.4144 (var=0.1037)
- pi_15: 0.3922 (var=0.0949)
- pi_16: 0.0033 (var=0.1677)
- pi_17: -0.2833 (var=0.0318)
- pi_18: -0.0122 (var=0.0332)
- pi_19: 0.2856 (var=0.0681)
- pi_20: 0.0633 (var=0.0425)

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
  - iter 1: 0.6322 (var=0.0191) (Δ vs real +0.3678)
  - iter 2: 0.7914 (var=0.0125) (Δ vs real +0.5269)
  - iter 3: 0.6803 (var=0.0169) (Δ vs real +0.4158)
  - iter 4 (current): 0.6597 (var=0.0208) (Δ vs real +0.3953)
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
- pi_13: 0.5275 (var=0.1160)
- pi_14: 0.4253 (var=0.0988)
- pi_15: 0.6722 (var=0.0189)
- pi_16: 0.4183 (var=0.1303)
- pi_17: 0.5478 (var=0.0097)
- pi_18: 0.5439 (var=0.0093)
- pi_19: 0.7128 (var=0.0206)
- pi_20: 0.5211 (var=0.0066)

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
  - iter 1: 0.6567 (var=0.0204) (Δ vs real +0.5217)
  - iter 2: 0.7658 (var=0.0121) (Δ vs real +0.6308)
  - iter 3: 0.6767 (var=0.0170) (Δ vs real +0.5417)
  - iter 4 (current): 0.6875 (var=0.0250) (Δ vs real +0.5525)
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
- pi_13: 0.5663 (var=0.1188)
- pi_14: 0.5204 (var=0.1096)
- pi_15: 0.7350 (var=0.0215)
- pi_16: 0.4608 (var=0.1408)
- pi_17: 0.5283 (var=0.0146)
- pi_18: 0.5704 (var=0.0178)
- pi_19: 0.6446 (var=0.0251)
- pi_20: 0.5142 (var=0.0057)

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
  - iter 1: 0.4569 (var=0.0831) (Δ vs real -0.3462)
  - iter 2: 0.4577 (var=0.0663) (Δ vs real -0.3454)
  - iter 3: 0.4477 (var=0.0447) (Δ vs real -0.3554)
  - iter 4 (current): 0.5431 (var=0.1031) (Δ vs real -0.2600)
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
- pi_13: 0.3331 (var=0.0926)
- pi_14: 0.3246 (var=0.0925)
- pi_15: 0.4715 (var=0.0666)
- pi_16: 0.5038 (var=0.1611)
- pi_17: 0.8215 (var=0.0185)
- pi_18: 0.5500 (var=0.0244)
- pi_19: 0.4662 (var=0.0295)
- pi_20: 0.4600 (var=0.0189)

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
  - iter 1: 0.1979 (var=0.0166) (Δ vs real -0.0632)
  - iter 2: 0.1995 (var=0.0201) (Δ vs real -0.0616)
  - iter 3: 0.1789 (var=0.0155) (Δ vs real -0.0821)
  - iter 4 (current): 0.2221 (var=0.0226) (Δ vs real -0.0389)
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
- pi_13: 0.2089 (var=0.0253)
- pi_14: 0.2716 (var=0.0199)
- pi_15: 0.2158 (var=0.0213)
- pi_16: 0.3584 (var=0.0092)
- pi_17: 0.0974 (var=0.0035)
- pi_18: 0.0779 (var=0.0063)
- pi_19: 0.0900 (var=0.0047)
- pi_20: 0.0732 (var=0.0025)

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
  - iter 1: 0.6713 (var=0.0257) (Δ vs real +0.3888)
  - iter 2: 0.6704 (var=0.0106) (Δ vs real +0.3879)
  - iter 3: 0.6858 (var=0.0156) (Δ vs real +0.4033)
  - iter 4 (current): 0.6746 (var=0.0303) (Δ vs real +0.3921)
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
- pi_13: 0.5713 (var=0.0494)
- pi_14: 0.5308 (var=0.0766)
- pi_15: 0.7100 (var=0.0246)
- pi_16: 0.4537 (var=0.1374)
- pi_17: 0.7708 (var=0.0132)
- pi_18: 0.5429 (var=0.0066)
- pi_19: 0.5775 (var=0.0111)
- pi_20: 0.5204 (var=0.0068)

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
  - iter 1: 0.6571 (var=0.0182) (Δ vs real +0.3113)
  - iter 2: 0.6983 (var=0.0171) (Δ vs real +0.3525)
  - iter 3: 0.6779 (var=0.0169) (Δ vs real +0.3321)
  - iter 4 (current): 0.6358 (var=0.0293) (Δ vs real +0.2900)
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
- pi_13: 0.6421 (var=0.0262)
- pi_14: 0.4813 (var=0.1119)
- pi_15: 0.7438 (var=0.0279)
- pi_16: 0.4288 (var=0.1416)
- pi_17: 0.7392 (var=0.0154)
- pi_18: 0.5437 (var=0.0077)
- pi_19: 0.5954 (var=0.0132)
- pi_20: 0.5142 (var=0.0053)

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
  - iter 1: 0.6921 (var=0.0203) (Δ vs real +0.5163)
  - iter 2: 0.7242 (var=0.0141) (Δ vs real +0.5484)
  - iter 3: 0.6700 (var=0.0209) (Δ vs real +0.4942)
  - iter 4 (current): 0.6900 (var=0.0265) (Δ vs real +0.5142)
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
- pi_13: 0.4011 (var=0.1217)
- pi_14: 0.5968 (var=0.0905)
- pi_15: 0.7458 (var=0.0231)
- pi_16: 0.4921 (var=0.1261)
- pi_17: 0.6858 (var=0.0124)
- pi_18: 0.5726 (var=0.0087)
- pi_19: 0.6084 (var=0.0154)
- pi_20: 0.4921 (var=0.0062)

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
  - iter 1: -0.0475 (var=0.0259) (Δ vs real -0.2500)
  - iter 2: -0.1525 (var=0.0358) (Δ vs real -0.3550)
  - iter 3: -0.0037 (var=0.0319) (Δ vs real -0.2062)
  - iter 4 (current): -0.0038 (var=0.0308) (Δ vs real -0.2063)
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
- pi_13: -0.1350 (var=0.0840)
- pi_14: -0.0663 (var=0.0301)
- pi_15: -0.0688 (var=0.0219)
- pi_16: -0.0075 (var=0.0196)
- pi_17: 0.2375 (var=0.0603)
- pi_18: 0.0162 (var=0.0389)
- pi_19: -0.1538 (var=0.0844)
- pi_20: 0.0112 (var=0.0262)

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
  - iter 1: 0.5050 (var=0.0081) (Δ vs real +0.1000)
  - iter 2: 0.3756 (var=0.0117) (Δ vs real -0.0294)
  - iter 3: 0.5081 (var=0.0063) (Δ vs real +0.1031)
  - iter 4 (current): 0.4963 (var=0.0082) (Δ vs real +0.0912)
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
- pi_13: 0.6406 (var=0.0538)
- pi_14: 0.4944 (var=0.0050)
- pi_15: 0.4969 (var=0.0047)
- pi_16: 0.5069 (var=0.0043)
- pi_17: 0.5837 (var=0.0119)
- pi_18: 0.4587 (var=0.0145)
- pi_19: 0.3531 (var=0.0187)
- pi_20: 0.4881 (var=0.0089)

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
  - iter 1: 0.3347 (var=0.0258) (Δ vs real -0.2337)
  - iter 2: 0.2174 (var=0.0125) (Δ vs real -0.3511)
  - iter 3: 0.3400 (var=0.0145) (Δ vs real -0.2284)
  - iter 4 (current): 0.3595 (var=0.0269) (Δ vs real -0.2089)
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
- pi_13: 0.5484 (var=0.1431)
- pi_14: 0.3879 (var=0.1063)
- pi_15: 0.2926 (var=0.0282)
- pi_16: 0.5274 (var=0.1550)
- pi_17: 0.4779 (var=0.0167)
- pi_18: 0.4532 (var=0.0175)
- pi_19: 0.3232 (var=0.0227)
- pi_20: 0.4768 (var=0.0050)

### Experiment 23
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify zero-conflict trials where the total number of positive cues is equal
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    zero_conflict_mask = (a_sums == b_sums)
    
    df_zero = data[zero_conflict_mask]
    if df_zero.empty:
        return 0.0
        
    # Calculate proportion of A choices (response == 0) per subject
    p_a = (df_zero['response'] == 0).groupby(df_zero['subject_id']).mean()
    
    # Mean absolute deviation from 0.5 across subjects
    return float(np.mean(np.abs(p_a - 0.5)))

```

**Observed (real) value:** 0.3133 (var=0.0161)
**Candidate trajectory (this loop):**
  - iter 1: 0.2204 (var=0.0199) (Δ vs real -0.0929)
  - iter 2: 0.1487 (var=0.0135) (Δ vs real -0.1646)
  - iter 3: 0.2029 (var=0.0161) (Δ vs real -0.1104)
  - iter 4 (current): 0.1762 (var=0.0207) (Δ vs real -0.1371)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0563 (var=0.0017)
- pi_13: 0.2817 (var=0.0201)
- pi_1: 0.3250 (var=0.0135)
- pi_2: 0.0521 (var=0.0017)
- pi_3: 0.2079 (var=0.0144)
- pi_4: 0.2804 (var=0.0146)
- pi_5: 0.1404 (var=0.0110)
- pi_6: 0.0675 (var=0.0028)
- pi_7: 0.2533 (var=0.0121)
- pi_9: 0.2367 (var=0.0190)
- pi_10: 0.1783 (var=0.0203)
- pi_11: 0.0700 (var=0.0032)
- pi_12: 0.3850 (var=0.0106)
- pi_14: 0.2350 (var=0.0202)
- pi_15: 0.2029 (var=0.0223)
- pi_16: 0.3871 (var=0.0078)
- pi_17: 0.1263 (var=0.0068)
- pi_18: 0.0546 (var=0.0012)
- pi_19: 0.0754 (var=0.0031)
- pi_20: 0.0608 (var=0.0024)

### Experiment 24
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate sum of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for zero-difference trials (where total cue counts are equal)
    zero_diff = data[sum_a == sum_b].copy()
    
    # Create a unique string identifier for the trial types
    zero_diff['trial_type'] = zero_diff['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + zero_diff['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times each subject chose Option A (response == 0) for each trial type
    prop_a = zero_diff.groupby(['subject_id', 'trial_type'])['response'].apply(lambda x: (x == 0).mean()).reset_index()
    
    # Calculate the absolute deviation from 0.5 (random guessing)
    prop_a['abs_dev'] = (prop_a['response'] - 0.5).abs()
    
    # Average the absolute deviation across trial types for each subject, then return the overall mean
    return float(prop_a.groupby('subject_id')['abs_dev'].mean().mean())
```

**Observed (real) value:** 0.3702 (var=0.0075)
**Candidate trajectory (this loop):**
  - iter 1: 0.1772 (var=0.0102) (Δ vs real -0.1930)
  - iter 2: 0.2586 (var=0.0099) (Δ vs real -0.1116)
  - iter 3: 0.2088 (var=0.0147) (Δ vs real -0.1614)
  - iter 4 (current): 0.2242 (var=0.0184) (Δ vs real -0.1460)
**Other theories' values on this metric (for reference):**
- pi_13: 0.3481 (var=0.0112)
- pi_8: 0.0958 (var=0.0022)
- pi_1: 0.3684 (var=0.0095)
- pi_2: 0.0979 (var=0.0013)
- pi_3: 0.3025 (var=0.0075)
- pi_4: 0.2744 (var=0.0080)
- pi_5: 0.2260 (var=0.0141)
- pi_6: 0.1372 (var=0.0048)
- pi_7: 0.2681 (var=0.0117)
- pi_9: 0.2996 (var=0.0161)
- pi_10: 0.2449 (var=0.0134)
- pi_11: 0.1646 (var=0.0105)
- pi_12: 0.3611 (var=0.0193)
- pi_14: 0.3074 (var=0.0144)
- pi_15: 0.2088 (var=0.0158)
- pi_16: 0.3656 (var=0.0105)
- pi_17: 0.1347 (var=0.0051)
- pi_18: 0.1235 (var=0.0051)
- pi_19: 0.1856 (var=0.0100)
- pi_20: 0.0926 (var=0.0016)

### Experiment 25
**Design**
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        # In all 10 trials, one option has 4 or 5 cues, the other has 0 or 1.
        # We identify the dominant option (the one with more 1s).
        if sum(a) > sum(b):
            return str(a) + str(b), 1 if resp == 0 else 0
        else:
            return str(b) + str(a), 1 if resp == 1 else 0

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_dom'])
    df['subject_id'] = data['subject_id'].values
    
    counts = df.groupby(['subject_id', 'trial'])['chose_dom'].agg(['sum', 'count'])
    
    def calc_M(sub_df):
        valid = sub_df[sub_df['count'] > 1]
        if len(valid) < 2:
            return np.nan
        
        X = valid['sum'].values.astype(float)
        R = valid['count'].values.astype(float)
        Y = X / R
        
        # S2_Y is the sample variance of the observed choice proportions across the 10 trials
        S2_Y = np.var(Y, ddof=1)
        
        # W_t is the exact unbiased estimator of the binomial variance for trial t: p_t(1-p_t)/R_t
        W = X * (R - X) / (R**2 * (R - 1.0))
        mean_W = np.mean(W)
        
        # M is the unbiased estimator of the variance of the true underlying choice probabilities
        return S2_Y - mean_W

    M_per_subj = counts.groupby('subject_id').apply(calc_M).dropna()
    if M_per_subj.empty:
        return 0.0
        
    return float(M_per_subj.mean())
```

**Observed (real) value:** -0.0010 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: -0.0015 (var=0.0000) (Δ vs real -0.0005)
  - iter 2: -0.0013 (var=0.0001) (Δ vs real -0.0003)
  - iter 3: -0.0007 (var=0.0001) (Δ vs real +0.0003)
  - iter 4 (current): -0.0002 (var=0.0001) (Δ vs real +0.0008)
**Other theories' values on this metric (for reference):**
- pi_14: 0.0002 (var=0.0001)
- pi_13: 0.0005 (var=0.0000)
- pi_1: 0.0001 (var=0.0000)
- pi_2: -0.0010 (var=0.0001)
- pi_3: 0.0025 (var=0.0001)
- pi_4: -0.0001 (var=0.0001)
- pi_5: -0.0002 (var=0.0001)
- pi_6: -0.0005 (var=0.0002)
- pi_7: 0.0011 (var=0.0001)
- pi_8: 0.0019 (var=0.0001)
- pi_9: 0.0061 (var=0.0002)
- pi_10: -0.0006 (var=0.0001)
- pi_11: -0.0016 (var=0.0001)
- pi_12: -0.0004 (var=0.0000)
- pi_15: -0.0005 (var=0.0000)
- pi_16: 0.0000 (var=0.0000)
- pi_17: 0.0182 (var=0.0006)
- pi_18: 0.0001 (var=0.0001)
- pi_19: 0.0012 (var=0.0001)
- pi_20: 0.0029 (var=0.0002)

### Experiment 26
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    is_A = (data['response'] == 0).astype(float)
    
    t1 = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 0))
    t4 = (data['A_tuple'] == (1, 0, 0, 0, 1)) & (data['B_tuple'] == (0, 1, 1, 0, 0))
    t5 = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0))
    t7 = (data['A_tuple'] == (1, 0, 1, 1, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 0))
    
    p1 = is_A[t1].mean() if t1.sum() > 0 else 0.5
    p4 = is_A[t4].mean() if t4.sum() > 0 else 0.5
    p5 = is_A[t5].mean() if t5.sum() > 0 else 0.5
    p7 = is_A[t7].mean() if t7.sum() > 0 else 0.5
    
    return float((p1 - p4) + (p5 - p7))
```

**Observed (real) value:** 0.0154 (var=0.0544)
**Candidate trajectory (this loop):**
  - iter 1: 0.0015 (var=0.0581) (Δ vs real -0.0138)
  - iter 2: 0.2062 (var=0.1123) (Δ vs real +0.1908)
  - iter 3: -0.0692 (var=0.0538) (Δ vs real -0.0846)
  - iter 4 (current): 0.0154 (var=0.0516) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_13: -0.6923 (var=0.7401)
- pi_14: -0.0462 (var=0.0386)
- pi_1: -0.0323 (var=0.0341)
- pi_2: 0.0154 (var=0.0627)
- pi_3: 0.1354 (var=0.0877)
- pi_4: -0.0092 (var=0.0380)
- pi_5: 0.2585 (var=0.0577)
- pi_6: 0.0815 (var=0.0832)
- pi_7: 0.0492 (var=0.0560)
- pi_8: 0.0138 (var=0.0854)
- pi_9: -0.3308 (var=0.3165)
- pi_10: 0.1492 (var=0.0567)
- pi_11: 0.1662 (var=0.0718)
- pi_12: -0.8031 (var=0.1420)
- pi_15: -0.0354 (var=0.0400)
- pi_16: 0.0015 (var=0.0408)
- pi_17: -0.0954 (var=0.1085)
- pi_18: 0.1477 (var=0.1318)
- pi_19: 0.2908 (var=0.0718)
- pi_20: -0.0385 (var=0.0862)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['a_first'] = df['option_a_ratings'].apply(lambda x: x[0])
    df['b_sum'] = df['option_b_ratings'].apply(sum)
    df['a_sum'] = df['option_a_ratings'].apply(sum)
    
    # T5, T6: A wins TTB (a_first == 1) and B has massive tally advantage (b_sum in [5, 6])
    mask_A = (df['a_first'] == 1) & (df['b_sum'].isin([5, 6]))
    p_A = (df.loc[mask_A, 'response'] == 0).mean() if mask_A.any() else 0.5
    
    # T9: B wins TTB (a_first == 0) and A has massive tally advantage (a_sum == 6)
    mask_B = (df['a_first'] == 0) & (df['a_sum'] == 6)
    p_B = (df.loc[mask_B, 'response'] == 1).mean() if mask_B.any() else 0.5
    
    return float((p_A + p_B) / 2.0)
```

**Observed (real) value:** 0.8320 (var=0.0112)
**Candidate trajectory (this loop):**
  - iter 1: 0.4295 (var=0.0891) (Δ vs real -0.4025)
  - iter 2: 0.5360 (var=0.0528) (Δ vs real -0.2960)
  - iter 3: 0.4675 (var=0.0457) (Δ vs real -0.3645)
  - iter 4 (current): 0.5150 (var=0.0891) (Δ vs real -0.3170)
**Other theories' values on this metric (for reference):**
- pi_14: 0.2405 (var=0.0648)
- pi_15: 0.5480 (var=0.0879)
- pi_1: 0.8665 (var=0.0132)
- pi_2: 0.1265 (var=0.0098)
- pi_3: 0.5865 (var=0.1055)
- pi_4: 0.5895 (var=0.0309)
- pi_5: 0.5865 (var=0.0381)
- pi_6: 0.4670 (var=0.0248)
- pi_7: 0.7810 (var=0.0244)
- pi_8: 0.7690 (var=0.0442)
- pi_9: 0.4465 (var=0.1257)
- pi_10: 0.6190 (var=0.0457)
- pi_11: 0.2805 (var=0.0282)
- pi_12: 0.8245 (var=0.0590)
- pi_13: 0.3855 (var=0.1025)
- pi_16: 0.4980 (var=0.1427)
- pi_17: 0.7755 (var=0.0316)
- pi_18: 0.6070 (var=0.0268)
- pi_19: 0.4195 (var=0.0403)
- pi_20: 0.4635 (var=0.0108)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_ttb_choice(row):
        # Cue 0 is the highest validity cue (0.95)
        a_wins_ttb = row['option_a_ratings'][0] > row['option_b_ratings'][0]
        ttb_winner = 0 if a_wins_ttb else 1
        return 1.0 if row['response'] == ttb_winner else 0.0
        
    return float(data.apply(is_ttb_choice, axis=1).mean())
```

**Observed (real) value:** 0.1467 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: 0.8179 (var=0.0154) (Δ vs real +0.6712)
  - iter 2: 0.8306 (var=0.0094) (Δ vs real +0.6840)
  - iter 3: 0.8558 (var=0.0086) (Δ vs real +0.7092)
  - iter 4 (current): 0.8519 (var=0.0085) (Δ vs real +0.7052)
**Other theories' values on this metric (for reference):**
- pi_15: 0.9419 (var=0.0103)
- pi_14: 0.8358 (var=0.0066)
- pi_1: 0.8494 (var=0.0101)
- pi_2: 0.8677 (var=0.0069)
- pi_3: 0.8154 (var=0.0156)
- pi_4: 0.9071 (var=0.0113)
- pi_5: 0.8698 (var=0.0077)
- pi_6: 0.5813 (var=0.0238)
- pi_7: 0.7802 (var=0.0151)
- pi_8: 0.8060 (var=0.0123)
- pi_9: 0.7481 (var=0.0211)
- pi_10: 0.7519 (var=0.0261)
- pi_11: 0.7129 (var=0.0139)
- pi_12: 0.8721 (var=0.0114)
- pi_13: 0.8573 (var=0.0078)
- pi_16: 0.4644 (var=0.0984)
- pi_17: 0.6400 (var=0.0077)
- pi_18: 0.5896 (var=0.0189)
- pi_19: 0.7556 (var=0.0291)
- pi_20: 0.7354 (var=0.0198)

### Experiment 29
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['chose_A'] = 1 - data['response']
    data['tally_A'] = data['option_a_ratings'].apply(sum)
    data['tally_B'] = data['option_b_ratings'].apply(sum)
    data['tally_diff'] = data['tally_A'] - data['tally_B']
    
    pos_mean = data[data['tally_diff'] > 0]['chose_A'].mean()
    neg_mean = data[data['tally_diff'] < 0]['chose_A'].mean()
    
    if pd.isna(pos_mean): pos_mean = 0.5
    if pd.isna(neg_mean): neg_mean = 0.5
    
    return float(pos_mean - neg_mean)
```

**Observed (real) value:** -0.6071 (var=0.0412)
**Candidate trajectory (this loop):**
  - iter 1: 0.3223 (var=0.0919) (Δ vs real +0.9294)
  - iter 2: 0.3633 (var=0.0854) (Δ vs real +0.9704)
  - iter 3: 0.3365 (var=0.0634) (Δ vs real +0.9435)
  - iter 4 (current): 0.3754 (var=0.0893) (Δ vs real +0.9825)
**Other theories' values on this metric (for reference):**
- pi_16: 0.0298 (var=0.2381)
- pi_15: 0.4731 (var=0.0711)
- pi_1: 0.0175 (var=0.0050)
- pi_2: 0.7331 (var=0.0311)
- pi_3: 0.0921 (var=0.0541)
- pi_4: 0.2338 (var=0.0241)
- pi_5: 0.2946 (var=0.0357)
- pi_6: 0.0677 (var=0.0350)
- pi_7: -0.0010 (var=0.0080)
- pi_8: 0.1056 (var=0.0333)
- pi_9: 0.2456 (var=0.0555)
- pi_10: 0.1169 (var=0.0287)
- pi_11: 0.3692 (var=0.0559)
- pi_12: 0.1290 (var=0.0699)
- pi_13: 0.4502 (var=0.1616)
- pi_14: 0.5008 (var=0.0910)
- pi_17: -0.1767 (var=0.0234)
- pi_18: 0.0394 (var=0.0236)
- pi_19: 0.3385 (var=0.0934)
- pi_20: 0.2892 (var=0.0403)

### Experiment 30
**Design**
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    # Trial 1: A has 4 cues, B has 1 cue
    t1_mask = (a_sum == 4) & (b_sum == 1)
    
    # Trial 7: A has 1 cue, B has 4 cues
    t7_mask = (a_sum == 1) & (b_sum == 4)
    
    # response == 0 means Option A was chosen
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t7 = 1.0 - data.loc[t7_mask, 'response'].mean()
    
    # Handle cases where a subject might have missed a trial type (fallback to 0)
    if np.isnan(p_a_t1):
        p_a_t1 = 0.5
    if np.isnan(p_a_t7):
        p_a_t7 = 0.5
        
    return float(p_a_t1 - p_a_t7)
```

**Observed (real) value:** -0.7262 (var=0.0611)
**Candidate trajectory (this loop):**
  - iter 1: 0.4015 (var=0.1082) (Δ vs real +1.1277)
  - iter 2: 0.3692 (var=0.0933) (Δ vs real +1.0954)
  - iter 3: 0.3938 (var=0.0603) (Δ vs real +1.1200)
  - iter 4 (current): 0.3046 (var=0.0973) (Δ vs real +1.0308)
**Other theories' values on this metric (for reference):**
- pi_15: 0.4292 (var=0.0864)
- pi_16: 0.0246 (var=0.3400)
- pi_1: 0.0308 (var=0.0118)
- pi_2: 0.7138 (var=0.0455)
- pi_3: 0.1200 (var=0.0703)
- pi_4: 0.3246 (var=0.0498)
- pi_5: 0.2323 (var=0.0306)
- pi_6: 0.0662 (var=0.0627)
- pi_7: 0.1046 (var=0.0695)
- pi_8: 0.0477 (var=0.0203)
- pi_9: 0.1631 (var=0.0823)
- pi_10: 0.1508 (var=0.0914)
- pi_11: 0.5292 (var=0.0818)
- pi_12: 0.0415 (var=0.0258)
- pi_13: 0.1938 (var=0.1066)
- pi_14: 0.5462 (var=0.1264)
- pi_17: -0.2677 (var=0.0576)
- pi_18: 0.0600 (var=0.0484)
- pi_19: 0.3369 (var=0.0968)
- pi_20: 0.0369 (var=0.0569)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_a = (1, 0, 0, 0, 0, 0)
    t1_b = (0, 1, 0, 0, 0, 0)
    t3_a = (1, 0, 1, 0, 1, 0)
    t3_b = (0, 1, 0, 0, 0, 0)
    
    t4_a = (1, 0, 0, 0, 0, 0)
    t4_b = (0, 1, 0, 1, 0, 0)
    t7_a = (1, 0, 1, 0, 1, 0)
    t7_b = (0, 1, 0, 1, 0, 1)
    
    mask_t1 = (a_tuples == t1_a) & (b_tuples == t1_b)
    mask_t3 = (a_tuples == t3_a) & (b_tuples == t3_b)
    mask_t4 = (a_tuples == t4_a) & (b_tuples == t4_b)
    mask_t7 = (a_tuples == t7_a) & (b_tuples == t7_b)
    
    diff1 = 0.0
    if mask_t1.sum() > 0 and mask_t3.sum() > 0:
        diff1 = data.loc[mask_t3, 'response'].mean() - data.loc[mask_t1, 'response'].mean()
        
    diff2 = 0.0
    if mask_t4.sum() > 0 and mask_t7.sum() > 0:
        diff2 = data.loc[mask_t7, 'response'].mean() - data.loc[mask_t4, 'response'].mean()
        
    return float(diff1 + diff2)
```

**Observed (real) value:** 0.5022 (var=0.1976)
**Candidate trajectory (this loop):**
  - iter 1: -0.3111 (var=0.0968) (Δ vs real -0.8133)
  - iter 2: -0.3267 (var=0.1246) (Δ vs real -0.8289)
  - iter 3: -0.3133 (var=0.1263) (Δ vs real -0.8156)
  - iter 4 (current): -0.3911 (var=0.1038) (Δ vs real -0.8933)
**Other theories' values on this metric (for reference):**
- pi_17: 0.5044 (var=0.2465)
- pi_15: -0.4333 (var=0.1325)
- pi_1: 0.0311 (var=0.0543)
- pi_2: -0.6311 (var=0.1123)
- pi_3: -0.0756 (var=0.0950)
- pi_4: -0.2356 (var=0.0946)
- pi_5: -0.0533 (var=0.0979)
- pi_6: -0.0089 (var=0.1100)
- pi_7: -0.0822 (var=0.0681)
- pi_8: -0.1356 (var=0.0885)
- pi_9: -0.1400 (var=0.1382)
- pi_10: -0.0733 (var=0.0847)
- pi_11: -0.1533 (var=0.1585)
- pi_12: -0.7511 (var=0.1983)
- pi_13: -0.3733 (var=0.2028)
- pi_14: -0.4133 (var=0.1264)
- pi_16: 0.1956 (var=0.2897)
- pi_18: 0.0756 (var=0.1523)
- pi_19: -0.1378 (var=0.1104)
- pi_20: 0.0067 (var=0.1049)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify which option possesses the highest validity cue (Cue 0)
    has_cue_0_a = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    has_cue_0_b = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    # Determine if the subject chose the option with Cue 0
    chose_cue_0 = ((data['response'] == 0) & has_cue_0_a) | ((data['response'] == 1) & has_cue_0_b)
    
    # Count total cues in each option to identify trial types
    n_cues_a = data['option_a_ratings'].apply(sum)
    n_cues_b = data['option_b_ratings'].apply(sum)
    
    # Mask for 1 vs 1 cue trials (Trial 1 and 6)
    mask_1 = (n_cues_a == 1) & (n_cues_b == 1)
    # Mask for 3 vs 3 cue trials (Trial 3 and 8)
    mask_3 = (n_cues_a == 3) & (n_cues_b == 3)
    
    # Calculate the proportion of times the subject chose Cue 0 in each condition
    p1 = chose_cue_0[mask_1].mean()
    p3 = chose_cue_0[mask_3].mean()
    
    # Return the difference in choice probability
    return float(p1 - p3)
```

**Observed (real) value:** -0.0200 (var=0.0414)
**Candidate trajectory (this loop):**
  - iter 1: 0.0192 (var=0.0196) (Δ vs real +0.0392)
  - iter 2: -0.0075 (var=0.0150) (Δ vs real +0.0125)
  - iter 3: 0.0117 (var=0.0172) (Δ vs real +0.0317)
  - iter 4 (current): 0.0183 (var=0.0167) (Δ vs real +0.0383)
**Other theories' values on this metric (for reference):**
- pi_15: -0.0017 (var=0.0212)
- pi_17: 0.2200 (var=0.0285)
- pi_1: -0.0225 (var=0.0095)
- pi_2: 0.0183 (var=0.0208)
- pi_3: 0.0842 (var=0.0125)
- pi_4: -0.0075 (var=0.0178)
- pi_5: 0.0067 (var=0.0164)
- pi_6: -0.0175 (var=0.0178)
- pi_7: 0.0458 (var=0.0127)
- pi_8: 0.0000 (var=0.0215)
- pi_9: 0.0033 (var=0.0180)
- pi_10: 0.0175 (var=0.0138)
- pi_11: -0.0092 (var=0.0181)
- pi_12: -0.3775 (var=0.0298)
- pi_13: -0.1408 (var=0.0613)
- pi_14: 0.0083 (var=0.0083)
- pi_16: -0.0042 (var=0.0113)
- pi_18: 0.0133 (var=0.0113)
- pi_19: 0.0192 (var=0.0210)
- pi_20: 0.0175 (var=0.0184)

### Experiment 33
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['sig_A'] = df['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    df['sig_B'] = df['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    df['sig'] = df['sig_A'] + "_" + df['sig_B']
    
    means = df.groupby('sig')['response'].mean()
    
    t1 = "10000_01100"
    t2 = "10010_01110"
    t3 = "10011_01111"
    
    t4 = "01100_10000"
    t5 = "01110_10010"
    t6 = "01111_10011"
    
    t7 = "11000_00110"
    t8 = "11001_00111"
    
    t9 = "10100_01010"
    t10 = "10101_01011"
    
    diff = 0.0
    if t1 in means and t2 in means: diff += abs(means[t2] - means[t1])
    if t1 in means and t3 in means: diff += abs(means[t3] - means[t1])
    if t4 in means and t5 in means: diff += abs(means[t5] - means[t4])
    if t4 in means and t6 in means: diff += abs(means[t6] - means[t4])
    if t7 in means and t8 in means: diff += abs(means[t8] - means[t7])
    if t9 in means and t10 in means: diff += abs(means[t10] - means[t9])
    
    return float(diff)
```

**Observed (real) value:** 0.1067 (var=0.2133)
**Candidate trajectory (this loop):**
  - iter 1: 0.1400 (var=0.1870) (Δ vs real +0.0333)
  - iter 2: 0.1244 (var=0.1343) (Δ vs real +0.0178)
  - iter 3: 0.1111 (var=0.1377) (Δ vs real +0.0044)
  - iter 4 (current): 0.1022 (var=0.1143) (Δ vs real -0.0044)
**Other theories' values on this metric (for reference):**
- pi_17: 0.6000 (var=0.3276)
- pi_18: 0.1444 (var=0.1538)
- pi_1: 0.0822 (var=0.1247)
- pi_2: 0.1622 (var=0.1036)
- pi_3: 0.1067 (var=0.1028)
- pi_4: 0.1311 (var=0.1591)
- pi_5: 0.0889 (var=0.1205)
- pi_6: 0.2244 (var=0.0804)
- pi_7: 0.1600 (var=0.1419)
- pi_8: 0.1133 (var=0.1893)
- pi_9: 0.1222 (var=0.1497)
- pi_10: 0.1444 (var=0.1698)
- pi_11: 0.0956 (var=0.1451)
- pi_12: 0.1267 (var=0.1652)
- pi_13: 0.0756 (var=0.1101)
- pi_14: 0.0533 (var=0.1180)
- pi_15: 0.1067 (var=0.1577)
- pi_16: 0.1267 (var=0.1296)
- pi_19: 0.1067 (var=0.1936)
- pi_20: 0.2222 (var=0.1967)

### Experiment 34
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create string representation for A and B options to identify trials
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Map rows to trial types
    def get_trial(row):
        a, b = row['A_str'], row['B_str']
        if a == '10000' and b == '01100': return 1
        if a == '10010' and b == '01110': return 2
        if a == '10011' and b == '01111': return 3
        if a == '01000' and b == '00110': return 4
        if a == '11000' and b == '10110': return 5
        if a == '11001' and b == '10111': return 6
        if a == '00100' and b == '00011': return 7
        if a == '01100' and b == '01011': return 8
        return 0

    data['trial'] = data.apply(get_trial, axis=1)
    
    # Calculate P(choose A) for each trial. Response 0 means A was chosen.
    data['chose_A'] = (data['response'] == 0).astype(float)
    
    # Group by subject and trial
    subj_trial = data[data['trial'] > 0].groupby(['subject_id', 'trial'])['chose_A'].mean().unstack(fill_value=np.nan)
    
    # For each subject, compute the absolute difference in choice probabilities 
    # between the base trial and the trial with the most added shared features.
    diffs = []
    for subj in subj_trial.index:
        p = subj_trial.loc[subj]
        d1 = abs(p.get(1, np.nan) - p.get(3, np.nan))
        d2 = abs(p.get(4, np.nan) - p.get(6, np.nan))
        d3 = abs(p.get(7, np.nan) - p.get(8, np.nan))
        
        valid_d = [d for d in [d1, d2, d3] if pd.notna(d)]
        if valid_d:
            diffs.append(np.mean(valid_d))
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
```

**Observed (real) value:** 0.1022 (var=0.0050)
**Candidate trajectory (this loop):**
  - iter 1: 0.1406 (var=0.0064) (Δ vs real +0.0383)
  - iter 2: 0.1283 (var=0.0023) (Δ vs real +0.0261)
  - iter 3: 0.1494 (var=0.0042) (Δ vs real +0.0472)
  - iter 4 (current): 0.1433 (var=0.0032) (Δ vs real +0.0411)
**Other theories' values on this metric (for reference):**
- pi_18: 0.1483 (var=0.0064)
- pi_17: 0.2783 (var=0.0119)
- pi_1: 0.1111 (var=0.0060)
- pi_2: 0.0928 (var=0.0030)
- pi_3: 0.1522 (var=0.0041)
- pi_4: 0.1378 (var=0.0053)
- pi_5: 0.1322 (var=0.0040)
- pi_6: 0.1667 (var=0.0047)
- pi_7: 0.1250 (var=0.0030)
- pi_8: 0.1083 (var=0.0041)
- pi_9: 0.1333 (var=0.0032)
- pi_10: 0.1472 (var=0.0043)
- pi_11: 0.1583 (var=0.0048)
- pi_12: 0.0850 (var=0.0034)
- pi_13: 0.1222 (var=0.0060)
- pi_14: 0.1167 (var=0.0045)
- pi_15: 0.1356 (var=0.0055)
- pi_16: 0.0883 (var=0.0025)
- pi_19: 0.1422 (var=0.0046)
- pi_20: 0.1667 (var=0.0049)

### Experiment 35
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    
    t1 = (a_str == '10000') & (b_str == '01000')
    t4 = (a_str == '10111') & (b_str == '01000')
    
    if t1.sum() == 0 or t4.sum() == 0:
        return 0.0
        
    p_a_t1 = 1.0 - data.loc[t1, 'response'].mean()
    p_a_t4 = 1.0 - data.loc[t4, 'response'].mean()
    
    return float(p_a_t4 - p_a_t1)
```

**Observed (real) value:** 0.0640 (var=0.0143)
**Candidate trajectory (this loop):**
  - iter 1: 0.1740 (var=0.0487) (Δ vs real +0.1100)
  - iter 2: 0.1100 (var=0.0481) (Δ vs real +0.0460)
  - iter 3: 0.1800 (var=0.0384) (Δ vs real +0.1160)
  - iter 4 (current): 0.2360 (var=0.0443) (Δ vs real +0.1720)
**Other theories' values on this metric (for reference):**
- pi_17: -0.3560 (var=0.0849)
- pi_19: 0.0720 (var=0.0508)
- pi_1: 0.0060 (var=0.0266)
- pi_2: 0.3300 (var=0.0249)
- pi_3: 0.0480 (var=0.0213)
- pi_4: 0.1780 (var=0.0253)
- pi_5: 0.0140 (var=0.0416)
- pi_6: 0.0260 (var=0.0599)
- pi_7: 0.0060 (var=0.0374)
- pi_8: 0.3260 (var=0.0407)
- pi_9: 0.1800 (var=0.0680)
- pi_10: 0.0260 (var=0.0331)
- pi_11: 0.1960 (var=0.0468)
- pi_12: 0.6800 (var=0.0984)
- pi_13: 0.2760 (var=0.1470)
- pi_14: 0.3680 (var=0.1286)
- pi_15: 0.2800 (var=0.0328)
- pi_16: 0.0220 (var=0.2577)
- pi_18: -0.0460 (var=0.0593)
- pi_20: 0.0380 (var=0.0492)

### Experiment 36
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Response 0 means choosing Option A, so 1.0 - response is the indicator for choosing A
    p_a = 1.0 - data['response']
    
    m1 = p_a[(a_str == '10000') & (b_str == '01000')].mean()
    m2 = p_a[(a_str == '10100') & (b_str == '01000')].mean()
    m3 = p_a[(a_str == '10110') & (b_str == '01000')].mean()
    m4 = p_a[(a_str == '10111') & (b_str == '01000')].mean()
    
    # Fallback to 0.5 if any trial type is completely missing (should not happen in this design)
    m1 = 0.5 if np.isnan(m1) else m1
    m2 = 0.5 if np.isnan(m2) else m2
    m3 = 0.5 if np.isnan(m3) else m3
    m4 = 0.5 if np.isnan(m4) else m4
    
    # Compare the later trials (more unique features) to the earlier trials
    return float((m4 + m3) - (m2 + m1))
```

**Observed (real) value:** -0.0320 (var=0.0398)
**Candidate trajectory (this loop):**
  - iter 1: 0.2040 (var=0.0664) (Δ vs real +0.2360)
  - iter 2: 0.1440 (var=0.0637) (Δ vs real +0.1760)
  - iter 3: 0.2020 (var=0.0806) (Δ vs real +0.2340)
  - iter 4 (current): 0.1980 (var=0.0662) (Δ vs real +0.2300)
**Other theories' values on this metric (for reference):**
- pi_19: 0.0760 (var=0.0914)
- pi_17: -0.3280 (var=0.1464)
- pi_1: 0.0800 (var=0.0608)
- pi_2: 0.3740 (var=0.0687)
- pi_3: 0.0660 (var=0.0522)
- pi_4: 0.1600 (var=0.0336)
- pi_5: 0.0760 (var=0.0730)
- pi_6: -0.0420 (var=0.1076)
- pi_7: 0.0840 (var=0.0769)
- pi_8: 0.3600 (var=0.1004)
- pi_9: 0.1980 (var=0.1002)
- pi_10: 0.0180 (var=0.0815)
- pi_11: 0.1720 (var=0.1156)
- pi_12: 0.6960 (var=0.1520)
- pi_13: 0.4280 (var=0.1272)
- pi_14: 0.4480 (var=0.1745)
- pi_15: 0.1760 (var=0.0466)
- pi_16: 0.0080 (var=0.6215)
- pi_18: -0.1040 (var=0.1324)
- pi_20: 0.0720 (var=0.0900)

### Experiment 37
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 0))
              
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 1, 1)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 0))
              
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t4 = 1.0 - data[t4_mask]['response'].mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t4):
        return 0.0
        
    return float(p_a_t4 - p_a_t1)

```

**Observed (real) value:** 0.0092 (var=0.0153)
**Candidate trajectory (this loop):**
  - iter 1: 0.2262 (var=0.0408) (Δ vs real +0.2169)
  - iter 2: 0.0954 (var=0.0292) (Δ vs real +0.0862)
  - iter 3: 0.1754 (var=0.0426) (Δ vs real +0.1662)
  - iter 4 (current): 0.1877 (var=0.0341) (Δ vs real +0.1785)
**Other theories' values on this metric (for reference):**
- pi_17: -0.4569 (var=0.0855)
- pi_20: 0.0062 (var=0.0603)
- pi_1: 0.0031 (var=0.0125)
- pi_2: 0.3662 (var=0.0382)
- pi_3: 0.0338 (var=0.0211)
- pi_4: 0.1123 (var=0.0221)
- pi_5: 0.0631 (var=0.0435)
- pi_6: 0.0185 (var=0.0408)
- pi_7: 0.0492 (var=0.0302)
- pi_8: 0.3231 (var=0.0436)
- pi_9: 0.1877 (var=0.0798)
- pi_10: 0.0108 (var=0.0298)
- pi_11: 0.1400 (var=0.0454)
- pi_12: 0.6600 (var=0.1193)
- pi_13: 0.3785 (var=0.1200)
- pi_14: 0.4077 (var=0.1641)
- pi_15: 0.1985 (var=0.0422)
- pi_16: -0.0308 (var=0.1924)
- pi_18: -0.1523 (var=0.0430)
- pi_19: 0.0554 (var=0.0327)

### Experiment 38
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(row['option_a_ratings']) == (1, 0, 0, 0, 0) and tuple(row['option_b_ratings']) == (0, 1, 0, 0, 0)
    
    def is_t4(row):
        return tuple(row['option_a_ratings']) == (1, 0, 1, 1, 1) and tuple(row['option_b_ratings']) == (0, 1, 1, 1, 1)
        
    t1_mask = data.apply(is_t1, axis=1)
    t4_mask = data.apply(is_t4, axis=1)
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t4 = 1.0 - data[t4_mask]['response'].mean()
    
    return float(p_a_t1 - p_a_t4)
```

**Observed (real) value:** -0.0200 (var=0.0274)
**Candidate trajectory (this loop):**
  - iter 1: 0.0217 (var=0.0391) (Δ vs real +0.0417)
  - iter 2: 0.0200 (var=0.0310) (Δ vs real +0.0400)
  - iter 3: 0.0433 (var=0.0298) (Δ vs real +0.0633)
  - iter 4 (current): 0.0200 (var=0.0393) (Δ vs real +0.0400)
**Other theories' values on this metric (for reference):**
- pi_20: 0.0383 (var=0.0445)
- pi_17: 0.2467 (var=0.0405)
- pi_1: -0.0183 (var=0.0156)
- pi_2: 0.0217 (var=0.0372)
- pi_3: 0.0117 (var=0.0169)
- pi_4: 0.0283 (var=0.0288)
- pi_5: -0.0017 (var=0.0293)
- pi_6: 0.0450 (var=0.0292)
- pi_7: 0.0183 (var=0.0259)
- pi_8: 0.0500 (var=0.0444)
- pi_9: 0.0017 (var=0.0396)
- pi_10: -0.0383 (var=0.0264)
- pi_11: 0.0150 (var=0.0485)
- pi_12: -0.0017 (var=0.0235)
- pi_13: 0.0250 (var=0.0159)
- pi_14: -0.0117 (var=0.0144)
- pi_15: -0.0183 (var=0.0276)
- pi_16: 0.0267 (var=0.0165)
- pi_18: 0.0117 (var=0.0283)
- pi_19: -0.0150 (var=0.0421)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The Strategy Mixture candidate successfully establishes a new baseline by avoiding the aggressive self-dilution of the Lateral Inhibition model. However, it exhibits catastrophic failures (predicting the opposite sign of the empirical effect) on several key experiments, notably Experiments 11, 12, 29, 30, and 31. These experiments heavily probe conflict scenarios where the compensatory response should be sensitive to the *relative validities* of secondary cues. Pure Unit-Weight Tallying completely ignores these validities, forcing the model to either behave like pure TTB or pure unweighted counting. Consequently, when TTB is incorrect but subjects use a weighted compensatory approach, the model guesses blindly or leans the wrong way.
Rationale: The Strategy Mixture framework is correct and should be maintained, but the compensatory component needs a minor adjustment. Replace pure Unit-Weight Tallying with a Weighted Additive (WADD) strategy (using the provided cue validities as weights) for the compensatory heuristic. This will allow the compensatory arm to respect the hierarchy of secondary cues. Additionally, consider modulating the mixture weight `w` by the *difference between the top two cue validities* (or the max validity) rather than the global standard deviation, as local dispersion at the top of the cue hierarchy is often a stronger driver of TTB adoption.

**Outcome of your advice:** iter 1 candidate loss=0.3958 -> iter 2 candidate loss=0.4079 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The gate rejected the recent candidate, indicating that replacing Unit-Weight Tallying with a Weighted Additive (WADD) strategy and modulating the mixture weight by the top-cue validity difference worsened the overall fit. The accepted base (Iteration 1) successfully captures many basic effects but fails on conflict experiments (e.g., 11, 12, 29, 30) where TTB and Tallying make opposing predictions. The failure of WADD suggests that subjects' compensatory behavior is indeed better captured by unweighted counting (Tallying) rather than tracking exact validities, but the rule for choosing *between* TTB and Tallying in the accepted base was miscalibrated.
Rationale: Since the WADD + top-cue difference approach was rejected, we must build on the accepted Iteration 1 base (Unit-Weight Tallying). Instead of complicating the compensatory arm, simplify the Strategy Mixture. Remove the validity-dispersion modulation on the mixture weight entirely, as it may be overfitting or misaligning with subject behavior. Let the probability of using TTB (`w_ttb`) be a pure, unmodulated free parameter. To help with conflict trials, ensure that the Tallying evidence is the strict sum of positive cues, and that both TTB and Tallying use independent softmax inverse temperatures (`beta_ttb` and `beta_tally`). By relying on standard Unit-Weight Tallying and a fixed mixture weight, the model can leverage between-subject variance in `w_ttb` to explain why some subjects follow TTB while others follow Tallying in conflict scenarios, without forcing a rigid validity-based modulation.

**Outcome of your advice:** iter 2 candidate loss=0.4079 -> iter 3 candidate loss=0.4078 -> the gate marked it REJECTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The programmatic gate rejected the recent candidates, meaning the move to a pure, unmodulated mixture (Iteration 3) and the move to WADD (Iteration 2) both worsened the fit compared to the Iteration 1 base. The Iteration 1 base, which likely included some form of validity-dispersion modulation on the mixture weight, remains the running best. However, it still fails on conflict experiments (e.g., 11, 12, 29, 30) where the compensatory and non-compensatory heuristics strongly disagree. The failure of WADD indicates that subjects do not precisely weight secondary cues by their exact validities, supporting simple Unit-Weight Tallying. The failure of the pure mixture indicates that the choice of strategy *is* context-dependent and modulated by the environment (e.g., validity dispersion).
Rationale: Since both the pure mixture and WADD variants were rejected, the proposer should build on the unchanged Iteration 1 base. Keep the compensatory arm as Unit-Weight Tallying (which counts positive cues), but refine the non-compensatory TTB arm and the mixture rule. Specifically, instead of TTB outputting a hard [1.0, 0.0] evidence vector, scale the TTB evidence by the validity of the discriminating cue (e.g., `ev_ttb = np.array([val[i], 0.0])`). This allows the confidence of the TTB strategy to naturally scale with the cue's reliability. Second, reinstate the validity-dispersion modulation for the mixture weight `w` (as it was in Iteration 1), but introduce a free parameter to control the sensitivity or threshold of this modulation (e.g., a logistic function over the dispersion or max validity difference). This keeps the model within the prescribed Strategy Mixture family while giving it the flexibility needed to capture context-dependent strategy shifts.

**Outcome of your advice:** iter 3 candidate loss=0.4078 -> CURRENT candidate loss=0.3823 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate was ACCEPTED, validating the addition of validity-scaled TTB evidence and logistic modulation. The loss improved to 0.3823, solidifying this Strategy Mixture as the running best. However, massive discrepancies remain in conflict experiments (e.g., Exps 11, 12, 29, 30) where the model's predictions are completely flipped compared to human data. The current implementation modulates the mixture weight `w` globally per experiment based on the standard deviation of validities. While this captures some environment-level variance, it fails to capture trial-level strategy shifts. Humans likely fall back to a compensatory Tallying strategy on a trial-by-trial basis if the first discriminating cue has low validity, but rely on TTB if the discriminating cue is highly reliable.",
  "verdict": "regenerate",
  "rationale": "Build on this accepted Strategy Mixture base, but refine the mixture weight `w` to be trial-specific rather than environment-global. Specifically, modulate the logistic mixture weight `w` using the *validity of the first discriminating cue* on that specific trial, instead of the standard deviation of all validities. For example: `logit_w = w_base + gamma * val_discrim` (where `val_discrim` is the validity of the cue that triggered the TTB decision). If no cues discriminate, default to Tallying. This minor in-family adjustment allows the model to dynamically shift between non-compensatory and compensatory heuristics trial-by-trial based on the reliability of the immediate evidence, which should help resolve the flipped signs in conflict experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 53217,
  "candidates_token_count": 354,
  "total_token_count": 56187
}
```
