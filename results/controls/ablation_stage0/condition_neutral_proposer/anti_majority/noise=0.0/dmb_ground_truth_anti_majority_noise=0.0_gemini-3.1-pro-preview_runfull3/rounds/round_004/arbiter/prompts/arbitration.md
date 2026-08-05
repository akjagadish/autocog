# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Decision-makers use an adaptive toolbox of strategies, dynamically selecting between Take The Best (TTB) and Weighted Additive (WADD) evaluation on a trial-by-trial basis. The probability of using TTB over WADD depends on the conflict between the cues: when the top discriminating cue is much stronger than the opposing cues, TTB dominates; when multiple opposing cues rival the top cue's validity, the decision-maker is more likely to switch to the compensatory WADD strategy. By applying independent scaling weights to the top cue's validity and the sum of opposing validities, the model can penalize a large number of weak opposing cues, keeping TTB dominant in those cases while still allowing WADD to differentiate between strong compensatory profiles.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    v_top = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            v_top = val[j]
            break
        if b[j] > a[j]:
            winner_ttb = 1
            v_top = val[j]
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
        p_ttb_weight = 0.5
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
        # Dynamic mixture weight based on conflict
        v_opp_sum = 0.0
        for j in range(len(val)):
            if winner_ttb == 0 and b[j] > a[j]:
                v_opp_sum += val[j]
            elif winner_ttb == 1 and a[j] > b[j]:
                v_opp_sum += val[j]
                
        theta_top = float(parameters["theta_top"])
        theta_opp = float(parameters["theta_opp"])
        bias_ttb = float(parameters["bias_ttb"])
        z_mix = theta_top * v_top - theta_opp * v_opp_sum + bias_ttb
        p_ttb_weight = 1.0 / (1.0 + np.exp(-z_mix))
        
    # --- Weighted Additive (WADD) ---
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores_wadd = np.array([score_a, score_b])
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # --- Strategy Mixture ---
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_wadd_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Decision-makers evaluate options using a combination of the direct, unscaled linear sum of the mean-centered cue validities (Linear WADD) and a proportional tally of the cues. By centering validities around their mean rather than a fixed neutral point like 0.5, cues with below-average validity are treated as negative evidence (a dilution effect), meaning that adding weak cues to an option can actually decrease its overall evaluation. The tallying count is normalized into a proportion to put both strategies on a comparable numerical scale, allowing the mixing parameter to effectively balance the influence of the highest-validity cues against the sheer quantity of positive cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_tally = float(parameters["w_tally"])
    
    # Linear WADD scores (mean-centered validities to capture dilution effects)
    centered_val = val - np.mean(val)
    wadd_a = np.sum(centered_val * a)
    wadd_b = np.sum(centered_val * b)
    
    # Tallying scores (proportion of positive features to fix scaling mismatch)
    tally_a = np.mean(a)
    tally_b = np.mean(b)
    
    # Combine scores
    score_a = (1.0 - w_tally) * wadd_a + w_tally * tally_a
    score_b = (1.0 - w_tally) * wadd_b + w_tally * tally_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Lapse noise
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 9: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 12: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This experiment is designed to maximally dissociate THEORY 1 (TTB/WADD mixture) from THEORY 2 (Linear WADD with mean-centered validities + Tallying). THEORY 2 evaluates options using mean-centered validities, meaning that any feature with a validity below the average validity acts as *negative* evidence (a dilution effect). In contrast, THEORY 1 uses strictly positive validities (both in TTB and its WADD component), meaning that adding *any* positive feature to an option strictly increases its evaluation. By creating pairs where Option B is identical to Option A but with additional low-validity features, THEORY 1 will strictly prefer Option B (both TTB and WADD see positive evidence), whereas THEORY 2's WADD component will penalize Option B for these below-average features. The inclusion of Tallying in THEORY 2 ensures that the model can be fit to different degrees of this dilution effect, but the qualitative divergence on these 'added weak feature' trials provides a stark quantitative dissociation.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times subjects choose Option B when it strictly dominates Option A by containing all of A's positive features plus additional, lower-validity features. Theory 1 (TTB/WADD mixture with positive validities) predicts a very high probability of choosing B, as the added features represent strictly positive evidence. In contrast, Theory 2 (Linear WADD with mean-centered validities) exhibits a dilution effect: the added lower-validity features have below-average validities, acting as negative evidence in the Linear WADD component. Thus, Theory 2 predicts a significantly lower probability of choosing B on these specific trials.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option B has all the positive features of Option A,
    # plus additional features (specifically features 4 and 5 in the design).
    is_subset = [
        all(b_val >= a_val for a_val, b_val in zip(a, b)) and sum(b) > sum(a)
        for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])
    ]
    
    subset_data = data[is_subset]
    if len(subset_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B was chosen in these trials.
    return float(subset_data['response'].mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.8325 (var=0.0137)
- Predicted under pi_6 (simulated): 0.6008 (var=0.0458)
- Observed on real data: 0.1658 (var=0.0159)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]

**Rationale:** To maximally distinguish THEORY 1 (mean-centered WADD + Tallying) from THEORY 2 (adaptive TTB/WADD mixture), we exploit the 'dilution effect' inherent to THEORY 1's mean-centered validities. Features with below-average validity act as negative evidence in THEORY 1's WADD component. We include trials where Option A and Option B share a high-validity feature, but Option B has additional below-average validities. THEORY 2 will strictly prefer B (as both TTB and WADD treat all features as positive evidence), whereas THEORY 1 can strongly prefer A due to the negative weight of the weak features. We also include conflict trials to test THEORY 2's dynamic strategy switching.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'dominance' trials where Option B has all the positive features of Option A, plus additional features (specifically features 4, 5, and 6). According to Theory 2 (TTB + WADD), all validities are positive, so Option B strictly dominates Option A; the model will almost never choose A. However, Theory 1 uses mean-centered validities. Because features 4, 5, and 6 have below-average validity, they act as negative evidence (the dilution effect), lowering Option B's WADD score below Option A's. Thus, Theory 1 predicts a significantly higher rate of choosing the 'dominated' Option A than Theory 2.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_dominance(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # B dominates A if B has all of A's cues plus more
        return all(b[i] >= a[i] for i in range(len(a))) and sum(b) > sum(a)
    
    is_target = data.apply(is_dominance, axis=1)
    target_data = data[is_target]
    if len(target_data) == 0:
        return 0.0
    
    # Return the proportion of times the subject chose Option A (response == 0)
    return float((target_data['response'] == 0).mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.1939 (var=0.0119)
- Predicted under pi_6 (simulated): 0.4150 (var=0.0765)
- Observed on real data: 0.8733 (var=0.0162)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner == resp:
            ttb_consistent += 1
            
    return float(ttb_consistent / len(data))
```

**Observed (real) value:** 0.6533 (var=0.0443)
**Predicted under pi_5:** 0.6344 (var=0.0556)
**Predicted under pi_6:** 0.5040 (var=0.0439)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        if row['response'] == tally_choice:
            tallying_consistent += 1
        total += 1
        
    if total == 0:
        return 0.5
    return float(tallying_consistent / total)
```

**Observed (real) value:** 0.3696 (var=0.0477)
**Predicted under pi_5:** 0.3658 (var=0.0568)
**Predicted under pi_6:** 0.4633 (var=0.0308)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4021 (var=0.0113)
**Predicted under pi_5:** 0.3458 (var=0.0270)
**Predicted under pi_6:** 0.4794 (var=0.0228)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In all 8 trials of this design, Option A is constructed to be the Take-The-Best (TTB) choice
    # (it always possesses the highest-validity discriminating cue). Option B is the compensatory
    # choice (it possesses multiple lower-validity cues).
    # This metric simply calculates the overall proportion of times Option B was chosen.
    # TTB will only choose B due to lapse noise (epsilon), keeping this metric low.
    # WADD will choose B systematically when gamma is low enough to allow compensation,
    # leading to a significantly higher mean.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.2479 (var=0.0104)
**Predicted under pi_5:** 0.4015 (var=0.0712)
**Predicted under pi_6:** 0.5177 (var=0.0431)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project lists to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 6: A has cue 1 (0.95), B has cue 2 (0.85)
    t6_A = (1, 0, 0, 0, 0)
    t6_B = (0, 1, 0, 0, 0)
    
    # Trial 8: A has cue 3 (0.75), B has cue 4 (0.65)
    t8_A = (0, 0, 1, 0, 0)
    t8_B = (0, 0, 0, 1, 0)
    
    mask_6 = (a_tuples == t6_A) & (b_tuples == t6_B)
    mask_8 = (a_tuples == t8_A) & (b_tuples == t8_B)
    
    if not mask_6.any() or not mask_8.any():
        return 0.0
        
    # Calculate probability of choosing A (response == 0)
    pA_t6 = (data.loc[mask_6, 'response'] == 0).mean()
    pA_t8 = (data.loc[mask_8, 'response'] == 0).mean()
    
    return float(pA_t6 - pA_t8)
```

**Observed (real) value:** 0.0133 (var=0.0126)
**Predicted under pi_5:** 0.0433 (var=0.0406)
**Predicted under pi_6:** -0.0033 (var=0.0439)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create a hashable surrogate for the trial type using Option B's ratings
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the mean response (probability of choosing B) for each trial type
    trial_means = data.groupby(b_str)['response'].mean()
    
    # Calculate the standard deviation of these means
    return float(np.std(trial_means))
```

**Observed (real) value:** 0.1327 (var=0.0146)
**Predicted under pi_5:** 0.0444 (var=0.0031)
**Predicted under pi_6:** 0.0948 (var=0.0032)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # 1v1 trials where B has its strongest possible opposing cue
        m_t1 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,0,0,0))
        m_t5 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,1,0,0))
        m_t8 = (a_tuples == (0,0,1,0,0)) & (b_tuples == (0,0,0,1,0))
        
        # 1v2 trials where B gets the NEXT strongest possible cue added
        m_t4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,1,0,0))
        m_t7 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,1,1,0))
        m_t9 = (a_tuples == (0,0,1,0,0)) & (b_tuples == (0,0,0,1,1))
        
        mask_1v1 = m_t1 | m_t5 | m_t8
        mask_1v2 = m_t4 | m_t7 | m_t9
        
        df_1v1 = subj_df[mask_1v1]
        df_1v2 = subj_df[mask_1v2]
        
        if len(df_1v1) > 0 and len(df_1v2) > 0:
            p_1v1 = (df_1v1['response'] == 0).mean()
            p_1v2 = (df_1v2['response'] == 0).mean()
            diffs.append(p_1v1 - p_1v2)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
```

**Observed (real) value:** -0.6222 (var=0.0329)
**Predicted under pi_5:** 0.1281 (var=0.0547)
**Predicted under pi_6:** 0.1030 (var=0.0369)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # T2: A=[1, 0, 0, 0, 0, 0], B=[0, 0, 0, 1, 1, 1]
    t2_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1, 1))
    
    # T7: A=[1, 1, 0, 0, 0, 0], B=[0, 0, 1, 1, 1, 1]
    t7_mask = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1, 1))
              
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    p_a_t7 = 1.0 - data.loc[t7_mask, 'response'].mean()
    
    if pd.isna(p_a_t2) or pd.isna(p_a_t7):
        return 0.0
        
    return float(p_a_t7 - p_a_t2)
```

**Observed (real) value:** 0.1050 (var=0.1084)
**Predicted under pi_5:** -0.0475 (var=0.0674)
**Predicted under pi_6:** 0.0050 (var=0.0600)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories completely fail to capture the massive 'dilution effect' observed in the data. In Experiments 1 and 2, when Option B contains all the positive features of Option A plus additional lower-validity features, subjects overwhelmingly choose Option A (e.g., 84% of the time in Exp 1, 87% of the time in Exp 2). Both theories predict subjects will prefer Option B, either because they treat all features as positive evidence (Theory 1) or because the negative evidence from mean-centering isn't strong enough to overcome the tallying and positive weights (Theory 2). The data strongly suggests that decision-makers are not adding evidence, but rather *averaging* it. If an option's value is the average of its active cues' validities, adding a weak cue to a strong cue reduces the overall value.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Cue Averaging' theory. Instead of summing validities (WADD) or using a lexicographic rule (TTB), decision-makers evaluate each option by taking the average of the validities of its present (positive) features. If an option has a single cue with 0.95 validity, its value is 0.95. If another option has that same cue plus a 0.55 validity cue, its value drops to (0.95 + 0.55)/2 = 0.75. This perfectly explains the strong preference for options with fewer, higher-validity cues over options that add lower-validity cues (the strong dilution effect seen across multiple experiments)."
}
```

## Usage

```json
{
  "prompt_token_count": 10219,
  "candidates_token_count": 370,
  "total_token_count": 11414
}
```
