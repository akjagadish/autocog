# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_17" and "pi_21") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_17" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_21" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_17
Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_21
Strategy Mixture with Validity-Scaled TTB, Mean-Normalized Tallying, and Logistic Dispersion Modulation

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
            
    # Tallying Strategy (Unit-Weight Additive) - normalized by total cues to match scale
    ev_tally = np.array([np.mean(a), np.mean(b)])
    
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


## EXPERIMENT 1 (proposed by pi_17)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Strategy Mixture (Competing) model by exploiting two core structural differences: the 'perverse' self-dilution effect and invariance to symmetric evidence addition. In the Competing model, choice probability is a mixture of TTB (based on the highest validity discriminating cue) and Mean-Normalized Tallying. Adding unique lower-validity cues to Option A strictly increases its tallying score without changing the TTB winner, forcing the Competing model to predict a monotonic increase in preference for A. In contrast, the Advocated model employs divisive normalization; adding lower-validity cues to A dilutes its primary high-validity cue, which can paradoxically decrease A's choice probability. Furthermore, adding shared cues symmetrically to both options leaves the TTB winner and the tallying difference strictly unchanged in the Competing model, mandating identical choice probabilities. The Advocated model, however, processes these shared cues holistically, increasing both self and conflict suppression, thus predicting systematic shifts in preference.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric directly tests the 'perverse self-dilution' effect versus monotonic evidence accumulation. Between trial 1 and trial 4, Option A gains additional unique lower-validity cues while Option B remains unchanged. The Competing Theory (Strategy Mixture) predicts a strict increase in the probability of choosing Option A, because the TTB winner is unchanged and the Tallying score for Option A strictly increases. Therefore, the mean response (where 1 = choosing B) should decrease from trial 1 to trial 4. In contrast, the Advocated Theory predicts that adding lower-validity cues to Option A dilutes the impact of its primary high-validity cue via divisive normalization, potentially decreasing the probability of choosing A and thus increasing the mean response. The difference in mean response between trial 4 and trial 1 directly captures this opposing directional prediction.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 0, 0, 0))
    t4_mask = (a_tuples == (1, 0, 1, 1, 1)) & (b_tuples == (0, 1, 0, 0, 0))
    
    t1_mean = data.loc[t1_mask, 'response'].mean()
    t4_mean = data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(t1_mean) or pd.isna(t4_mean):
        return 0.0
        
    return float(t4_mean - t1_mean)

### RESULTS
- Predicted under pi_17 (simulated): 0.3754 (var=0.0837)
- Predicted under pi_21 (simulated): -0.1554 (var=0.0439)
- Observed on real data: 0.0246 (var=0.0235)

## EXPERIMENT 2 (proposed by pi_21)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=15):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 10: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 12: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 15: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Strategy Mixture (Advocated) model from the Lateral Inhibition Evidence Accumulator (Competing) model by exploiting two distinct structural properties: strict invariance to shared features and strict monotonicity of unique features. In the Advocated Strategy Mixture model, choice probability is derived from a linear combination of a TTB component (based solely on the highest-validity discriminating cue) and a Mean-Normalized Tallying component. Because the softmax function operates on the difference in means, adding shared cues to both options perfectly preserves both the TTB winner and the tallying difference, forcing the Advocated model to predict identical choice probabilities across these trials. Furthermore, adding unique lower-validity cues to an option strictly increases its tally mean, forcing a monotonic increase in its choice probability. In stark contrast, the Competing Lateral Inhibition model employs divisive normalization. Adding shared cues increases both self-dilution and conflict suppression, breaking the invariance and systematically shifting choice probabilities. Moreover, adding unique lower-validity cues to an option dilutes the perceived value of its primary high-validity cue, which can perversely decrease its overall choice probability (a non-monotonic reversal mathematically impossible for the Advocated model).

**Computed schedule:** 15 unique pairs × 6 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the change in the probability of choosing Option A when minor supporting features are added to it. According to the Advocated theory (Strategy Mixture), adding unique supporting cues to A strictly increases its tallying value without changing the Take-The-Best winner, leading to a monotonic increase in the probability of choosing A (metric > 0). In contrast, the Competing theory (Lateral Inhibition) predicts that adding minor cues triggers divisive normalization, which dilutes the perceived value of A's primary, highest-validity cue. This causes a paradoxical decrease in the probability of choosing A (metric < 0).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    # Base trial: A has only the best cue, B has the second best
    t1_mask = (a_tup == (1, 0, 0, 0, 0)) & (b_tup == (0, 1, 0, 0, 0))
    # Enriched trial: A has the best cue plus three minor cues, B remains the same
    t15_mask = (a_tup == (1, 0, 1, 1, 1)) & (b_tup == (0, 1, 0, 0, 0))
    
    p1 = (data[t1_mask]['response'] == 0).mean()
    p15 = (data[t15_mask]['response'] == 0).mean()
    
    if pd.isna(p1): p1 = 0.5
    if pd.isna(p15): p15 = 0.5
        
    return float(p15 - p1)

### RESULTS
- Predicted under pi_17 (simulated): -0.3117 (var=0.1319)
- Predicted under pi_21 (simulated): 0.1750 (var=0.0534)
- Observed on real data: 0.0200 (var=0.0163)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_17:** 0.7825 (var=0.0205)
**Predicted under pi_21:** 0.5550 (var=0.0780)

### Experiment 4
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
**Predicted under pi_17:** 0.3364 (var=0.0082)
**Predicted under pi_21:** 0.3850 (var=0.0733)

### Experiment 5
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
**Predicted under pi_17:** 0.6192 (var=0.0061)
**Predicted under pi_21:** 0.5877 (var=0.0525)

### Experiment 6
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
**Predicted under pi_17:** 0.6208 (var=0.0059)
**Predicted under pi_21:** 0.6142 (var=0.0852)

### Experiment 7
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
**Predicted under pi_17:** -0.3162 (var=0.0979)
**Predicted under pi_21:** 0.0112 (var=0.0396)

### Experiment 8
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
**Predicted under pi_17:** 0.0783 (var=0.0222)
**Predicted under pi_21:** -0.0122 (var=0.0053)

### Experiment 9
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
**Predicted under pi_17:** -0.4150 (var=0.1487)
**Predicted under pi_21:** 0.0363 (var=0.0256)

### Experiment 10
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
**Predicted under pi_17:** 1.5772 (var=0.0251)
**Predicted under pi_21:** 1.7130 (var=0.0281)

### Experiment 11
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
**Predicted under pi_17:** 0.8947 (var=0.0400)
**Predicted under pi_21:** 0.8463 (var=0.0423)

### Experiment 12
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
**Predicted under pi_17:** 0.2047 (var=0.1812)
**Predicted under pi_21:** 0.1125 (var=0.2241)

### Experiment 13
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
**Predicted under pi_17:** -0.2081 (var=0.0388)
**Predicted under pi_21:** 0.0600 (var=0.0329)

### Experiment 14
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
**Predicted under pi_17:** -0.2833 (var=0.0318)
**Predicted under pi_21:** 0.2011 (var=0.0629)

### Experiment 15
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
**Predicted under pi_17:** 0.5478 (var=0.0097)
**Predicted under pi_21:** 0.6858 (var=0.0329)

### Experiment 16
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
**Predicted under pi_17:** 0.5283 (var=0.0146)
**Predicted under pi_21:** 0.6571 (var=0.0407)

### Experiment 17
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
**Predicted under pi_17:** 0.8215 (var=0.0185)
**Predicted under pi_21:** 0.4877 (var=0.1283)

### Experiment 18
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
**Predicted under pi_17:** 0.0974 (var=0.0035)
**Predicted under pi_21:** 0.1842 (var=0.0239)

### Experiment 19
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
**Predicted under pi_17:** 0.7708 (var=0.0132)
**Predicted under pi_21:** 0.7183 (var=0.0303)

### Experiment 20
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
**Predicted under pi_17:** 0.7392 (var=0.0154)
**Predicted under pi_21:** 0.7025 (var=0.0450)

### Experiment 21
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
**Predicted under pi_17:** 0.6858 (var=0.0124)
**Predicted under pi_21:** 0.7153 (var=0.0422)

### Experiment 22
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
**Predicted under pi_17:** 0.2375 (var=0.0603)
**Predicted under pi_21:** -0.0600 (var=0.0292)

### Experiment 23
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
**Predicted under pi_17:** 0.5837 (var=0.0119)
**Predicted under pi_21:** 0.4994 (var=0.0030)

### Experiment 24
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
**Predicted under pi_17:** 0.4779 (var=0.0167)
**Predicted under pi_21:** 0.3589 (var=0.0376)

### Experiment 25
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
**Predicted under pi_17:** 0.1263 (var=0.0068)
**Predicted under pi_21:** 0.1983 (var=0.0250)

### Experiment 26
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
**Predicted under pi_17:** 0.1347 (var=0.0051)
**Predicted under pi_21:** 0.2526 (var=0.0226)

### Experiment 27
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
**Predicted under pi_17:** 0.0182 (var=0.0006)
**Predicted under pi_21:** -0.0007 (var=0.0001)

### Experiment 28
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
**Predicted under pi_17:** -0.0954 (var=0.1085)
**Predicted under pi_21:** -0.0154 (var=0.0514)

### Experiment 29
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
**Predicted under pi_17:** 0.7755 (var=0.0316)
**Predicted under pi_21:** 0.4140 (var=0.1086)

### Experiment 30
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
**Predicted under pi_17:** 0.6400 (var=0.0077)
**Predicted under pi_21:** 0.8283 (var=0.0132)

### Experiment 31
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
**Predicted under pi_17:** -0.1767 (var=0.0234)
**Predicted under pi_21:** 0.2069 (var=0.0742)

### Experiment 32
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
**Predicted under pi_17:** -0.2677 (var=0.0576)
**Predicted under pi_21:** 0.2015 (var=0.1022)

### Experiment 33
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
**Predicted under pi_17:** 0.5044 (var=0.2465)
**Predicted under pi_21:** -0.1822 (var=0.1540)

### Experiment 34
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
**Predicted under pi_17:** 0.2200 (var=0.0285)
**Predicted under pi_21:** 0.0000 (var=0.0149)

### Experiment 35
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
**Predicted under pi_17:** 0.6000 (var=0.3276)
**Predicted under pi_21:** 0.1089 (var=0.1544)

### Experiment 36
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
**Predicted under pi_17:** 0.2783 (var=0.0119)
**Predicted under pi_21:** 0.1128 (var=0.0038)

### Experiment 37
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
**Predicted under pi_17:** -0.3560 (var=0.0849)
**Predicted under pi_21:** 0.1340 (var=0.0514)

### Experiment 38
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
**Predicted under pi_17:** -0.3280 (var=0.1464)
**Predicted under pi_21:** 0.2780 (var=0.0993)

### Experiment 39
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
**Predicted under pi_17:** -0.4569 (var=0.0855)
**Predicted under pi_21:** 0.1431 (var=0.0594)

### Experiment 40
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
**Predicted under pi_17:** 0.2467 (var=0.0405)
**Predicted under pi_21:** -0.0167 (var=0.0297)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple recent experiments (1, 2, 37, 38, 39), the data consistently shows that adding minor, lower-validity cues to an option already favored by the highest-validity cue results in almost zero change in choice probability. THEORY 1 (pi_17) predicts a strong negative shift (perverse dilution), which is entirely falsified by the near-zero observed metrics. THEORY 2 (pi_21) predicts a strong positive shift due to its Tallying component, which is also incorrect, though it correctly captures invariance to shared features (Experiment 40). The profound lack of sensitivity to minor cues strongly implies that subjects are employing a strict non-compensatory strategy, ignoring secondary information once a primary discriminating cue is found.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strict Sequential Search (Take-The-Best)' theory. The data heavily rejects the 'Extreme Dilution' hypothesis, as well as compensatory tallying. The new theory should model choices as a strict lexicographic process where subjects evaluate cues in descending order of validity and stop search immediately at the first discriminating cue, making their choice based solely on that cue. To account for noise, include a simple lapse rate or a probability of executing the search incorrectly, but remove any compensatory integration or lateral inhibition mechanisms."
}
```

## Usage

```json
{
  "prompt_token_count": 31558,
  "candidates_token_count": 305,
  "total_token_count": 33080
}
```
