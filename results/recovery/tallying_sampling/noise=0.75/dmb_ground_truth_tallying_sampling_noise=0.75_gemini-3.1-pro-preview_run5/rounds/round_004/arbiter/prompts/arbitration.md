# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## THEORY 2 — pi_6
Tallying with Conditional Take-The-Best (TTB) Fallback: Decision makers primarily rely on a compensatory, equal-weight heuristic (Tallying). However, when the options are difficult to distinguish based on tallies alone (i.e., when the tallies are tied or very close), subjects fall back on a non-compensatory strategy, checking the single most valid discriminating cue (Take-The-Best). Importantly, this fallback is weak, meaning that even when tallies are tied, subjects often guess rather than deterministically applying TTB. This predicts localized, minor validity-driven asymmetries specifically when tallies fail to provide a strong signal, avoiding the overprediction of global validity effects characteristic of full WADD models.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Strategy 1: Tallying
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    diff = abs(tally_a - tally_b)
    
    # Strategy 2: Take-The-Best (TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    ttb_a = 0.5
    ttb_b = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_a, ttb_b = 1.0, 0.0
            break
        elif b[j] > a[j]:
            ttb_a, ttb_b = 0.0, 1.0
            break
            
    # Conditional reliance on TTB based on tally closeness
    threshold = float(parameters["threshold"])
    if diff <= threshold:
        w_ttb = float(parameters["w_close"])
    else:
        w_ttb = float(parameters["w_far"])
        
    # Softmax for Tallying
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally /= np.sum(p_tally)
    
    # Softmax for TTB
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    z_ttb -= np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb /= np.sum(p_ttb)
    
    # Mixture
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate pure Tallying from Tallying with a Conditional Take-The-Best (TTB) Fallback, we focus on trials where the tally difference is zero or small, but the highest-validity discriminating cue strongly points in a specific direction. Pure Tallying strictly evaluates the options based on the difference in the raw count of positive features, predicting exactly equal preference (50/50) when tallies are tied, regardless of cue validities. The Competing Theory posits that when tallies are tied or close (difference <= threshold), subjects fall back on TTB. By including 'tally-tied' trials where the most valid cue favors one option, and 'tally-close' trials where the most valid cue opposes the tally winner, we can detect the localized validity-driven asymmetries predicted by the Competing Theory. If pure Tallying is correct, choice probabilities will depend only on the tally differences and remain completely blind to the identity of the highest-validity cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
To cleanly dissociate Pure Tallying from TTB Fallback, we examine trials where the tallies are not tied, but are close (difference of 1 or 2). We split these trials into 'Aligned' (where the highest-validity cue points in the same direction as the tally winner) and 'Opposed' (where the highest-validity cue points to the tally loser). By computing the difference in the rate of tally-consistent choices between Aligned and Opposed trials—separately for diff=1 and diff=2 to perfectly control for baseline probabilities—Pure Tallying will predict a difference of exactly zero, as it is blind to cue identity. The TTB Fallback theory predicts a positive difference, as TTB acts as a secondary force that boosts the tally winner in Aligned trials and hurts the tally winner in Opposed trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_arr = np.array(data['option_a_ratings'].tolist())
    b_arr = np.array(data['option_b_ratings'].tolist())
    
    sum_a = a_arr.sum(axis=1)
    sum_b = b_arr.sum(axis=1)
    diffs = np.abs(sum_a - sum_b)
    resp = data['response'].values
    
    effects = []
    # Evaluate for each magnitude of tally difference separately to control for diff-based probabilities
    for d_val in [1, 2]:
        mask = diffs == d_val
        if not np.any(mask):
            continue
            
        a_sub = a_arr[mask]
        b_sub = b_arr[mask]
        sum_a_sub = sum_a[mask]
        sum_b_sub = sum_b[mask]
        resp_sub = resp[mask]
        
        aligned = []
        opposed = []
        
        for i in range(len(a_sub)):
            tally_favors_a = sum_a_sub[i] > sum_b_sub[i]
            
            # Determine TTB prediction (first differing cue)
            d_vec = a_sub[i] - b_sub[i]
            nonzero = np.nonzero(d_vec)[0]
            if len(nonzero) == 0:
                continue
            ttb_favors_a = d_vec[nonzero[0]] > 0
            
            chose_a = (resp_sub[i] == 0)
            tally_consistent = float((tally_favors_a and chose_a) or (not tally_favors_a and not chose_a))
            
            if tally_favors_a == ttb_favors_a:
                aligned.append(tally_consistent)
            else:
                opposed.append(tally_consistent)
                
        if len(aligned) > 0 and len(opposed) > 0:
            effects.append(np.mean(aligned) - np.mean(opposed))
            
    if len(effects) == 0:
        return 0.0
    return float(np.mean(effects))

### RESULTS
- Predicted under pi_3 (simulated): 0.0033 (var=0.0137)
- Predicted under pi_6 (simulated): 0.1371 (var=0.0271)
- Observed on real data: 0.0358 (var=0.0142)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** To uniquely dissociate the Advocated Theory (Tallying with Conditional TTB Fallback) from the Competing Theory (pure Tallying), we exploit trials where the tally difference is either zero or one, while the highest-validity discriminating cue strongly points in a specific direction. Pure Tallying strictly evaluates options based on the difference in the raw count of positive features, predicting exactly equal preference (50/50) when tallies are tied, regardless of cue validities. The Advocated Theory posits that when tallies are tied or close, subjects fall back on TTB. By including 'tally-tied' trials where the most valid cue favors one option, and 'tally-close' trials (diff = 1) where the most valid cue opposes the tally winner, we can detect the localized validity-driven asymmetries predicted by the Advocated Theory. If pure Tallying is correct, choice probabilities will depend only on the tally differences and remain completely blind to the identity of the highest-validity cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Pure Tallying predicts that choice probabilities are strictly a function of the tally difference. Because the log-odds scale linearly with the tally difference, the tally-winner choice rate on diff=2 trials (m2) is tightly coupled to the rate on diff=1 trials (m1). Specifically, 1.5*m1 - m2 is nearly invariant across all Tallying parameters (evaluating to ~ -0.23), and TTB match on diff=0 (m0) is strictly 0.5, resulting in a highly stable baseline metric of ~0.73. In contrast, the Advocated theory posits a Conditional TTB fallback. This fallback naturally increases m0 above 0.5. Furthermore, because TTB fallback opposes the tally winner on all diff=1 and diff=2 trials in this design, and because the tallying softmax is often saturated (high beta_tally), m2 is heavily suppressed relative to m1. This breaks the Pure Tallying invariant and pushes 1.5*m1 - m2 significantly higher. Combining these signals as m0 + 1.5*m1 - m2 maximizes the mean gap between the theories while effectively eliminating the parameter-driven variance of the Competing theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    diff0_ttb = []
    diff1_tally = []
    diff2_tally = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a)
        sum_b = sum(b)
        diff = abs(sum_a - sum_b)
        
        if diff == 0:
            # Determine TTB winner
            ttb_winner = None
            for j in range(len(a)):
                if a[j] > b[j]:
                    ttb_winner = 0
                    break
                elif b[j] > a[j]:
                    ttb_winner = 1
                    break
            if ttb_winner is not None:
                diff0_ttb.append(1 if resp == ttb_winner else 0)
                
        elif diff == 1:
            tally_winner = 0 if sum_a > sum_b else 1
            diff1_tally.append(1 if resp == tally_winner else 0)
            
        elif diff == 2:
            tally_winner = 0 if sum_a > sum_b else 1
            diff2_tally.append(1 if resp == tally_winner else 0)
            
    m0 = float(np.mean(diff0_ttb)) if diff0_ttb else 0.5
    m1 = float(np.mean(diff1_tally)) if diff1_tally else 0.5
    m2 = float(np.mean(diff2_tally)) if diff2_tally else 0.5
    
    return float(m0 + 1.5 * m1 - m2)


### RESULTS
- Predicted under pi_3 (simulated): 0.7178 (var=0.0228)
- Predicted under pi_6 (simulated): 0.8667 (var=0.0331)
- Observed on real data: 0.8344 (var=0.0346)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f1 = np.array([x[0] for x in data['option_a_ratings']])
    b_f1 = np.array([x[0] for x in data['option_b_ratings']])
    responses = data['response'].values
    
    favors_A = a_f1 > b_f1
    favors_B = b_f1 > a_f1
    
    aligned = (favors_A & (responses == 0)) | (favors_B & (responses == 1))
    return float(np.mean(aligned))
```

**Observed (real) value:** 0.4171 (var=0.0031)
**Predicted under pi_3:** 0.3875 (var=0.0065)
**Predicted under pi_6:** 0.3281 (var=0.0138)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_prediction(a, b):
        # Cues are already in descending order of validity
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_b > val_a:
                return 1
        return -1
        
    ttb_preds = np.array([get_ttb_prediction(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])])
    valid_mask = (ttb_preds != -1)
    
    if not np.any(valid_mask):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[valid_mask] == ttb_preds[valid_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4083 (var=0.0021)
**Predicted under pi_3:** 0.3448 (var=0.0115)
**Predicted under pi_6:** 0.3292 (var=0.0112)

### Experiment 5
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify critical trials where Option A has 3 positive features and Option B has 2.
    # In the provided design, these correspond to Trials 1, 2, and 3.
    # Tallying will consistently prefer A (3 > 2), whereas WADD will tend to prefer B
    # because B's features are drawn from the high-validity tier.
    is_target = data.apply(lambda row: sum(row['option_a_ratings']) == 3 and sum(row['option_b_ratings']) == 2, axis=1)
    
    target_data = data[is_target]
    
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(target_data['response'] == 0))
```

**Observed (real) value:** 0.5678 (var=0.0020)
**Predicted under pi_3:** 0.5928 (var=0.0078)
**Predicted under pi_6:** 0.6667 (var=0.0167)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Find 'tie trials' where the number of positive features is equal for both options
    def is_tie(row):
        return sum(row['option_a_ratings']) == sum(row['option_b_ratings'])
        
    tie_mask = data.apply(is_tie, axis=1)
    tie_data = data[tie_mask].copy()
    
    if tie_data.empty:
        return 0.0
        
    # Create a hashable ID for each trial type to group by
    def make_id(row):
        a_str = ''.join([str(int(x)) for x in row['option_a_ratings']])
        b_str = ''.join([str(int(x)) for x in row['option_b_ratings']])
        return a_str + "_" + b_str
        
    tie_data['trial_id'] = tie_data.apply(make_id, axis=1)
    
    # Calculate the choice proportion for each subject and trial type
    means = tie_data.groupby(['subject_id', 'trial_id'])['response'].mean()
    
    # Compute absolute deviation from random guessing (0.5)
    abs_devs = (means - 0.5).abs()
    
    return float(abs_devs.mean())
```

**Observed (real) value:** 0.1133 (var=0.0017)
**Predicted under pi_3:** 0.1133 (var=0.0021)
**Predicted under pi_6:** 0.1221 (var=0.0019)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    scores = []
    for subject, subj_df in data.groupby('subject_id'):
        x = []
        y = []
        for _, row in subj_df.iterrows():
            a = row['option_a_ratings']
            b = row['option_b_ratings']
            
            # TTB winner is determined by cue 0 (highest validity)
            # In this design, cue 0 always differs between A and B.
            ttb_winner = 0 if a[0] > b[0] else 1
            
            sum_a = sum(a)
            sum_b = sum(b)
            
            # Tally difference in favor of the TTB winner
            if ttb_winner == 0:
                diff = sum_a - sum_b
            else:
                diff = sum_b - sum_a
                
            x.append(diff)
            
            # Response 0 means option A, 1 means option B
            chose_ttb = 1.0 if row['response'] == ttb_winner else 0.0
            y.append(chose_ttb)
            
        # Linear regression of TTB choice on Tally difference
        if len(np.unique(x)) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            scores.append(intercept)
        else:
            scores.append(0.5)
            
    return float(np.mean(scores))
```

**Observed (real) value:** 0.4692 (var=0.0060)
**Predicted under pi_3:** 0.4872 (var=0.0097)
**Predicted under pi_6:** 0.4949 (var=0.0110)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    tally_diff = a_ratings.sum(axis=1) - b_ratings.sum(axis=1)
    
    cue_order = [1, 3, 5, 0, 2, 4]
    ttb_A = np.zeros(len(a_ratings), dtype=int)
    for i in range(len(a_ratings)):
        a = a_ratings[i]
        b = b_ratings[i]
        for j in cue_order:
            if a[j] > b[j]:
                ttb_A[i] = 1
                break
            elif b[j] > a[j]:
                ttb_A[i] = -1
                break
                
    a_str = [''.join(map(lambda x: str(int(x)), row)) for row in a_ratings]
    b_str = [''.join(map(lambda x: str(int(x)), row)) for row in b_ratings]
    
    df = pd.DataFrame({
        'subj': data['subject_id'].values,
        'tally_diff': tally_diff,
        'ttb_A': ttb_A,
        'choice_A': (data['response'].values == 0).astype(int),
        'trial_id': [a + "_" + b for a, b in zip(a_str, b_str)]
    })
    
    grouped = df.groupby(['subj', 'trial_id'])
    agg = grouped.agg(
        tally_diff=('tally_diff', 'first'),
        ttb_A=('ttb_A', 'first'),
        N_A=('choice_A', 'sum'),
        N_tot=('choice_A', 'count')
    ).reset_index()
    
    agg['C'] = (agg['N_A'] + 0.5) / (agg['N_tot'] + 1.0)
    agg['L'] = np.log(agg['C'] / (1.0 - agg['C']))
    
    subj_metrics = []
    for subj, subj_df in agg.groupby('subj'):
        L_0A = subj_df[(subj_df['tally_diff'] == 0) & (subj_df['ttb_A'] == 1)]['L'].mean()
        L_0B = subj_df[(subj_df['tally_diff'] == 0) & (subj_df['ttb_A'] == -1)]['L'].mean()
        L_2A = subj_df[(subj_df['tally_diff'] == 2) & (subj_df['ttb_A'] == 1)]['L'].mean()
        L_m2A = subj_df[(subj_df['tally_diff'] == -2) & (subj_df['ttb_A'] == 1)]['L'].mean()
        
        if pd.isna(L_0A) or pd.isna(L_0B) or pd.isna(L_2A) or pd.isna(L_m2A):
            continue
            
        metric_val = (L_0A - L_0B) + (L_2A + L_m2A)
        subj_metrics.append(metric_val)
        
    if not subj_metrics:
        return 0.0
        
    return float(np.mean(subj_metrics))
```

**Observed (real) value:** 0.2999 (var=0.6654)
**Predicted under pi_3:** -0.0629 (var=0.9997)
**Predicted under pi_6:** 1.1740 (var=1.4243)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    target_A = (1, 1, 0, 0, 0)
    target_B = (0, 0, 0, 1, 1)
    
    count = 0
    match = 0
    
    for _, row in data.iterrows():
        a_val = tuple(row['option_a_ratings'])
        b_val = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a_val == target_A and b_val == target_B:
            count += 1
            if resp == 0:
                match += 1
        elif a_val == target_B and b_val == target_A:
            count += 1
            if resp == 1:
                match += 1
                
    if count == 0:
        return 0.5
    return match / count
```

**Observed (real) value:** 0.4967 (var=0.0203)
**Predicted under pi_3:** 0.5350 (var=0.0203)
**Predicted under pi_6:** 0.5133 (var=0.0282)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    # Extract option ratings as 2D arrays
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    # Calculate tallies for each option
    sum_A = np.sum(A, axis=1)
    sum_B = np.sum(B, axis=1)
    
    # Isolate "tally-tied" trials where Tallying predicts guessing
    tied_mask = (sum_A == sum_B)
    
    if not np.any(tied_mask):
        return 0.5
        
    # Calculate WADD values for the tied trials
    val_A = np.sum(A[tied_mask] * validities, axis=1)
    val_B = np.sum(B[tied_mask] * validities, axis=1)
    
    responses = data['response'].values[tied_mask]
    
    # Determine the option favored by the WADD model (0 for A, 1 for B)
    expected = (val_B > val_A).astype(int)
    
    # Return the proportion of choices aligning with the WADD model's prediction
    return float(np.mean(responses == expected))
```

**Observed (real) value:** 0.4633 (var=0.0070)
**Predicted under pi_3:** 0.5017 (var=0.0060)
**Predicted under pi_6:** 0.5567 (var=0.0110)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 1 (Pure Tallying) provides a surprisingly strong fit for many baseline cases (e.g., Experiments 5, 6, 7, 10), outperforming Theory 2 (Tallying with Conditional TTB Fallback) which consistently over-predicts the magnitude of validity-driven effects. However, Pure Tallying systematically fails in experiments designed to isolate validity effects when tallies are tied or close (e.g., Experiments 1, 2, and 8). In these cases, the observed data show small but reliable positive shifts toward the more valid cues, which Pure Tallying rigidly predicts to be zero. Theory 2 correctly predicts the direction of these effects but drastically overestimates them because a TTB fallback is too strong and localized. The data suggest that subjects do not abruptly switch to a non-compensatory TTB strategy when tallies are close; rather, they seem to use a compensatory strategy where weights are heavily compressed toward equality but still retain a weak sensitivity to cue validity.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory: 'Compressed Weighted Additive (WADD) Model'. Instead of a discrete fallback to a non-compensatory TTB rule, this theory should posit that subjects integrate all features using a compensatory WADD approach, but the subjective weights applied to the cues are heavily compressed toward equal weighting. This means the overall tally dominates the choice probability, but the cue validities exert a small, continuous influence across all trials. This avoids the over-prediction of TTB fallback while correctly capturing the subtle, pervasive validity effects missed by Pure Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 10965,
  "candidates_token_count": 371,
  "total_token_count": 12268
}
```
