# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Variance Penalization (Risk Aversion in Cue Integration): Subjects evaluate options by integrating the validities of present features, but they actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. The subjective value of an option is its non-linearly weighted sum of validities minus a penalty proportional to the standard deviation of its active validities, which scales appropriately to strongly influence decisions.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    lambda_pen = float(parameters["lambda_pen"])
    
    def get_score(x):
        active_mask = (x > 0.5)
        if not np.any(active_mask):
            return 0.0
        
        # Weighted sum using exponentiated validities for compensatoriness flexibility
        sum_v = np.sum(val[active_mask] ** gamma)
        
        # Standard deviation penalty using original validities to capture dispersion
        std_v = np.std(val[active_mask]) if np.sum(active_mask) > 1 else 0.0
        
        return sum_v - lambda_pen * std_v

    score_a = get_score(a)
    score_b = get_score(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_3
People make decisions by computing a weighted sum of the features for each option, using the cue validities scaled by an exponent as the weights. Unlike Take The Best, which is strictly non-compensatory, the Weighted Additive (WADD) strategy integrates all available information. However, by scaling validities with an exponent (gamma), the model can flexibly capture varying degrees of compensatoriness, ranging from equal-weighting (Tallying) to highly skewed weighting that approximates one-reason decision making (TTB). Choice probabilities are generated via a softmax over the weighted sums, combined with a uniform lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To distinguish the Variance Penalization model (THEORY 1) from the exponent-scaled Weighted Additive model (THEORY 2), we exploit the fact that THEORY 1 penalizes options with a high dispersion (standard deviation) of validities among their active features. In contrast, THEORY 2 always assigns positive weight to any active feature, meaning adding a feature strictly increases an option's value. We include 'dilution' trials where Option A is a strict subset of Option B (e.g., Option A has only a 0.95 validity feature, while Option B has both 0.95 and 0.55). THEORY 2 will always prefer Option B because the extra feature adds value. THEORY 1, however, can strongly prefer Option A because adding the 0.55 feature introduces a high variance penalty that outweighs its additive value. We also include tied-sum pairs with different variances (e.g., 0.95 + 0.55 vs 0.85 + 0.65) to further dissociate the non-linear scaling of THEORY 2 from the explicit variance penalty of THEORY 1.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the frequency with which subjects choose the 'diluted' option in trials where one option is a strict subset of the other (e.g., A has only feature 1, while B has features 1 and 5). According to the Weighted Additive model (Theory 2), adding a feature always increases the value, so the subset option should never be chosen (except due to random lapse/noise). In contrast, the Variance Penalization model (Theory 1) predicts that adding a low-validity feature to a high-validity feature increases the variance penalty, potentially making the strict subset more attractive. Thus, a higher rate of choosing the subset strongly discriminates Theory 1 from Theory 2.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # Identify trials where A is a strict subset of B
    a_le_b = np.all(a_ratings <= b_ratings, axis=1)
    a_lt_b = np.any(a_ratings < b_ratings, axis=1)
    strict_subset_a = a_le_b & a_lt_b
    
    # Identify trials where B is a strict subset of A
    b_le_a = np.all(b_ratings <= a_ratings, axis=1)
    b_lt_a = np.any(b_ratings < a_ratings, axis=1)
    strict_subset_b = b_le_a & b_lt_a
    
    subset_chosen = 0
    subset_trials = 0
    
    subset_trials += np.sum(strict_subset_a)
    subset_chosen += np.sum((responses == 0) & strict_subset_a)
    
    subset_trials += np.sum(strict_subset_b)
    subset_chosen += np.sum((responses == 1) & strict_subset_b)
    
    if subset_trials == 0:
        return 0.0
        
    return float(subset_chosen / subset_trials)

### RESULTS
- Predicted under pi_6 (simulated): 0.8328 (var=0.0236)
- Predicted under pi_3 (simulated): 0.4344 (var=0.0280)
- Observed on real data: 0.8600 (var=0.0085)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 8: A=[0, 1, 1, 0]  B=[0, 1, 1, 1]

**Rationale:** To maximally distinguish the exponent-scaled Weighted Additive model (THEORY 1) from the Variance Penalization model (THEORY 2), we use 'dilution' trials. In THEORY 1, adding any active feature strictly increases an option's value because all validities are positive. In THEORY 2, adding a low-validity feature to a high-validity feature increases the standard deviation of active validities, which incurs a penalty that can decrease the option's total value. We pit options with a single high-validity feature against options with that same feature plus a low-validity feature. THEORY 1 will always prefer the option with more features, while THEORY 2 can strongly prefer the option with fewer features. We also include pairs with tied linear sums but different variances to separate the non-linear scaling of validities (THEORY 1) from the explicit variance penalty (THEORY 2).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses Option B over Option A when B is a strict superset of A (specifically, adding the lowest-validity 4th feature). According to the Weighted Additive model (THEORY 1), adding any feature strictly increases an option's value, so subjects should consistently prefer B. According to the Variance Penalization model (THEORY 2), adding a low-validity feature to a high-validity feature increases the variance of active features, which incurs a penalty. Thus, THEORY 2 predicts a much lower preference for B (and potentially a preference for A) on these specific trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify trials where Option B is identical to Option A except that B has the 4th feature (index 3) active
    is_target_trial = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        if tuple(a[:3]) == tuple(b[:3]) and a[3] == 0 and b[3] == 1:
            is_target_trial.append(True)
        else:
            is_target_trial.append(False)
            
    target_data = data[is_target_trial]
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B was chosen
    return float(target_data['response'].mean())


### RESULTS
- Predicted under pi_6 (simulated): 0.1697 (var=0.0137)
- Predicted under pi_3 (simulated): 0.5797 (var=0.0223)
- Observed on real data: 0.1360 (var=0.0092)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd

    def get_predictions(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
                
        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])
        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])
        if a_wins > b_wins:
            tally = 0
        elif b_wins > a_wins:
            tally = 1
        else:
            tally = -1
            
        return pd.Series({'ttb': ttb, 'tally': tally})
        
    preds = data.apply(get_predictions, axis=1)
    
    # Filter for trials where TTB and Tallying make strictly opposite predictions
    mask = (preds['ttb'] != -1) & (preds['tally'] != -1) & (preds['ttb'] != preds['tally'])
    
    if not mask.any():
        return 0.5
        
    # Return the proportion of choices that align with Take The Best on these critical trials
    ttb_choices = (data.loc[mask, 'response'] == preds.loc[mask, 'ttb']).mean()
    return float(ttb_choices)

```

**Observed (real) value:** 0.7858 (var=0.0260)
**Predicted under pi_6:** 0.8688 (var=0.0096)
**Predicted under pi_3:** 0.7021 (var=0.0525)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue  # Tallying predicts a tie
            
        # TTB prediction (cues are ordered by descending validity based on the design)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred is None:
            continue  # TTB predicts a tie
            
        # Only consider trials where the two theories make strictly opposite predictions
        if tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            count += 1
            
    if count == 0:
        return 0.5
        
    return float(tally_match / count)

```

**Observed (real) value:** 0.2306 (var=0.0195)
**Predicted under pi_6:** 0.1356 (var=0.0122)
**Predicted under pi_3:** 0.3344 (var=0.0398)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_mat - b_mat
    
    match = 0
    total = 0
    for i in range(len(diff)):
        row_diff = diff[i]
        nonzero_idx = np.nonzero(row_diff)[0]
        if len(nonzero_idx) > 0:
            first_idx = nonzero_idx[0]
            ttb_choice = 0 if row_diff[first_idx] > 0 else 1
            if resp[i] == ttb_choice:
                match += 1
            total += 1
            
    return float(match / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6521 (var=0.0118)
**Predicted under pi_6:** 0.7094 (var=0.0123)
**Predicted under pi_3:** 0.6856 (var=0.0224)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    mask = (sum_a == 1) & (sum_b >= 3)
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.2233 (var=0.0237)
**Predicted under pi_6:** 0.1408 (var=0.0228)
**Predicted under pi_3:** 0.2500 (var=0.0332)

### Experiment 7
**Design**
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Focus exclusively on consensus trials where both models 
        # theoretically predict the TTB option wins for almost all parameters.
        if a == (1, 1, 0, 0) and b == (0, 0, 1, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
        elif a == (1, 0, 1, 0) and b == (0, 1, 0, 1):
            if row['response'] == 0:
                match_count += 1
            total += 1
            
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.1175 (var=0.0070)
**Predicted under pi_6:** 0.7913 (var=0.0225)
**Predicted under pi_3:** 0.8662 (var=0.0111)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    moderate_chosen = 0
    tied_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(val * a)
        sum_b = np.sum(val * b)
        
        # Check if it's a tied pair (linear sum of validities is equal)
        if abs(sum_a - sum_b) < 1e-4:
            tied_trials += 1
            
            # Find the best discriminating cue
            for i in range(5):
                if a[i] != b[i]:
                    best_cue_favors_a = (a[i] > b[i])
                    break
            
            # The 'moderate' option is the one NOT favored by the best cue
            chose_a = (row['response'] == 0)
            
            if best_cue_favors_a and not chose_a:
                moderate_chosen += 1
            elif not best_cue_favors_a and chose_a:
                moderate_chosen += 1
                
    if tied_trials == 0:
        return 0.0
    return float(moderate_chosen / tied_trials)
```

**Observed (real) value:** 0.6593 (var=0.0108)
**Predicted under pi_6:** 0.8530 (var=0.0197)
**Predicted under pi_3:** 0.2067 (var=0.0154)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_t1 = (1, 0, 0, 1, 0)
    b_t1 = (0, 1, 1, 0, 0)
    
    a_t2 = (1, 0, 0, 0, 1)
    b_t2 = (0, 1, 0, 1, 0)
    
    def get_val(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == a_t1 and b == b_t1:
            return 1 - row['response']
        elif a == a_t2 and b == b_t2:
            return 1 - row['response']
        elif a == b_t1 and b == a_t1:
            return row['response']
        elif a == b_t2 and b == a_t2:
            return row['response']
        return np.nan
        
    vals = data.apply(get_val, axis=1).dropna()
    if len(vals) == 0:
        return 0.5
    return float(vals.mean())
```

**Observed (real) value:** 0.1450 (var=0.0164)
**Predicted under pi_6:** 0.1425 (var=0.0133)
**Predicted under pi_3:** 0.8650 (var=0.0095)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify trials where both options have exactly 2 active cues.
    # In the design, these are trials 1, 2, and 3.
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask = (a_sums == 2) & (b_sums == 2)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.0
        
    # Group by trial type using string representations of option A
    subset_key = subset['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    devs = []
    for key, group in subset.groupby(subset_key):
        # proportion of times A was chosen (response == 0)
        p_a = np.mean(group['response'] == 0)
        devs.append(abs(p_a - 0.5))
        
    return float(np.mean(devs))
```

**Observed (real) value:** 0.3517 (var=0.0118)
**Predicted under pi_6:** 0.3208 (var=0.0130)
**Predicted under pi_3:** 0.3317 (var=0.0063)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Variance Penalization) consistently outperforms Theory 2 (Weighted Additive) across almost all experiments, most notably on Experiments 1, 2, 8, and 9. Theory 2 completely fails to capture the 'dilution' effect observed in Experiments 1 and 2, where adding a low-validity feature to an option actually decreases its subjective value and choice probability. Theory 2 predicts that adding any feature strictly increases value, whereas the data strongly shows the opposite. While Theory 1 captures this via a variance penalty, another highly plausible cognitive mechanism for this dilution effect is that subjects evaluate options by averaging the validities of their active features rather than summing them. Theory 2 is fundamentally unable to capture these effects and should be completely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Weighted Additive theory with an 'Averaging of Active Validities' theory. The new theory should posit that subjects evaluate options by computing the mean (or a weighted combination of the mean and the sum) of the validities of the active features, rather than strictly summing them. This naturally explains the dilution effects in Experiments 1 and 2 (since adding a low-validity feature lowers the average) without needing an explicit variance penalty, and serves as a strong, distinct competitor to Theory 1."
}
```

## Usage

```json
{
  "prompt_token_count": 9709,
  "candidates_token_count": 301,
  "total_token_count": 11658
}
```
