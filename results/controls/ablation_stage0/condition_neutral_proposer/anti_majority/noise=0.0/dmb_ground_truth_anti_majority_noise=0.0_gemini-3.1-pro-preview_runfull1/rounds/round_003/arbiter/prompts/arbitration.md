# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Attention-Sharing (Fractional Discounting): Subjects evaluate options by linearly summing the validities of present features, but the overall value is discounted by the number of active cues. This reflects a cognitive bottleneck where attention must be divided among multiple features. The subjective value of an option is its sum of validities divided by `1.0 + kappa * (n_cues - 1)`. This mechanism allows a single highly valid cue to maintain its full weight and potentially dominate an option with multiple weaker cues that dilute each other's impact, explaining violations of tallying dominance while remaining mathematically stable.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    def get_score(x):
        val_x = val * x
        n_cues = np.sum(x)
        if n_cues == 0:
            return 0.0
        linear = np.sum(val_x)
        return linear / (1.0 + kappa * (n_cues - 1.0))

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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To distinguish between the Attention-Sharing (Fractional Discounting) model (THEORY 1) and the exponent-scaled Weighted Additive model (THEORY 2), we exploit trials where the number of active cues is equal across both options. In THEORY 1, the denominator '1 + kappa * (n_cues - 1)' applies equally to both options when they have the same number of features. Consequently, THEORY 1 is constrained to behave exactly like a linear Weighted Additive model on these trials and will always predict a tie if the linear sum of validities is equal. THEORY 2, however, applies a non-linear exponent (gamma) to the validities. For pairs where the linear sums are tied but the dispersion of validities differs (e.g., 0.95 + 0.65 = 1.60 vs 0.85 + 0.75 = 1.60), THEORY 2 will strongly prefer the extreme validities if gamma > 1, or the moderate validities if gamma < 1. We also include trials with unequal numbers of cues to capture the unique discounting dynamic of THEORY 1.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with the most extreme validities (Option A) on trials where the linear sum of validities and the number of cues are exactly equal (Trials 1 and 2). According to the Attention-Sharing (Fractional Discounting) model, both options have identical subjective values on these trials, so the choice proportion should be 0.5. In contrast, the Weighted Additive model uses an exponent (gamma) on the validities. Since the parameter space for gamma extends heavily into values > 1, the Weighted Additive model will consistently prefer the option with the single highest validity (Option A), leading to a choice proportion significantly greater than 0.5.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.5092 (var=0.0103)
- Predicted under pi_3 (simulated): 0.8650 (var=0.0095)
- Observed on real data: 0.1450 (var=0.0164)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To maximally distinguish the exponent-scaled Weighted Additive model (THEORY 1) from the Attention-Sharing/Fractional Discounting model (THEORY 2), this design uses a mix of equal and unequal cue-count pairs. For pairs where both options have the same number of active cues but different validity dispersions (e.g., 0.95 + 0.65 = 1.60 vs 0.85 + 0.75 = 1.60), THEORY 2 applies the same denominator to both and predicts a perfect tie, while THEORY 1 predicts a strong preference depending on its gamma parameter. For pairs with unequal active cues (e.g., 1 cue vs 3 cues), THEORY 2 strongly penalizes the option with more cues via the kappa-scaled denominator, making it possible for a single high-validity cue to defeat a much larger linear sum of lower-validity cues in ways that THEORY 1 can only match with extreme gamma values. The combination of tied linear sums, equal/unequal cue counts, and varying validity dispersions cleanly dissociates the non-linear scaling of validities (THEORY 1) from the attentional penalty on the number of cues (THEORY 2).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
In trials where both options have exactly 2 active cues and their linear sum of validities are perfectly tied (trials 1, 2, and 3), Theory 2 (Attention-Sharing) predicts exactly equal preference (p(A) = 0.5) because the denominators are the same and the linear numerators are the same. Theory 1 (WADD), however, applies a non-linear exponent (gamma) to the validities, which breaks the tie due to the differing dispersions of validities between the options. By calculating the mean absolute deviation of the choice proportion from 0.5 on these specific trials, Theory 2 will consistently yield values close to the expected binomial noise floor (around 0.13 for 8 repetitions), whereas Theory 1 will yield substantially higher deviations as it systematically prefers one option over the other.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.0158 (var=0.0043)
- Predicted under pi_3 (simulated): 0.3317 (var=0.0063)
- Observed on real data: 0.3517 (var=0.0118)

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
**Predicted under pi_5:** 0.6379 (var=0.1062)
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
**Predicted under pi_5:** 0.3372 (var=0.0868)
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
**Predicted under pi_5:** 0.6238 (var=0.0633)
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
**Predicted under pi_5:** 0.4100 (var=0.1130)
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
**Predicted under pi_5:** 0.7750 (var=0.0206)
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
**Predicted under pi_5:** 0.5204 (var=0.0043)
**Predicted under pi_3:** 0.2067 (var=0.0154)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories fail to capture key patterns in the observed data across multiple experiments. Specifically, in Experiment 1 and Experiment 8, subjects systematically prefer the 'moderate' option over the 'extreme' option when the linear sums of validities are tied (Observed Exp 1: 0.1450 extreme chosen, meaning ~0.85 moderate chosen; Exp 8: 0.6593 moderate chosen). Theory 1 (pi_5) predicts indifference (~0.50), while Theory 2 (pi_3) strongly predicts the extreme option (~0.86 and ~0.20 for moderate). Furthermore, in Experiment 7, both theories confidently predict that subjects will choose Option A (the option favored by Take-The-Best and higher linear sums, >0.77), yet the observed data shows subjects overwhelmingly choosing Option B (0.1175). This indicates a fundamental misunderstanding of the weighting or integration mechanism in both models. The preference for moderate cues suggests subjects might be averse to the variance or dispersion of cue validities, treating extreme high/low combinations as risky or unreliable compared to clustered moderate validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory based on 'Variance Penalization' or 'Risk Aversion' in cue integration. The theory should posit that subjects do not just sum validities (linearly or non-linearly), but actively penalize options that rely on highly dispersed or extreme expert ratings. An option with moderate, consistent validities is perceived as more reliable than one with a mix of very high and very low validities. This would naturally explain the strong preference for moderate validities in tied-sum scenarios (Exp 1 and 8) and could help explain the anomalous behavior in Exp 7."
}
```

## Usage

```json
{
  "prompt_token_count": 8438,
  "candidates_token_count": 404,
  "total_token_count": 10127
}
```
