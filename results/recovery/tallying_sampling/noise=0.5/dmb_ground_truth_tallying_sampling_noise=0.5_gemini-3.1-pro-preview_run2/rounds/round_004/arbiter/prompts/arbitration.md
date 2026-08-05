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
People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Probabilistic Strategy Mixture: Decision-makers do not blend strategies into a single value score, nor do they use deterministic tie-breaking. Instead, they probabilistically sample a decision strategy on each trial. The predominant strategy is Pure Tallying (Equal Weight), which counts positive features and ignores validities. However, on a small proportion of trials, they employ a fully compensatory Weighted Additive (WADD) strategy that integrates explicit cue validities. This mixture of pure heuristics preserves the strong tallying dominance and low validity bias in certain environments, while the occasional use of WADD produces the graded validity correlations observed in others.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Strategy 1: Pure Tallying
    tally_scores = stim.sum(axis=1)
    
    # Strategy 2: Weighted Additive (WADD)
    validities = np.asarray(parameters["validities"], dtype=float)
    val_scores = stim @ validities
    
    beta = float(parameters["beta"])
    p_val = float(parameters["p_val"])
    epsilon = float(parameters["epsilon"])
    
    # Probabilities under Tallying
    z_t = beta * (tally_scores - np.max(tally_scores))
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Probabilities under WADD
    z_w = beta * (val_scores - np.max(val_scores))
    e_w = np.exp(z_w)
    p_wadd = e_w / e_w.sum()
    
    # Probabilistic strategy mixture
    p_mix = (1.0 - p_val) * p_tally + p_val * p_wadd
    
    # Add uniform lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Pure Tallying from the Probabilistic Strategy Mixture (which occasionally uses WADD), this design exploits trials where the two models make divergent predictions. In 'Tally Tie' trials, both options have an equal number of positive features, but one option holds a significant validity advantage. Pure Tallying predicts exactly 50/50 guessing on these trials, whereas the Mixture model predicts a systematic preference for the option with higher validities because the WADD component breaks the tie. Furthermore, in 'Conflict' trials, one option has more positive features (favored by Tallying) while the other has fewer but highly valid features (favored by WADD). Pure Tallying predicts a constant preference based purely on the tally difference, while the Mixture model predicts that the choice probability will be attenuated by the opposing validity gradient.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'Tally Tie' trials where both options have an equal number of positive features, but one option has a clear advantage in validity scores (WADD score). Pure Tallying predicts 50/50 guessing on these trials, so the proportion of choices favoring the WADD prediction should be exactly 0.5. The Probabilistic Strategy Mixture model predicts that the occasional use of WADD will break these ties systematically, resulting in a proportion significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.6, 0.55, 0.5])
    
    wadd_consistent_choices = 0
    eligible_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a)
        tally_b = np.sum(b)
        
        if tally_a == tally_b:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b + 0.01:
                eligible_trials += 1
                if row['response'] == 0:
                    wadd_consistent_choices += 1
            elif wadd_b > wadd_a + 0.01:
                eligible_trials += 1
                if row['response'] == 1:
                    wadd_consistent_choices += 1
                    
    if eligible_trials == 0:
        return 0.5
        
    return wadd_consistent_choices / eligible_trials

### RESULTS
- Predicted under pi_3 (simulated): 0.5021 (var=0.0042)
- Predicted under pi_6 (simulated): 0.5675 (var=0.0051)
- Observed on real data: 0.5208 (var=0.0074)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Probabilistic Strategy Mixture (which occasionally uses WADD) from Pure Tallying, this design leverages 'Tally Tie' trials and 'Conflict' trials. On Tally Tie trials, both options have an equal number of positive features, but one option has higher validity features. Pure Tallying predicts exactly 50/50 guessing on these trials, whereas the Mixture model predicts a systematic preference for the higher-validity option because the WADD component breaks the tie. Furthermore, on Conflict trials, one option has more positive features (favored by Tallying) while the other has fewer but highly valid features (favored by WADD). Pure Tallying predicts a constant preference based solely on the tally difference, while the Mixture model predicts that the choice probability will be attenuated by the opposing WADD preference.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
The Competing Theory (Pure Tallying) restricts the inverse temperature parameter `beta` to [0.1, 1.5], meaning choices remain relatively noisy even when one option has a clear advantage. In contrast, the Advocated Theory (Probabilistic Strategy Mixture) allows `beta` to range up to 5.0, enabling highly deterministic choices. By isolating 'No-Conflict' trials—where both the Tallying and WADD strategies strongly favor the same option—we can measure the overall choice determinism. The Mixture theory will predict a significantly higher proportion of choices aligned with the favored option compared to the artificially noisy Pure Tallying theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    
    tally_diff = tally_a - tally_b
    wadd_diff = wadd_a - wadd_b
    
    # No-conflict trials: tally and wadd favor the same option
    mask = (tally_diff * wadd_diff) > 0
    
    if not np.any(mask):
        return 0.5
        
    tally_favors_a = tally_diff > 0
    tally_favors_b = tally_diff < 0
    
    resp = data['response'].values[mask]
    
    match_a = tally_favors_a[mask] & (resp == 0)
    match_b = tally_favors_b[mask] & (resp == 1)
    
    return float(np.mean(match_a | match_b))

### RESULTS
- Predicted under pi_3 (simulated): 0.6329 (var=0.0095)
- Predicted under pi_6 (simulated): 0.7258 (var=0.0181)
- Observed on real data: 0.6702 (var=0.0062)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    weights = np.array([1000, 100, 10, 1])
    score_diff = diff @ weights
    ttb_preds = np.where(score_diff > 0, 0, 1)
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3504 (var=0.0026)
**Predicted under pi_3:** 0.3846 (var=0.0045)
**Predicted under pi_6:** 0.2975 (var=0.0092)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6741 (var=0.0053)
**Predicted under pi_3:** 0.6930 (var=0.0104)
**Predicted under pi_6:** 0.7811 (var=0.0147)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify Trial 3 where both options have exactly 1 positive feature.
    # Tallying predicts a perfect tie (score 1 vs 1) for all subjects.
    # WADD predicts a preference based on subjective weights for the specific features.
    def is_t3(row):
        return sum(row['option_a_ratings']) == 1 and sum(row['option_b_ratings']) == 1
    
    mask = data.apply(is_t3, axis=1)
    t3_data = data[mask]
    
    if len(t3_data) == 0:
        return 0.0
        
    # Calculate each subject's absolute deviation from 0.5 probability of choosing A
    subject_devs = []
    for subj, subj_df in t3_data.groupby('subject_id'):
        pA = (subj_df['response'] == 0).mean()
        subject_devs.append(abs(pA - 0.5))
        
    if not subject_devs:
        return 0.0
        
    return float(np.mean(subject_devs))
```

**Observed (real) value:** 0.1000 (var=0.0041)
**Predicted under pi_3:** 0.0988 (var=0.0038)
**Predicted under pi_6:** 0.0875 (var=0.0067)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)

    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    trial_1 = a_11000 & b_00111
    trial_2 = a_00111 & b_11000
    
    mask = trial_1 | trial_2
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    chose_00111 = (trial_1 & (data['response'] == 1)) | (trial_2 & (data['response'] == 0))
    chose_00111_subset = chose_00111[mask]
    
    return float(chose_00111_subset.mean())
```

**Observed (real) value:** 0.7017 (var=0.0062)
**Predicted under pi_3:** 0.6583 (var=0.0175)
**Predicted under pi_6:** 0.7175 (var=0.0200)

### Experiment 7
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Use the exact validities specified in the experimental design
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    
    # Safely convert lists to numpy arrays for vectorized operations
    a_mat = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b_mat = np.array(data['option_b_ratings'].tolist(), dtype=float)
    
    # Compute the Tally Difference and Validity Difference (Option B - Option A)
    tally_diff = b_mat.sum(axis=1) - a_mat.sum(axis=1)
    val_diff = b_mat.dot(validities) - a_mat.dot(validities)
    
    df = pd.DataFrame({
        'td': tally_diff,
        'vd': val_diff,
        'resp': data['response'].values
    })
    
    # Compute the empirical mean response per subject per tally difference
    if 'subject_id' in data.columns:
        df['subject_id'] = data['subject_id'].values
        td_means = df.groupby(['subject_id', 'td'])['resp'].transform('mean')
    else:
        td_means = df.groupby('td')['resp'].transform('mean')
        
    # The residual choice perfectly partials out the main effect of the Tallying heuristic
    res = df['resp'] - td_means
    
    # The covariance between the residual choice and the validity difference 
    # isolates the unique contribution of the Validity Bias.
    return float(np.mean(res * df['vd']))
```

**Observed (real) value:** 0.0049 (var=0.0003)
**Predicted under pi_3:** -0.0017 (var=0.0004)
**Predicted under pi_6:** 0.0047 (var=0.0003)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    def get_tally(x):
        return sum(x)
        
    def get_val(x):
        return sum(v * r for v, r in zip(validities, x))
        
    tally_a = data['option_a_ratings'].apply(get_tally)
    tally_b = data['option_b_ratings'].apply(get_tally)
    
    # Only consider trials where the tally scores are equal (ties)
    tie_mask = tally_a == tally_b
    if tie_mask.sum() == 0:
        return 0.5
        
    tie_data = data[tie_mask]
    
    val_a = tie_data['option_a_ratings'].apply(get_val)
    val_b = tie_data['option_b_ratings'].apply(get_val)
    
    # Determine which option has the higher validity sum
    higher_val_is_b = (val_b > val_a).astype(int)
    
    # Calculate the proportion of choices that align with the higher validity option
    match = (tie_data['response'] == higher_val_is_b).mean()
    
    return float(match)
```

**Observed (real) value:** 0.4964 (var=0.0074)
**Predicted under pi_3:** 0.4974 (var=0.0052)
**Predicted under pi_6:** 0.5303 (var=0.0081)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1 = frozenset([(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)])
    t2 = frozenset([(1, 0, 1, 1, 0), (0, 1, 0, 0, 1)])
    t3 = frozenset([(0, 1, 1, 1, 0), (1, 0, 0, 0, 1)])
    t4 = frozenset([(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)])
    t5 = frozenset([(1, 1, 0, 0, 0), (0, 0, 1, 0, 0)])
    t6 = frozenset([(0, 0, 1, 1, 0), (1, 0, 0, 0, 0)])
    t7 = frozenset([(0, 0, 0, 1, 1), (1, 0, 0, 0, 0)])
    
    # Optimal linear contrast weights derived from mean-centered validity differences.
    # These sum to exactly 0, ensuring that any model predicting a constant choice
    # probability across these tally-diff=1 trials (like Pure Tallying) will yield
    # an expected score of exactly 0, perfectly canceling out subject-level baseline differences.
    weights = {
        t1: 0.957,   # val_diff = +1.5
        t5: 0.757,   # val_diff = +1.3
        t2: 0.157,   # val_diff = +0.7
        t3: -0.043,  # val_diff = +0.5
        t6: -0.443,  # val_diff = +0.1
        t7: -0.543,  # val_diff = 0.0
        t4: -0.843   # val_diff = -0.3
    }
    
    stats = {k: [] for k in weights.keys()}
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        pair = frozenset([a, b])
        
        if pair in weights:
            ta = sum(a)
            tb = sum(b)
            
            if ta > tb:
                winner_chosen = 1 if row['response'] == 0 else 0
            else:
                winner_chosen = 1 if row['response'] == 1 else 0
                
            stats[pair].append(winner_chosen)
            
    score = 0.0
    for pair, w in weights.items():
        if stats[pair]:
            score += w * np.mean(stats[pair])
            
    return float(score)
```

**Observed (real) value:** -0.0085 (var=0.0694)
**Predicted under pi_3:** -0.0060 (var=0.0495)
**Predicted under pi_6:** 0.1891 (var=0.0524)

### Experiment 10
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trials 1-5: Option A is the tally winner.
    # Trial 6: Option B is the tally winner.
    m1 = (a_tuples == (1, 1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 0, 0, 1, 1))
    m2 = (a_tuples == (1, 0, 1, 1, 0, 0)) & (b_tuples == (0, 1, 0, 0, 1, 0))
    m3 = (a_tuples == (0, 1, 1, 0, 1, 0)) & (b_tuples == (1, 0, 0, 1, 0, 0))
    m4 = (a_tuples == (0, 0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 0, 1))
    m5 = (a_tuples == (0, 0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0, 0))
    m6 = (a_tuples == (1, 1, 0, 0, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1, 1))
    
    y1 = (data.loc[m1, 'response'] == 0).mean()
    y2 = (data.loc[m2, 'response'] == 0).mean()
    y3 = (data.loc[m3, 'response'] == 0).mean()
    y4 = (data.loc[m4, 'response'] == 0).mean()
    y5 = (data.loc[m5, 'response'] == 0).mean()
    y6 = (data.loc[m6, 'response'] == 1).mean()
    
    Y = np.array([y1, y2, y3, y4, y5, y6], dtype=float)
    if np.isnan(Y).any():
        return 0.0
        
    # X represents the validity advantage of the tally winner in each trial.
    # T1: 1.6, T2: 0.9, T3: 0.7, T4: 0.5, T5: -0.2, T6: -0.2
    X = np.array([1.6, 0.9, 0.7, 0.5, -0.2, -0.2])
    
    vx = X - np.mean(X)
    vy = Y - np.mean(Y)
    
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    if denom == 0:
        return 0.0
        
    return float(np.sum(vx * vy) / denom)
```

**Observed (real) value:** 0.6551 (var=0.2215)
**Predicted under pi_3:** 0.2680 (var=0.1628)
**Predicted under pi_6:** 0.8693 (var=0.1995)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, we see a consistent pattern: the observed data falls between the predictions of Pure Tallying (pi_3) and the Probabilistic Strategy Mixture (pi_6). In experiments where the mixture model predicts a strong validity effect (e.g., Exp 9), the observed data shows almost zero effect, closely matching Pure Tallying. However, in other experiments (e.g., Exp 7, Exp 10), there is a small but reliable validity gradient that Pure Tallying fails to capture. The Mixture model (pi_6) overestimates the validity effect because it assumes that when subjects use WADD, they use the exact objective validities. The data suggests that subjects do not probabilistically switch between pure Tallying and exact WADD. Instead, they likely use a single compensatory strategy with subjective weights that are heavily compressed towards equality.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Compressive Weighted Additive' theory. Instead of a probabilistic mixture of Pure Tallying and exact WADD, this theory posits that subjects always use a weighted additive strategy, but their subjective feature weights are heavily compressed toward equal weighting (e.g., via a parameter that interpolates between uniform weights and objective validities). This naturally explains why decisions look overwhelmingly like Tallying (due to strong compression) but still exhibit small, reliable validity correlations (due to the residual gradient in subjective weights), fitting the intermediate values observed across the experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 10555,
  "candidates_token_count": 334,
  "total_token_count": 11988
}
```
