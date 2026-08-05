# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People use a non-compensatory, lexicographic heuristic called 'Take The Best' (TTB) to choose between options. They search through features in descending order of their subjective validity and stop at the first feature that discriminates between the two options (i.e., one option has a positive rating and the other does not). The choice is based entirely on this single discriminating cue, completely ignoring all remaining features, regardless of how many lower-validity cues might favor the alternative.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    chosen = -1
    for idx in order:
        if a[idx] > b[idx]:
            chosen = 0
            break
        elif b[idx] > a[idx]:
            chosen = 1
            break
            
    # Deterministic choice based on the first discriminating cue
    if chosen == 0:
        p_core = np.array([1.0, 0.0])
    elif chosen == 1:
        p_core = np.array([0.0, 1.0])
    else:
        # If all features tie, guess randomly
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend deterministic choice with uniform lapse rate for noise
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Softmax-Weighted Additive Model: Subjects use a fully compensatory integration process where they calculate a weighted sum of all features, but they subjectively scale the stated cue validities using a softmax transformation. By exponentiating and normalizing the validities with a scaling parameter, differences between high and low validity cues are amplified without shrinking the overall magnitudes of the weights. This allows a purely compensatory, additive mechanism to rival and approximate strict non-compensatory behavior (like Take-The-Best) when the scaling parameter is sufficiently large, while still integrating all available information.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Model expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax transformation of validities to amplify importance of top cues
    # without shrinking the overall magnitudes of the weights to zero.
    z_v = gamma * validities
    z_v = z_v - np.max(z_v)
    e_v = np.exp(z_v)
    weights = e_v / np.sum(e_v)
    
    # Compensatory integration (Weighted Additive)
    scores = stim @ weights

    # Softmax decision rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    # Blended with a uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1]

**Rationale:** To mimic TTB's strict non-compensatory choices on top-tier cues (e.g., cue 1 vs cues 2-5), Softmax-WADD must use a very large scaling parameter (gamma) to ensure the top weight outweighs the sum of all lower weights. However, doing so exponentially squashes the weights of lower-tier cues to near-zero. Consequently, when choices depend exclusively on these lower-tier cues (e.g., cue 4 vs cue 5), Softmax-WADD predicts near-random guessing because the score differences become vanishingly small and cannot be salvaged even by the maximum inverse-temperature (beta) parameter. TTB, in contrast, applies the same lexicographic rule uniformly, predicting strong preferences regardless of whether the discrimination happens at the top or bottom of the validity hierarchy.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
To mimic TTB on top-tier cues, Softmax-WADD requires a large scaling parameter (gamma) which exponentially squashes the weights of lower-tier cues to near-zero. On Trial 4, where the choice is entirely dependent on the two lowest-validity cues (Cue 4 vs Cue 5), Softmax-WADD will predict near-random guessing (0.5) because the score differences become vanishingly small. TTB, on the other hand, applies the same non-compensatory rule uniformly and will strongly prefer the option with Cue 4. This metric computes the proportion of TTB-consistent choices on Trial 4.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def check_ttb_match(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 4: Cue 4 vs Cue 5
        if a == (0, 0, 0, 1, 0) and b == (0, 0, 0, 0, 1):
            return 1 if row['response'] == 0 else 0
        elif a == (0, 0, 0, 0, 1) and b == (0, 0, 0, 1, 0):
            return 1 if row['response'] == 1 else 0
            
        return np.nan
        
    matches = data.apply(check_ttb_match, axis=1).dropna()
    if len(matches) == 0:
        return 0.5
    return float(matches.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8633 (var=0.0119)
- Predicted under pi_5 (simulated): 0.4967 (var=0.0175)
- Observed on real data: 0.7967 (var=0.0217)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 9: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Softmax-Weighted Additive Model (Softmax-WADD) from Take-The-Best (TTB) by exploiting Softmax-WADD's core mechanism: the exponential transformation of validities. Unlike standard TTB, which strictly ignores all lower-tier cues regardless of their quantity, Softmax-WADD allows for compensatory integration where a coalition of lower-validity cues can outweigh a single higher-validity cue if the sum of their exponentiated weights is large enough. We use evenly spaced validities to test this. Crucial trials pit the highest-validity cue against multiple lower-validity cues. TTB will deterministically choose the option favored by the single best discriminating cue in all such cases. Softmax-WADD, however, predicts a compensatory preference for the option with multiple lower-validity cues, but importantly, this preference strength will scale non-linearly depending on the exact combination of cue validity differences, a signature unique to the softmax transformation.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the overall proportion of choices that deviate from the deterministic Take-The-Best (TTB) prediction. TTB will only deviate due to its uniform lapse rate (epsilon / 2). Softmax-WADD, however, will deviate much more frequently because its compensatory nature allows coalitions of lower-validity cues to outweigh a single higher-validity cue (especially when the scaling parameter 'gamma' is not extremely large), and its softmax decision rule introduces additional stochasticity depending on the 'beta' parameter.

Source:
def metric(data: pd.DataFrame) -> float:
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5

    ttb_preds = [ttb_choice(row) for _, row in data.iterrows()]
    return float((data['response'] != ttb_preds).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1229 (var=0.0079)
- Predicted under pi_5 (simulated): 0.1951 (var=0.0020)
- Observed on real data: 0.1662 (var=0.0111)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers A (0), WADD prefers B (1)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        # Trial 2: Tallying prefers B (1), WADD prefers A (0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.1311 (var=0.0202)
**Predicted under pi_3:** 0.1222 (var=0.0114)
**Predicted under pi_5:** 0.0489 (var=0.0035)

### Experiment 4
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: Tallying prefers A (3 vs 2 wins), WADD prefers B (higher validity sum)
    t1_a = (0, 0, 0, 1, 1, 1)
    
    # Trial 5: Tallying prefers B (3 vs 2 wins), WADD prefers A (higher validity sum)
    t5_a = (1, 1, 0, 0, 0, 0)
    
    t1_mask = a_tuples == t1_a
    t5_mask = a_tuples == t5_a
    
    t1_resp = data.loc[t1_mask, 'response']
    t5_resp = data.loc[t5_mask, 'response']
    
    if len(t1_resp) == 0 or len(t5_resp) == 0:
        return 0.0
        
    p_a_t1 = (t1_resp == 0).mean()
    p_a_t5 = (t5_resp == 0).mean()
    
    # Tallying: P(A|T1) is high, P(A|T5) is low -> Positive difference
    # WADD: P(A|T1) is low, P(A|T5) is high -> Negative difference
    return float(p_a_t1 - p_a_t5)
```

**Observed (real) value:** -0.6650 (var=0.0405)
**Predicted under pi_3:** -0.7075 (var=0.0449)
**Predicted under pi_5:** -0.9025 (var=0.0101)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 pits an option A with 3 low-validity features against an option B with 2 high-validity features.
    # Tallying prefers A (3 wins vs 2 wins), whereas WADD prefers B (score 1.7 vs 1.8).
    is_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if is_target.sum() == 0:
        return 0.5
    return float((data.loc[is_target, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1600 (var=0.0219)
**Predicted under pi_3:** 0.1425 (var=0.0219)
**Predicted under pi_5:** 0.0475 (var=0.0062)

### Experiment 6
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify critical trials (Trial 1 and Trial 2) where A is [0, 0, 0, 1, 1, 1]
    # and B has the first feature as 1 (either [1, 1, 0, 0, 0, 0] or [1, 0, 1, 0, 0, 0]).
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_B_starts_1 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    mask = is_A_target & is_B_starts_1
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1333 (var=0.0092)
**Predicted under pi_3:** 0.1267 (var=0.0090)
**Predicted under pi_5:** 0.0583 (var=0.0040)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_choices = 0
    total = 0
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) > 0:
            first_idx = non_zero[0]
            if diff[first_idx] > 0 and np.sum(b) > np.sum(a):
                total += 1
                if resp == 0:
                    ttb_choices += 1
            elif diff[first_idx] < 0 and np.sum(a) > np.sum(b):
                total += 1
                if resp == 1:
                    ttb_choices += 1
                    
    return float(ttb_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8375 (var=0.0070)
**Predicted under pi_3:** 0.8883 (var=0.0058)
**Predicted under pi_5:** 0.7196 (var=0.0091)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if not mask.any():
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.1500 (var=0.0087)
**Predicted under pi_3:** 0.1217 (var=0.0073)
**Predicted under pi_5:** 0.2108 (var=0.0058)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Strict TTB always predicts option A (response 0) for all trials in this design
    # because the first discriminating cue always favors A.
    # Therefore, any response of 1 (choosing B) is due to noise.
    # PTTB will have a much higher rate of choosing B due to validity swaps
    # (especially when the gap between validities is small) and softmax smoothing.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.1229 (var=0.0065)
**Predicted under pi_3:** 0.1260 (var=0.0054)
**Predicted under pi_5:** 0.4440 (var=0.0043)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1))
    
    # Response is 0 for A, 1 for B. 
    # mean(response) is the rate of choosing option B.
    mean_t1 = data[t1_mask]['response'].mean()
    mean_t2 = data[t2_mask]['response'].mean()
    
    return float(mean_t1 - mean_t2)
```

**Observed (real) value:** 0.0033 (var=0.0153)
**Predicted under pi_3:** 0.0033 (var=0.0180)
**Predicted under pi_5:** -0.0117 (var=0.0169)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Take-The-Best) consistently and significantly outperforms Theory 2 (Softmax-WADD) across almost all 10 experiments. The observed data strongly supports strict non-compensatory decision-making, as the empirical metrics align very closely with TTB's predictions. Theory 2 predicts too much compensatory behavior and struggles to capture the sharp, lexicographic nature of the participants' choices, failing particularly on Experiments 1, 3, 5, and 9 where its predictions diverge dramatically from the observed data. Because Theory 1 is already an excellent fit, Theory 2 should be replaced with a more plausible competitor that retains a largely non-compensatory structure but introduces a different source of stochasticity.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Softmax-WADD with a 'Probabilistic Take-The-Best' (or Noisy Cue Search) theory. In this new theory, the decision rule remains strictly non-compensatory (stopping at the first discriminating cue), but the search order is not perfectly deterministic. Instead, the subjective validities of the cues are corrupted by trial-by-trial Gaussian noise (or sampled via a softmax over validities without replacement). This allows for occasional rank-reversals in the search order\u2014especially between cues with similar validities\u2014providing a more cognitively plausible mechanism for errors than a simple uniform lapse rate, while still preserving the core lexicographic dynamic that explains the data so well."
}
```

## Usage

```json
{
  "prompt_token_count": 9077,
  "candidates_token_count": 329,
  "total_token_count": 10596
}
```
