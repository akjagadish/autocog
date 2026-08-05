# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Take-The-Best (TTB) Heuristic

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        elif b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_core = np.array([0.5, 0.5])
    else:
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = float(parameters["beta"])
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


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
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To robustly dissociate Tallying from Take-The-Best (TTB), we must pit the sheer quantity of positive features against the highest-validity discriminating feature. Tallying ignores validities and simply counts the number of 1s, while TTB searches through features strictly in order of validity and stops at the first difference. The trials are designed such that one option has a clear advantage in the total number of positive features (favored by Tallying), but the other option possesses the single highest-validity feature that discriminates between the two (favored by TTB). We also include tie-breaking trials where Tallying sees an equal count but TTB has a strict preference, to further highlight the difference in their decision rules.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates 'conflict' trials where the Tallying heuristic and the Take-The-Best (TTB) heuristic make strictly opposing predictions. It computes the proportion of these conflict trials where the subject's choice aligns with the TTB prediction. Data generated by the TTB theory will yield a value close to 1 (minus noise), while data generated by the Tallying theory will yield a value close to 0 (plus noise), guaranteeing a large and stable discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_mat.sum(axis=1)
    sum_b = b_mat.sum(axis=1)
    
    tally_choice = np.zeros(len(data)) - 1
    tally_choice[sum_a > sum_b] = 0
    tally_choice[sum_b > sum_a] = 1
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    first_diff_idx = np.argmax(non_zero, axis=1)
    has_diff = np.any(non_zero, axis=1)
    
    ttb_choice = np.zeros(len(data)) - 1
    row_indices = np.arange(len(data))
    first_diff_vals = diff[row_indices, first_diff_idx]
    
    ttb_choice[(has_diff) & (first_diff_vals > 0)] = 0
    ttb_choice[(has_diff) & (first_diff_vals < 0)] = 1
    
    conflict = (tally_choice != -1) & (ttb_choice != -1) & (tally_choice != ttb_choice)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    matches = (responses[conflict] == ttb_choice[conflict])
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_3 (simulated): 0.4267 (var=0.0064)
- Predicted under pi_4 (simulated): 0.7261 (var=0.0194)
- Observed on real data: 0.4294 (var=0.0047)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** To robustly dissociate Take-The-Best (TTB) from Tallying, we pit the single highest-validity discriminating feature against the overall sum of positive features. TTB searches through features in descending order of validity and stops at the first difference, completely ignoring the remaining features. Tallying ignores validities and simply counts the total number of positive features. The critical dissociation trials are designed such that one option is favored by the highest-validity discriminating feature (chosen by TTB), while the other option possesses a greater total number of lower-validity positive features (chosen by Tallying). We also include trials where Tallying predicts a tie (equal number of positive features) but TTB has a strict preference, as well as agreement trials to serve as a baseline.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
We focus on trials where Tallying predicts a choice for option B (the sum of positive features is strictly greater for B than for A). Due to the experimental design, on all such trials, the highest validity discriminating feature favors option A, meaning Take-The-Best predicts a choice for option A. The metric calculates the proportion of choices for option A on these specific trials. Take-The-Best will yield a high value, whereas Tallying will yield a low value.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return np.nan
        
    return float(np.mean(data.loc[mask, 'response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.3997 (var=0.0059)
- Predicted under pi_4 (simulated): 0.7066 (var=0.0179)
- Observed on real data: 0.4069 (var=0.0041)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Features are already sorted by validity in descending order.
        # Find the first discriminating cue.
        ttb_choice = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            if resp == ttb_choice:
                ttb_consistent += 1
            total += 1
            
    return ttb_consistent / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4249 (var=0.0029)
**Predicted under pi_3:** 0.4247 (var=0.0036)
**Predicted under pi_4:** 0.7380 (var=0.0192)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = [0.9, 0.8, 0.7, 0.6]
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice
        ttb = -1
        for i in range(4):
            if a[i] > b[i]: 
                ttb = 0
                break
            elif b[i] > a[i]: 
                ttb = 1
                break
                
        # Expected WADD choice (with equal subjective weights)
        score_a = sum(a[i] * validities[i] for i in range(4))
        score_b = sum(b[i] * validities[i] for i in range(4))
        wadd = 0 if score_a > score_b else (1 if score_b > score_a else -1)
        
        # Only consider trials where TTB and WADD diverge
        if ttb != -1 and wadd != -1 and ttb != wadd:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.4027 (var=0.0071)
**Predicted under pi_3:** 0.4300 (var=0.0059)
**Predicted under pi_4:** 0.7050 (var=0.0209)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    def get_mask(a_target, b_target):
        m1 = (a_str == a_target) & (b_str == b_target)
        m2 = (a_str == b_target) & (b_str == a_target)
        return m1, m2
    
    t1_m1, t1_m2 = get_mask('00111', '11000')
    t5_m1, t5_m2 = get_mask('00110', '00001')
    t7_m1, t7_m2 = get_mask('01110', '10001')
    t8_m1, t8_m2 = get_mask('00111', '10100')
    
    def get_p_a(m1, m2):
        mask = m1 | m2
        if not mask.any():
            return 0.5
        chose_a = np.where(m1, data['response'] == 0, data['response'] == 1)
        return float(chose_a[mask].mean())
    
    p_a_t1 = get_p_a(t1_m1, t1_m2)
    p_a_t5 = get_p_a(t5_m1, t5_m2)
    p_a_t7 = get_p_a(t7_m1, t7_m2)
    p_a_t8 = get_p_a(t8_m1, t8_m2)
    
    group1_p = (p_a_t5 + p_a_t7) / 2.0
    group2_p = (p_a_t1 + p_a_t8) / 2.0
    
    return float(group1_p - group2_p)
```

**Observed (real) value:** -0.0383 (var=0.0218)
**Predicted under pi_3:** 0.0183 (var=0.0170)
**Predicted under pi_4:** 0.2325 (var=0.0240)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    t6_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    t1_wadd_match = (data['response'] == 0) & t1_mask
    t6_wadd_match = (data['response'] == 1) & t6_mask
    
    relevant_trials = t1_mask | t6_mask
    if not relevant_trials.any():
        return 0.5
        
    return float((t1_wadd_match.sum() + t6_wadd_match.sum()) / relevant_trials.sum())
```

**Observed (real) value:** 0.4012 (var=0.0049)
**Predicted under pi_3:** 0.4512 (var=0.0076)
**Predicted under pi_4:** 0.7106 (var=0.0257)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, Theory 1 (Tallying) consistently outperforms Theory 2 (Take-The-Best). The metrics isolate trials where compensatory strategies (like Tallying) and non-compensatory strategies (like TTB) make opposing predictions. The observed data consistently hovers around 0.40-0.43, indicating that subjects are slightly more likely to choose the option favored by compensatory strategies rather than the one favored by TTB. Theory 2 (TTB) consistently overpredicts the alignment with the lexicographic rule (predicting ~0.70-0.74). While Theory 1 captures the data well, an even stronger competitor would be a fully compensatory model that actually utilizes the cue validities, rather than ignoring them as Tallying does.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Take-The-Best with a Weighted Additive (WADD) theory. The WADD model posits that subjects compute an overall score for each option by multiplying each cue value by its corresponding stated validity, summing these weighted values, and then making a probabilistic choice (e.g., via softmax) based on the score differences. This introduces a fully compensatory strategy that respects cue validities, contrasting with both TTB (which is non-compensatory) and Tallying (which ignores validities)."
}
```

## Usage

```json
{
  "prompt_token_count": 6470,
  "candidates_token_count": 305,
  "total_token_count": 7941
}
```
