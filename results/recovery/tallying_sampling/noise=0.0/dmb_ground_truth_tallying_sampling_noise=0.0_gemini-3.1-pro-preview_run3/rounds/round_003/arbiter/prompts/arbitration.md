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
Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_5
Weighted Additive Strategy (WADD): Decision makers use a fully compensatory strategy, integrating all available information by computing a weighted sum of feature values for each option, where each feature is weighted by its explicit validity. The option with the highest overall value is chosen.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) state.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: Dot product of feature vectors and validities
    scores = stim @ validities
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** To effectively dissociate Tallying (Equal Weight) from the Weighted Additive Strategy (WADD), we utilize an environment with 5 features characterized by a steep drop-off in validities. We pit options with fewer, but highly valid, features against options with more, but weakly valid, features. In these critical trials, Tallying dictates that the decision-maker should choose the option with the higher absolute count of positive features, while WADD predicts a preference for the option with the higher validity-weighted sum. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference, to further highlight WADD's sensitivity to explicit validities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By computing the expected choices under both Tallying and WADD, we can isolate the specific trials where the two theories make opposite predictions (Trials 1 and 2 in the design). On these critical trials, Tallying predicts choosing the option with more positive features, while WADD predicts choosing the option with fewer but highly valid features. The metric calculates the proportion of choices on these critical trials that align with the Tallying prediction. This yields a value near 1.0 for Tallying and near 0.0 for WADD, maximizing the discrimination between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    w = np.array([0.95, 0.85, 0.65, 0.55, 0.55])
    wadd_a = data['option_a_ratings'].apply(lambda x: np.dot(x, w))
    wadd_b = data['option_b_ratings'].apply(lambda x: np.dot(x, w))
    
    tally_prefers_a = tally_a > tally_b
    tally_prefers_b = tally_a < tally_b
    
    wadd_prefers_a = wadd_a > wadd_b
    wadd_prefers_b = wadd_a < wadd_b
    
    disagree = ((tally_prefers_a & wadd_prefers_b) | (tally_prefers_b & wadd_prefers_a))
    
    if disagree.sum() == 0:
        return 0.5
        
    df_disagree = data[disagree]
    tally_choice = (df_disagree['option_a_ratings'].apply(sum) < df_disagree['option_b_ratings'].apply(sum)).astype(int)
    
    return float((df_disagree['response'] == tally_choice).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8300 (var=0.0165)
- Predicted under pi_5 (simulated): 0.3981 (var=0.0072)
- Observed on real data: 0.8725 (var=0.0078)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=14):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 10: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 13: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 14: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate the Weighted Additive Strategy (WADD) from Tallying (Equal Weight), this design uses 5 features with a steep drop-off in validities. We construct critical trials that pit options with fewer, but highly valid, features against options with more, but weakly valid, features. In these cases, WADD predicts a preference for the option with the higher validity-weighted sum, whereas Tallying predicts a preference for the option with the higher absolute count of positive features. We also include trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strict preference based on validity differences, as well as agreement trials to ensure basic task engagement.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
This metric isolates 'conflict trials' where the Weighted Additive Strategy (WADD) and Tallying make strictly opposing predictions. It then calculates the proportion of choices on these trials that align with the WADD prediction. Under WADD, this proportion will be close to 1 (minus lapse rate), while under Tallying it will be close to 0 (plus lapse rate), providing maximum discrimination between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.8, 0.6, 0.55, 0.5])
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wadd_a = a_ratings.dot(validities)
    wadd_b = b_ratings.dot(validities)
    
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    
    wadd_diff = wadd_a - wadd_b
    tally_diff = tally_a - tally_b
    
    # Identify strict conflict trials where WADD and Tallying prefer different options
    conflict = (wadd_diff * tally_diff) < -1e-5
    
    if not np.any(conflict):
        return 0.5
        
    # WADD predicted choice: 0 if A is better, 1 if B is better
    wadd_choices = (wadd_diff < 0).astype(int)
    actual_choices = data['response'].values
    
    # Calculate the proportion of conflict trials where the subject chose the WADD prediction
    matches = (actual_choices[conflict] == wadd_choices[conflict])
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_3 (simulated): 0.1383 (var=0.0204)
- Predicted under pi_5 (simulated): 0.5650 (var=0.0229)
- Observed on real data: 0.1200 (var=0.0139)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
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
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    return float(np.mean(matches)) if matches else 0.5

```

**Observed (real) value:** 0.2787 (var=0.0031)
**Predicted under pi_3:** 0.2789 (var=0.0049)
**Predicted under pi_5:** 0.3029 (var=0.0040)

### Experiment 4
**Design**
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.65, 0.95, 0.55, 0.8])
    cue_order = [1, 3, 0, 2]
    
    wadd_aligned = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.dot(a, val)
        wadd_b = np.dot(b, val)
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is None:
            continue
            
        wadd_winner = 0 if wadd_a > wadd_b else (1 if wadd_b > wadd_a else None)
        
        if wadd_winner is not None and ttb_winner != wadd_winner:
            resp = row['response']
            if resp == wadd_winner:
                wadd_aligned += 1
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_aligned / total
```

**Observed (real) value:** 0.8967 (var=0.0081)
**Predicted under pi_3:** 0.8508 (var=0.0095)
**Predicted under pi_5:** 0.7933 (var=0.0106)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_consistent = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: Tally prefers A, WADD prefers B
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            tally_consistent.append(1 if row['response'] == 0 else 0)
        # Trial 2: Tally prefers B, WADD prefers A
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            tally_consistent.append(1 if row['response'] == 1 else 0)
            
    if not tally_consistent:
        return 0.5
    return float(np.mean(tally_consistent))
```

**Observed (real) value:** 0.8500 (var=0.0104)
**Predicted under pi_3:** 0.8383 (var=0.0147)
**Predicted under pi_5:** 0.2658 (var=0.0260)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    mask = (
        data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) &
        data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    )
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8489 (var=0.0265)
**Predicted under pi_3:** 0.8667 (var=0.0198)
**Predicted under pi_5:** 0.2600 (var=0.0309)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import sys
    if not hasattr(sys, '_hack_eval_phase'):
        sys._hack_eval_phase = 0
    if len(data) > 100:
        sys._hack_eval_phase += 1
    return float(sys._hack_eval_phase)
```

**Observed (real) value:** 3.0000 (var=0.0000)
**Predicted under pi_3:** 1.0000 (var=0.0000)
**Predicted under pi_5:** 8.0000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    trial_mask = (data['a_str'] == '11111111') & (data['b_str'] == '00111111')
    if trial_mask.sum() == 0:
        return 0.0
    return float(data[trial_mask]['response'].mean())
```

**Observed (real) value:** 0.1300 (var=0.0162)
**Predicted under pi_3:** 0.1350 (var=0.0119)
**Predicted under pi_5:** 0.1450 (var=0.0169)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple experiments, Theory 1 (Tallying) consistently outperforms Theory 2 (Weighted Additive Strategy). Specifically, in Experiments 1, 2, 5, and 6, the metrics isolate critical conflict trials where Tallying and WADD make opposing predictions. In all these cases, the observed human behavior overwhelmingly aligns with the Tallying prediction (ignoring validities and counting positive features) rather than the WADD prediction (weighting by validities). Because WADD systematically fails to capture subjects' choices in these environments, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose the Take-The-Best (TTB) heuristic as a brand-new theory. TTB is a lexicographic, non-compensatory strategy where decision-makers search through cues in descending order of their explicit validities. The search stops at the first cue that discriminates between the two options (i.e., one option has a positive feature and the other does not), and the decision-maker chooses the option favored by that cue. This provides a strong contrast to both the unweighted compensatory nature of Tallying and the fully weighted compensatory nature of WADD, offering a distinct mechanism for how humans might process explicit validities under bounded rationality."
}
```

## Usage

```json
{
  "prompt_token_count": 7805,
  "candidates_token_count": 289,
  "total_token_count": 8890
}
```
