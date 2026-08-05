# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
High-Lapse Take-The-Best (TTB) Theory: Subjects generally find the multi-attribute choice task tedious without trial-by-trial feedback, leading to a very high rate of random guessing (high lapse rate). However, on the small fraction of trials where they actively engage, they do not integrate all features or tally them. Instead, they use a fast-and-frugal lexicographic heuristic (Take-The-Best): they inspect cues in descending order of their validity and make a deterministic choice based entirely on the single most valid cue that discriminates between the two options.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(val)[::-1]
    
    ttb_choice = -1
    for idx in order:
        if a[idx] > b[idx]:
            ttb_choice = 0
            break
        elif b[idx] > a[idx]:
            ttb_choice = 1
            break
            
    if ttb_choice == 0:
        p_core = np.array([1.0, 0.0])
    elif ttb_choice == 1:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
    
    # Mix the deterministic TTB choice with a high lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_5
High-Lapse Tallying Theory: Subjects find the multi-attribute binary feature task cognitively demanding, leading to a very high rate of random guessing. However, when they do engage with the stimulus, they rely on a simple unweighted tally of positive features (Tallying) rather than integrating complex cue validities. This produces a very weak but non-zero behavioral signal that slightly favors options with a higher simple count of positive features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate High-Lapse Take-The-Best (TTB) from High-Lapse Tallying, we exploit their core decision rules when subjects are engaged. TTB relies solely on the highest-validity discriminating cue, whereas Tallying relies on the unweighted sum of positive features. The trials are constructed to pit these two mechanisms directly against each other: in several trials, one option possesses the single most valid cue (favored by TTB) while the other option possesses a greater total number of lower-validity cues (favored by Tallying). We also include trials where Tallying predicts a strict tie (equal number of positive features) but TTB predicts a clear preference based on validity rank.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice matches the prediction of the Take-The-Best (TTB) heuristic. Because the experimental design heavily features trials where TTB and Tallying make diametrically opposed predictions, the TTB theory will yield a match rate significantly above 0.5 (even with high lapsing), whereas the Tallying theory will yield a match rate significantly below 0.5. This clear directional split ensures robust discrimination between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        if ttb_pred != -1:
            if ttb_pred == resp:
                matches += 1
            total += 1
            
    return float(matches / total) if total > 0 else 0.5

### RESULTS
- Predicted under pi_6 (simulated): 0.5952 (var=0.0061)
- Predicted under pi_5 (simulated): 0.4531 (var=0.0036)
- Observed on real data: 0.4455 (var=0.0040)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate High-Lapse Tallying from High-Lapse TTB, we exploit the fundamental difference in their core decision mechanisms while holding the assumption of a high lapse rate constant. High-Lapse Tallying relies on an unweighted sum of positive features, whereas High-Lapse TTB relies exclusively on the highest-validity discriminating cue. We construct trials that pit these two mechanisms directly against each other: in several trials, one option possesses the single most valid cue (favored by TTB) while the other option possesses a strictly greater total number of lower-validity cues (favored by Tallying). We also include trials where Tallying predicts a strict tie (equal number of positive features) but TTB predicts a clear preference based on validity rank.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the High-Lapse Tallying theory on trials where the two options have a different number of positive features. By experimental design, the trials where the tally differs (Trials 1, 2, 4, and 5) are constructed such that Take-The-Best (TTB) predicts the exact opposite choice of Tallying. Thus, data simulated under Tallying will yield a metric value > 0.5 (as the weak signal pushes choices towards the higher tally), whereas data simulated under TTB will yield a metric value < 0.5 (as its weak signal pushes choices towards the lower tally but higher validity cue). This structural opposition guarantees strong discrimination between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tallies = data['option_a_ratings'].apply(sum)
    b_tallies = data['option_b_ratings'].apply(sum)
    
    diff_mask = a_tallies != b_tallies
    data_diff = data[diff_mask]
    
    if len(data_diff) == 0:
        return 0.5
        
    tally_pred = (b_tallies[diff_mask] > a_tallies[diff_mask]).astype(int)
    match = (data_diff['response'] == tally_pred).mean()
    
    return float(match)

### RESULTS
- Predicted under pi_6 (simulated): 0.3895 (var=0.0039)
- Predicted under pi_5 (simulated): 0.5708 (var=0.0052)
- Observed on real data: 0.5384 (var=0.0057)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    valid_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB compares cues in order 0 to n_features-1
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_matches += 1
            valid_trials += 1
            
    return float(ttb_matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4737 (var=0.0046)
**Predicted under pi_6:** 0.6072 (var=0.0066)
**Predicted under pi_5:** 0.4398 (var=0.0056)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    tally_a = np.sum(a_ratings > b_ratings, axis=1)
    tally_b = np.sum(b_ratings > a_ratings, axis=1)
    valid = tally_a != tally_b
    if not np.any(valid):
        return 0.5
    pred = np.where(tally_a > tally_b, 0, 1)
    return float((pred[valid] == data['response'].values[valid]).mean())
```

**Observed (real) value:** 0.5479 (var=0.0020)
**Predicted under pi_6:** 0.4008 (var=0.0060)
**Predicted under pi_5:** 0.5650 (var=0.0035)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    return wadd_choices / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0233)
**Predicted under pi_6:** 0.6150 (var=0.0115)
**Predicted under pi_5:** 0.4483 (var=0.0116)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify Trial 3: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Tallying favors A (3 vs 2), meaning response should be closer to 0.
    # WADD with gamma > 1 favors B (due to the 0.95 and 0.85 validities), meaning response closer to 1.
    is_trial_3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 4: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 1]
    # Tallying favors B (2 vs 3), meaning response should be closer to 1.
    # WADD with gamma > 1 favors A (due to the 0.95 validity), meaning response closer to 0.
    is_trial_4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    
    resp_3 = data.loc[is_trial_3, 'response'].mean()
    resp_4 = data.loc[is_trial_4, 'response'].mean()
    
    if pd.isna(resp_3):
        resp_3 = 0.5
    if pd.isna(resp_4):
        resp_4 = 0.5
        
    # Tallying: resp_4 (high) - resp_3 (low) -> Positive
    # WADD: resp_4 (low) - resp_3 (high) -> Negative
    return float(resp_4 - resp_3)
```

**Observed (real) value:** -0.0025 (var=0.0589)
**Predicted under pi_6:** -0.1925 (var=0.0389)
**Predicted under pi_5:** 0.0650 (var=0.0244)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    
    mask = score_a != score_b
    if not np.any(mask):
        return 0.5
        
    expected = (score_b > score_a).astype(int)
    actual = data['response'].values
    
    return float(np.mean(expected[mask] == actual[mask]))
```

**Observed (real) value:** 0.5743 (var=0.0037)
**Predicted under pi_6:** 0.5433 (var=0.0027)
**Predicted under pi_5:** 0.5471 (var=0.0035)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    keys = [tuple(a) + tuple(b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_key': keys,
        'response': data['response']
    })
    grouped = df.groupby(['subject_id', 'trial_key'])['response'].mean()
    return float((grouped - 0.5).abs().mean())
```

**Observed (real) value:** 0.1371 (var=0.0006)
**Predicted under pi_6:** 0.1462 (var=0.0020)
**Predicted under pi_5:** 0.1269 (var=0.0010)

### Experiment 9
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    
    mask = (a_sums >= b_sums) & (a_first == 1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.5300 (var=0.0073)
**Predicted under pi_6:** 0.5863 (var=0.0077)
**Predicted under pi_5:** 0.5246 (var=0.0066)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5, where Option A is strictly dominant (all 1s) and Option B has all 0s.
    is_trial_5 = data['option_a_ratings'].apply(lambda x: sum(x) == 5)
    
    if not is_trial_5.any():
        return 0.5
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float((data.loc[is_trial_5, 'response'] == 0).mean())
```

**Observed (real) value:** 0.5877 (var=0.0189)
**Predicted under pi_6:** 0.6046 (var=0.0227)
**Predicted under pi_5:** 0.5831 (var=0.0206)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 2 (High-Lapse Tallying) consistently outperforms Theory 1 (High-Lapse Take-The-Best). In Experiment 1, TTB predicted a match rate of ~0.60, but the observed rate was 0.44, closely matching Tallying's prediction of 0.45. In Experiment 2, Tallying correctly predicted a metric > 0.5 (observed 0.538), while TTB predicted < 0.5. This pattern holds across Experiments 3, 4, 5, and 9, where Tallying's predictions tightly track the observed data. The evidence strongly contradicts the lexicographic TTB mechanism in this high-lapse setting, favoring a compensatory mechanism like Tallying. However, there are slight deviations in Experiments 6 and 7, suggesting that subjects might be sensitive to cue validities in a compensatory way rather than just doing a strict unweighted tally.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'High-Lapse Weighted Additive (WADD)' theory. The current TTB theory is degenerate because its lexicographic stopping rule contradicts the observed compensatory behavior. A WADD theory would posit that when subjects engage, they integrate all features weighted by their given validities, rather than just counting them (Tallying) or stopping at the best cue (TTB). Mixed with a high lapse rate, this could capture the baseline success of Tallying while better explaining the subtle variance in trials where high-validity cues might slightly override pure tally counts."
}
```

## Usage

```json
{
  "prompt_token_count": 7868,
  "candidates_token_count": 362,
  "total_token_count": 9265
}
```
