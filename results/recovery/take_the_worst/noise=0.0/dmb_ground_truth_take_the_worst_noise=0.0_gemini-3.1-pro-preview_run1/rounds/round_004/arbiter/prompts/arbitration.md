# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Weighted Tallying: People compare options by computing a weighted sum of feature wins, where the weights are a heavily compressed function of the stated cue validities (w_i = validity_i ** gamma, with gamma close to 0). This allows the strategy to behave almost exactly like simple Tallying when one option has a clear numerical advantage in feature wins, but smoothly and compensatorily resolves ties using all available validity information rather than defaulting to a strict lexicographic tie-breaker.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Compress validities to create weights that are close to 1 (Tallying-like) 
    # but still retain ordinal validity information for compensatory tie-breaking.
    weights = val ** gamma
    
    # Calculate weighted feature wins (ignoring ties on individual features)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate Weighted Tallying from standard Tallying, we exploit trials where the two options have an equal number of feature wins (a tally tie). Standard Tallying completely ignores validities and predicts a 50/50 guess whenever the tallies are tied. Weighted Tallying, however, applies a heavy compression to the validities (gamma close to 0) which preserves ordinal validity information, allowing it to smoothly and compensatorily break ties. We include trials where the highest-validity cue is outweighed by multiple slightly-lower validity cues (compensatory tie-breaking), which separates Weighted Tallying from simple Lexicographic Tie-Breaking, as well as standard tie trials and unambiguous filler trials where both theories agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the two options have an equal number of feature wins (Tallying ties) but where Weighted Tallying's compressed validities break the tie in a specific predictable direction. Standard Tallying will guess 50/50 on these trials, yielding a metric value around 0.5. Weighted Tallying will systematically favor the option with the higher weighted sum, yielding a metric value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    wt_favored = {
        (1, 0, 0, 0, 1): 1,
        (0, 1, 1, 0, 0): 0,
        (1, 1, 0, 0, 0): 0,
        (0, 0, 1, 1, 0): 1,
        (1, 0, 0, 0, 0): 0,
        (0, 1, 0, 0, 0): 0
    }
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a_tuple = tuple(row['option_a_ratings'])
        if a_tuple in wt_favored:
            if row['response'] == wt_favored[a_tuple]:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
    return match_count / total_count

### RESULTS
- Predicted under pi_6 (simulated): 0.5550 (var=0.0042)
- Predicted under pi_2 (simulated): 0.5003 (var=0.0046)
- Observed on real data: 0.1872 (var=0.0127)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.8, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To cleanly dissociate standard Tallying from Weighted Tallying, we focus on trials where the two options are tied in their raw feature tallies. Standard Tallying predicts a uniform 50/50 guess on these trials because it completely ignores validities. Weighted Tallying, however, compresses validities (gamma close to 0) to maintain a Tallying-like behavior for unequal counts, but uses the retained ordinal validity information to smoothly break ties. By creating trials where the raw tally is tied but one option wins on features with higher validities, Weighted Tallying will systematically prefer that option while standard Tallying remains indifferent. We also include filler trials where one option wins the tally to ensure subjects remain engaged and to confirm that both models correctly predict the majority choices when tallies are unequal.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By focusing strictly on trials where the raw feature tallies are tied, we isolate the exact mechanism where the two theories diverge. Standard Tallying has no basis to prefer either option on these trials and must guess (producing an alignment with Weighted Tallying's preference of ~50%). Weighted Tallying, however, breaks the tie using compressed validity weights, systematically preferring one option over the other. The metric calculates the proportion of choices on these tie trials that align with Weighted Tallying's prediction.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.6, 0.5])
    weights = val ** 0.1
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    a_wins = A > B
    b_wins = B > A
    
    tally_a = a_wins.sum(axis=1)
    tally_b = b_wins.sum(axis=1)
    
    tie_mask = (tally_a == tally_b) & (tally_a > 0)
    
    if not tie_mask.any():
        return 0.5
        
    wt_a = (a_wins[tie_mask] * weights).sum(axis=1)
    wt_b = (b_wins[tie_mask] * weights).sum(axis=1)
    
    expected = np.where(wt_b > wt_a, 1, 0)
    valid = wt_a != wt_b
    
    if not valid.any():
        return 0.5
        
    actual = data['response'].values[tie_mask][valid]
    expected = expected[valid]
    
    return float(np.mean(actual == expected))

### RESULTS
- Predicted under pi_6 (simulated): 0.6239 (var=0.0105)
- Predicted under pi_2 (simulated): 0.4853 (var=0.0029)
- Observed on real data: 0.1633 (var=0.0099)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))
```

**Observed (real) value:** 0.1383 (var=0.0087)
**Predicted under pi_6:** 0.1614 (var=0.0108)
**Predicted under pi_2:** 0.1297 (var=0.0093)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8286 (var=0.0105)
**Predicted under pi_6:** 0.8748 (var=0.0081)
**Predicted under pi_2:** 0.8479 (var=0.0103)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A='11000', B='00111' -> WADD prefers A (1.9 vs 1.6), Tally prefers B (2 vs 3)
    # Trial 2: A='00111', B='11000' -> WADD prefers B (1.6 vs 1.9), Tally prefers A (3 vs 2)
    
    t1 = df[(df['A_str'] == '11000') & (df['B_str'] == '00111')]
    t2 = df[(df['A_str'] == '00111') & (df['B_str'] == '11000')]
    
    score = 0.0
    n = 0
    
    if len(t1) > 0:
        score += (t1['response'] == 0).sum()
        n += len(t1)
    if len(t2) > 0:
        score += (t2['response'] == 1).sum()
        n += len(t2)
        
    if n == 0:
        return 0.5
    return float(score / n)

```

**Observed (real) value:** 0.1333 (var=0.0128)
**Predicted under pi_6:** 0.1425 (var=0.0165)
**Predicted under pi_2:** 0.1667 (var=0.0219)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    wadd_chosen = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_chosen += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_chosen += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_chosen / total)
```

**Observed (real) value:** 0.1956 (var=0.0240)
**Predicted under pi_6:** 0.1578 (var=0.0203)
**Predicted under pi_2:** 0.1511 (var=0.0177)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of features for option A and option B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Select trials where Option A has fewer total positive features than Option B
    # Standard Tallying will consistently choose Option B on these trials.
    # Top-K Tallying will often choose Option A or tie, because Option A's positive features are concentrated in the top K most valid cues.
    mask = a_sums < b_sums
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(subset['response'] == 0))
```

**Observed (real) value:** 0.1283 (var=0.0075)
**Predicted under pi_6:** 0.1267 (var=0.0058)
**Predicted under pi_2:** 0.1172 (var=0.0062)

### Experiment 8
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Return the proportion of times Option A was chosen.
    # Tallying counts all features, meaning Option A wins on most trials (because it has more 1s overall in these specific pairs).
    # Top-K Tallying ignores the lowest validity features where Option A dominates, so Option B wins on most trials.
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.8588 (var=0.0081)
**Predicted under pi_6:** 0.6652 (var=0.0041)
**Predicted under pi_2:** 0.7265 (var=0.0054)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tie_mask = (a_wins == b_wins)
    
    if not np.any(tie_mask):
        return 0.5
        
    a_tie = a_mat[tie_mask]
    b_tie = b_mat[tie_mask]
    resp_tie = data['response'].values[tie_mask]
    
    diff = a_tie - b_tie
    
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    row_indices = np.arange(len(a_tie))
    first_diff_sign = diff[row_indices, first_diff_idx]
    
    favored = np.where(first_diff_sign == 1, 0, 1)
    consistent = (resp_tie == favored)
    
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.3300 (var=0.0032)
**Predicted under pi_6:** 0.5616 (var=0.0050)
**Predicted under pi_2:** 0.4969 (var=0.0048)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_matrix = np.array(data['option_a_ratings'].tolist())
    b_matrix = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_matrix > b_matrix, axis=1)
    b_wins = np.sum(b_matrix > a_matrix, axis=1)
    
    ties = (a_wins == b_wins)
    
    if not np.any(ties):
        return 0.5
        
    a_ties = a_matrix[ties]
    b_ties = b_matrix[ties]
    responses_ties = data['response'].values[ties]
    
    diffs = a_ties - b_ties
    
    matches = 0
    valid_trials = 0
    
    for i in range(len(diffs)):
        row_diff = diffs[i]
        nonzero = np.nonzero(row_diff)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choice = 0 if row_diff[first_diff_idx] > 0 else 1
            if responses_ties[i] == ttb_choice:
                matches += 1
            valid_trials += 1
            
    if valid_trials == 0:
        return 0.5
        
    return float(matches / valid_trials)
```

**Observed (real) value:** 0.4571 (var=0.0020)
**Predicted under pi_6:** 0.5435 (var=0.0053)
**Predicted under pi_2:** 0.5035 (var=0.0028)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories fail to capture the observed behavior on tally-tie trials. Experiments 1, 2, 9, and 10 specifically isolate trials where the raw feature counts (tallies) are equal. Standard Tallying (pi_2) predicts a 50/50 guess (metric ~0.5). Weighted Tallying (pi_6) attempts to break ties using compressed validities, predicting metric values of ~0.55-0.62. However, the observed data yields metrics around 0.16-0.18 in Exps 1 and 2, and 0.33-0.45 in Exps 9 and 10. These extreme deviations from 0.5 indicate that subjects are systematically using a tie-breaking mechanism that strictly opposes the predictions of Weighted Tallying as currently formulated. Since both models fundamentally mispredict the direction of the tie-breaking behavior, a brand-new theory is required to reconceptualize how subjects evaluate tied options.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that explains the systematic reversal observed on tally-tie trials. Instead of compressing validities to break ties compensatorily (which favors multiple moderately-high validities over a single highest validity), subjects might be using a strict Lexicographic / Take-The-Best (TTB) tie-breaker when tallies are equal. Alternatively, subjects might be exhibiting a 'feature-drop' heuristic where they ignore the lowest validity features completely when counts are tied. The new theory should combine a primary Tallying/Equal-Weight phase with a Lexicographic tie-breaker phase (e.g., 'Tally-then-TTB'), ensuring that when raw sums are equal, the option with the single highest-validity discriminating cue wins, rather than the option with a higher sum of compressed validities."
}
```

## Usage

```json
{
  "prompt_token_count": 9835,
  "candidates_token_count": 424,
  "total_token_count": 13393
}
```
