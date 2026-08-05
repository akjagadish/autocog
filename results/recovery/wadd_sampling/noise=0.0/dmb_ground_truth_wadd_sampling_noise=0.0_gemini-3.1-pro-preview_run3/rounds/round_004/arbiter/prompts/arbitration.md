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
Weighted Additive (WADD) Theory: Decision-makers evaluate options using a fully compensatory strategy. They multiply each feature's value by its corresponding cue validity and sum these products to form an overall subjective value for each option. The option with the higher weighted sum is chosen. This allows multiple lower-validity cues to collectively outweigh a single high-validity cue, capturing behavior that falls between pure Take The Best and pure Tallying. To account for empirical response noise, the decision process incorporates a moderate degree of stochasticity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_6
Contingent Heuristic Theory (Difference-based Compensatory): Decision-makers conditionally apply decision strategies based on the relative strength of the best available evidence compared to the next best piece of evidence. They identify the most valid cue that discriminates between the two options and the second most valid discriminating cue. If the difference in validity between these two cues exceeds a subjective threshold (indicating the top cue is highly salient relative to the rest), they rely entirely on the top cue (Take-The-Best). If the difference is not sufficiently large, they fall back to a fully compensatory Weighted Additive (WADD) strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    cue_order = np.argsort(-val, kind='stable')
    
    ttb_winner = None
    v_max = 0.0
    v_next = 0.0
    found_first = False
    
    for cue in cue_order:
        if a[cue] != b[cue]:
            if not found_first:
                v_max = val[cue]
                ttb_winner = 0 if a[cue] > b[cue] else 1
                found_first = True
            else:
                v_next = val[cue]
                break
                
    delta = float(parameters["delta"])
    
    if ttb_winner is not None and (v_max - v_next) >= delta:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
    else:
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        scores = np.array([score_a, score_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design dissociates the pure Weighted Additive (WADD) theory from the Contingent Heuristic theory by systematically varying the validity gap between the best and second-best discriminating cues. WADD evaluates options via a fully compensatory sum across all trials. In contrast, the Contingent Heuristic falls back to WADD only when the validity gap is smaller than a subjective threshold (delta); when the gap exceeds this threshold, it abruptly switches to a non-compensatory Take-The-Best (TTB) strategy. By including critical trials with large validity gaps where the top cue favors option A but the sum of remaining cues strongly favors option B, we can observe whether choice probabilities reflect a continuous compensatory evaluation (WADD) or a discrete threshold-based switch to a single-cue strategy (Contingent).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates a specific trial where Option A is supported only by the most valid cue (0.95), while Option B is supported by three lower-validity cues (0.65, 0.55, 0.50). Under the pure Weighted Additive (WADD) theory, the compensatory sum strongly favors Option B (1.70 vs 0.95), meaning WADD will reliably choose B. Under the Contingent Heuristic theory, the validity gap between the best discriminating cue (0.95) and the next best (0.65) is 0.3. Since this gap is often larger than the subjective threshold delta (drawn from [0.0, 0.5]), the Contingent model frequently switches to Take-The-Best and chooses Option A. Thus, the proportion of Option A choices on this trial will be near 0 for WADD, but substantially higher for the Contingent Heuristic.

Source:
def metric(data: pd.DataFrame) -> float:
    a_vals = data['option_a_ratings'].apply(tuple)
    b_vals = data['option_b_ratings'].apply(tuple)
    
    # Target trial: A=[1, 0, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t_a = (1, 0, 0, 0, 0)
    t_b = (0, 0, 1, 1, 1)
    
    mask = (a_vals == t_a) & (b_vals == t_b)
    
    if mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2600 (var=0.0246)
- Predicted under pi_6 (simulated): 0.5267 (var=0.1115)
- Observed on real data: 0.2833 (var=0.0656)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Rationale:** This design systematically dissociates the Contingent Heuristic theory from the pure Weighted Additive (WADD) theory by manipulating the validity gap between the best and second-best discriminating cues. We use a set of five validities with a large gap between the top two and the bottom three, and a small gap between the top two themselves. In critical 'large gap' trials, the top discriminating cue strongly favors Option A, but the sum of the remaining cues favors Option B. The Contingent Heuristic theory predicts that subjects will notice the large gap, switch to a non-compensatory Take-The-Best strategy, and choose A. Pure WADD, however, always integrates all cues and predicts a choice of B. In 'small gap' trials, the gap between the top two discriminating cues is tiny, so the Contingent Heuristic falls back to WADD, predicting a choice of B (agreeing with pure WADD). Thus, the Contingent theory uniquely predicts a preference reversal based on the validity gap.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trials 1 and 2, there is a 'large gap' between the best discriminating cue and the next best discriminating cue. The Contingent Heuristic Theory predicts that subjects will notice this gap, rely on a non-compensatory Take-The-Best strategy, and choose Option A because it is favored by the top cue. Conversely, the purely compensatory Weighted Additive (WADD) Theory integrates all cues, and since the sum of the remaining cues heavily favors Option B, WADD overwhelmingly predicts a choice of Option B. Therefore, the proportion of times Option A is chosen on these specific trials will be significantly higher under the Contingent Heuristic Theory than under WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify critical trials with a large validity gap where WADD and Contingent diverge.
    # Trial 1: A=[1, 1, 0, 0, 0] vs B=[0, 1, 1, 1, 1]
    t1 = (a_str == '11000') & (b_str == '01111')
    # Trial 2: A=[0, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    t2 = (a_str == '01000') & (b_str == '00111')
    
    target_trials = data[t1 | t2]
    
    if len(target_trials) == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((target_trials['response'] == 0).mean())


### RESULTS
- Predicted under pi_3 (simulated): 0.2700 (var=0.0229)
- Predicted under pi_6 (simulated): 0.6217 (var=0.1019)
- Observed on real data: 0.2400 (var=0.0298)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    diff = A - B
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    
    first_diff_idx = np.argmax(is_diff, axis=1)
    ttb_choice = np.where(diff[np.arange(len(diff)), first_diff_idx] == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    
    tally_choice = np.full(len(data), -1)
    tally_choice[b_wins > a_wins] = 1
    tally_choice[a_wins > b_wins] = 0
    
    disagree = (has_diff) & (tally_choice != -1) & (ttb_choice != tally_choice)
    
    if np.sum(disagree) == 0:
        return 0.5
        
    responses = data['response'].values
    match = (responses[disagree] == ttb_choice[disagree])
    
    return float(np.mean(match))

```

**Observed (real) value:** 0.3450 (var=0.0120)
**Predicted under pi_3:** 0.3000 (var=0.0083)
**Predicted under pi_6:** 0.2753 (var=0.0356)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_align = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_tally = np.sum(a > b)
        b_tally = np.sum(b > a)
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if a_tally > b_tally:
            tally_winner = 0
        elif b_tally > a_tally:
            tally_winner = 1
        else:
            tally_winner = None
            
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            disagree_count += 1
            if row['response'] == tally_winner:
                tally_align += 1
                
    if disagree_count == 0:
        return 0.5
    return float(tally_align / disagree_count)
```

**Observed (real) value:** 0.6887 (var=0.0239)
**Predicted under pi_3:** 0.6925 (var=0.0093)
**Predicted under pi_6:** 0.5744 (var=0.0822)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: WADD favors A, Tallying favors B
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 0:
                wadd_consistent += 1
            total += 1
        # Trial 2: WADD favors B, Tallying favors A
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                wadd_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_consistent / total
```

**Observed (real) value:** 0.5667 (var=0.0904)
**Predicted under pi_3:** 0.5756 (var=0.0118)
**Predicted under pi_6:** 0.6578 (var=0.0193)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify critical dissociation trials where Tallying and WADD predict opposite choices.
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Tallying picks A (3 wins vs 2 wins), WADD picks B (1.65 vs 1.90)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1, 1])
    
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    # Tallying picks B (2 wins vs 3 wins), WADD picks A (1.90 vs 1.65)
    is_trial_2 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0, 0])
    
    # Count Tallying-consistent choices
    t1_consistent = (data[is_trial_1]['response'] == 0).sum()
    t2_consistent = (data[is_trial_2]['response'] == 1).sum()
    
    total_relevant = is_trial_1.sum() + is_trial_2.sum()
    
    if total_relevant == 0:
        return 0.5
        
    return float((t1_consistent + t2_consistent) / total_relevant)
```

**Observed (real) value:** 0.3962 (var=0.0872)
**Predicted under pi_3:** 0.4119 (var=0.0112)
**Predicted under pi_6:** 0.2806 (var=0.0180)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.75, 0.7, 0.65, 0.6])
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    score_A = np.dot(A, val)
    score_B = np.dot(B, val)
    wadd_choice = (score_B > score_A).astype(int)
    return float(np.mean(data['response'] == wadd_choice))
```

**Observed (real) value:** 0.7029 (var=0.0117)
**Predicted under pi_3:** 0.7469 (var=0.0101)
**Predicted under pi_6:** 0.6356 (var=0.0408)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    dissociation_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # WADD prediction
        score_a = np.sum(a * val)
        score_b = np.sum(b * val)
        wadd_pred = 0 if score_a > score_b else 1
        
        # Only evaluate on dissociation trials where the two models disagree
        if ttb_pred != wadd_pred and ttb_pred != -1:
            dissociation_matches.append(1 if row['response'] == ttb_pred else 0)
            
    if len(dissociation_matches) == 0:
        return 0.5
        
    return float(np.mean(dissociation_matches))
```

**Observed (real) value:** 0.3672 (var=0.0421)
**Predicted under pi_3:** 0.3247 (var=0.0087)
**Predicted under pi_6:** 0.3347 (var=0.0477)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def chose_ttb_option(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Identify trials where one option is exactly (1, 0, 0, 0, 0) 
        # and the other has 3 or more positive cues (Trials 5 and 7).
        if a == (1, 0, 0, 0, 0) and sum(b) >= 3:
            return 1.0 if row['response'] == 0 else 0.0
        elif b == (1, 0, 0, 0, 0) and sum(a) >= 3:
            return 1.0 if row['response'] == 1 else 0.0
        else:
            return np.nan

    choices = data.apply(chose_ttb_option, axis=1)
    return float(choices.dropna().mean())
```

**Observed (real) value:** 0.2583 (var=0.0615)
**Predicted under pi_3:** 0.2225 (var=0.0126)
**Predicted under pi_6:** 0.2483 (var=0.0568)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Identify trials where Option A has the highest-validity cue (cue 0) and Option B does not.
    # In the experimental design, these correspond to trials 1, 3, 5, and 7.
    # For all these trials, the Weighted Additive (WADD) score actually favors Option B.
    mask = data['option_a_ratings'].apply(lambda x: x[0] == 1) & data['option_b_ratings'].apply(lambda x: x[0] == 0)
    sub_data = data[mask]
    if len(sub_data) == 0:
        return 0.5
    
    # Return the proportion of times Option A was chosen.
    return float(np.mean(sub_data['response'] == 0))
```

**Observed (real) value:** 0.4358 (var=0.0429)
**Predicted under pi_3:** 0.4367 (var=0.0067)
**Predicted under pi_6:** 0.5042 (var=0.0550)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Weighted Additive / WADD) consistently and accurately predicts the observed data, whereas Theory 2 (Contingent Heuristic) significantly overpredicts the use of Take-The-Best (TTB) in trials with large validity gaps. For example, in Experiments 1 and 2, the observed proportions of Option A choices were 0.2833 and 0.2400, respectively, closely matching WADD's predictions (0.2600 and 0.2700) but falling far short of the Contingent Heuristic's predictions (0.5267 and 0.6217). The data strongly suggests that decision-makers are engaging in fully compensatory integration of cues rather than discrete, threshold-based switching to a single-cue strategy.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Rank-Dependent Weighting' (or Rank-based WADD) theory. Instead of assuming decision-makers use the exact numerical validities provided (as in pure WADD), this theory posits that they rank the cues by validity and assign subjective weights based strictly on their ordinal rank (e.g., using exponentially decaying weights or 1/rank). This captures the compensatory behavior observed in the data without assuming perfect mathematical integration of the stated probabilities, offering a cognitively plausible and strong alternative to Theory 1."
}
```

## Usage

```json
{
  "prompt_token_count": 9634,
  "candidates_token_count": 323,
  "total_token_count": 11061
}
```
