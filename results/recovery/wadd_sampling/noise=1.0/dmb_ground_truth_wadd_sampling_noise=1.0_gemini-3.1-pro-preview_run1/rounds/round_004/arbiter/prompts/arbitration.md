# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Decision-makers use a Compensatory / Weighted Additive (WADD) strategy, where an overall value for each option is computed by weighting each feature by a subjective transformation of its validity (validity raised to a power) and summing these weighted features. On strict conflict trials, the high weight of the single discriminating cue roughly balances the combined weights of multiple opposing lower-validity cues. This produces a near-zero difference in overall value, naturally resulting in a choice probability near 0.5 without requiring a mixture of distinct heuristics.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective transformation of validities
    w = val ** gamma
    
    val_a = np.sum(a * w)
    val_b = np.sum(b * w)
    
    beta = float(parameters["beta"])
    scores = np.array([val_a, val_b])
    
    z = beta * (scores - np.max(scores))
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    p = (1.0 - epsilon) * p + epsilon * 0.5
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
Conflict-Induced Guessing with Evidence Threshold: Decision-makers evaluate options using simple heuristics (Take The Best and Tallying) but also monitor the overall Weighted Additive (WADD) evidence. When the heuristics make strict opposing predictions, OR when the overall WADD evidence difference between the options is too small to confidently discriminate, the decision-maker experiences uncertainty and resorts to random guessing. This captures the pervasive ~0.5 choice probabilities and low variance across both strict heuristic conflict trials and trials with nominally agreeing heuristics but weak overall evidence.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    # Determine Take The Best (TTB) winner
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Determine Tallying winner
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Compute WADD difference
    wadd_a = np.sum(a * val)
    wadd_b = np.sum(b * val)
    wadd_diff = abs(wadd_a - wadd_b)
    threshold = float(parameters["threshold"])
        
    # Check for strict conflict or insufficient WADD evidence
    conflict = (ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner)
    
    if conflict or (wadd_diff <= threshold):
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: rely on the agreed winner (or the one that isn't tied)
        winner = ttb_winner if ttb_winner is not None else tally_winner
        if winner == 0:
            p_core = np.array([1.0, 0.0])
        elif winner == 1:
            p_core = np.array([0.0, 1.0])
        else:
            p_core = np.array([0.5, 0.5])
            
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** We use a 5-feature design with a mix of highly valid and low-validity cues to cleanly dissociate the Advocated Theory (WADD with exponentiated validities) from the Competing Theory (Conflict-Induced Guessing). In trials where Take The Best (TTB) and Tallying agree (e.g., Option A has the highest validity cue AND more cues overall), the Competing Theory predicts a preference for A (or 0.5 if the WADD difference is below threshold). However, the Advocated Theory strongly predicts B because the combined exponentiated weight of the 2nd and 3rd most valid cues outweighs the 1st plus the two lowest. Furthermore, on strict conflict trials where TTB and Tallying disagree, the Competing Theory uniformly predicts a 0.5 choice probability due to heuristic conflict. In contrast, the Advocated Theory predicts extreme probabilities on some conflict trials (when the exponentiated sum heavily favors one side) and 0.5 on others (when the highest cue perfectly balances the remaining cues), testing whether guessing is a blanket response to conflict or a natural result of value balancing.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
We compute the choice proportion for options that are favored by the Compensatory (WADD) strategy but are involved in strict heuristic conflict. Specifically, we look at trials comparing [1,1,0,0,0] vs [0,0,1,1,1] and [1,0,0,0,0] vs [0,1,1,1,1]. In these trials, Take The Best and Tallying point in opposite directions, causing the Competing Theory (Conflict-Induced Guessing) to predict exactly 50% choice probability. However, the Advocated Theory (WADD with exponentiated validities) consistently predicts a preference for [1,1,0,0,0] and [0,1,1,1,1], respectively, across all allowed parameter values. Thus, the metric will closely hover around 0.5 for the Competing Theory (with variance purely due to binomial noise) and be significantly > 0.5 for the Advocated Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    target_chosen = 0
    total = 0
    
    t3_target = (1, 1, 0, 0, 0)
    t3_other  = (0, 0, 1, 1, 1)
    
    t5_target = (0, 1, 1, 1, 1)
    t5_other  = (1, 0, 0, 0, 0)
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Check if trial is T3/4
        if (a == t3_target and b == t3_other) or (a == t3_other and b == t3_target):
            total += 1
            if (row['response'] == 0 and a == t3_target) or (row['response'] == 1 and b == t3_target):
                target_chosen += 1
                
        # Check if trial is T5/6
        elif (a == t5_target and b == t5_other) or (a == t5_other and b == t5_target):
            total += 1
            if (row['response'] == 0 and a == t5_target) or (row['response'] == 1 and b == t5_target):
                target_chosen += 1
                
    if total == 0:
        return 0.5
    return float(target_chosen) / total

### RESULTS
- Predicted under pi_4 (simulated): 0.6046 (var=0.0074)
- Predicted under pi_6 (simulated): 0.5012 (var=0.0061)
- Observed on real data: 0.4808 (var=0.0053)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.98, 0.94, 0.9, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design systematically dissociates the Advocated Theory (Conflict-Induced Guessing) from the Competing Theory (Compensatory WADD with exponentiated validities). By using three highly valid features and two low-validity features, we construct two crucial types of trials. First, 'heuristic agreement' trials where Option A possesses the single most valid cue AND the highest total number of cues (favored by both TTB and Tallying). The Advocated Theory predicts a strong preference for Option A. However, Option B possesses the second and third most valid cues, whose combined exponentiated weight easily surpasses Option A under the Competing Theory, leading it to strongly predict Option B. Second, 'strict conflict' trials where TTB favors one option and Tallying favors the other. The Advocated Theory posits that such conflict induces uncertainty and random guessing (P=0.5). In contrast, the Competing Theory naturally resolves these conflicts via its non-linear weighting, predicting strong deterministic preferences rather than 0.5.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times Option A is chosen on Trial 2 (where A=[0,1,1,1,1] and B=[1,0,0,0,0]). Under the Advocated theory, this is a strict conflict trial (TTB favors B, Tallying favors A), leading to random guessing and a choice proportion of ~0.5. Under the Competing theory, the combined exponentiated weights of A's four cues easily surpass B's single cue, leading to a strong deterministic preference for A (proportion ~1.0).

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 2 by option_a_ratings
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    if mask.sum() == 0:
        return 0.0
    
    # Return the proportion of times A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.5842 (var=0.0111)
- Predicted under pi_6 (simulated): 0.4947 (var=0.0142)
- Observed on real data: 0.4884 (var=0.0172)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for row in data.itertuples():
        a = row.option_a_ratings
        b = row.option_b_ratings
        
        # Determine Take The Best (TTB) prediction
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        # Determine Tallying prediction
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        tally_winner = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        # Only consider trials where TTB and Tallying make strict and opposing predictions
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if row.response == ttb_winner:
                ttb_matches += 1
            total += 1
            
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5075 (var=0.0051)
**Predicted under pi_4:** 0.4896 (var=0.0059)
**Predicted under pi_6:** 0.5012 (var=0.0057)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    tally_pred = (b_sums > a_sums).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.5079 (var=0.0012)
**Predicted under pi_4:** 0.4508 (var=0.0034)
**Predicted under pi_6:** 0.5029 (var=0.0028)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        a_sum = sum(a)
        b_sum = sum(b)
        tally_winner = 0 if a_sum > b_sum else (1 if b_sum > a_sum else None)
        
        if tally_winner is not None and ttb_winner != tally_winner:
            ttb_match.append(1 if resp == ttb_winner else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.5012 (var=0.0025)
**Predicted under pi_4:** 0.5044 (var=0.0053)
**Predicted under pi_6:** 0.4947 (var=0.0040)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    # TTB winner: first cue where options differ (since validities are strictly descending)
    diff = a_mat - b_mat
    nz = diff != 0
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    ttb_winner = np.where(first_diff > 0, 0, 1)
    
    # Tallying winner: option with more total feature wins
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    tally_winner = np.full(len(diff), -1)
    tally_winner[a_wins > b_wins] = 0
    tally_winner[b_wins > a_wins] = 1
    
    # Identify strict conflict trials where TTB and Tallying favor different options
    conflict = (ttb_winner != tally_winner) & (tally_winner != -1)
    
    if not np.any(conflict):
        return 0.5
        
    # Return the proportion of times the subject chose the TTB-favored option on conflict trials
    match = (resp[conflict] == ttb_winner[conflict])
    return float(np.mean(match))
```

**Observed (real) value:** 0.5175 (var=0.0039)
**Predicted under pi_4:** 0.4888 (var=0.0054)
**Predicted under pi_6:** 0.4984 (var=0.0040)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    choices_01100 = (data.loc[t1_mask, 'response'] == 1).sum() + (data.loc[t2_mask, 'response'] == 0).sum()
    total = t1_mask.sum() + t2_mask.sum()
    
    return float(choices_01100 / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5212 (var=0.0057)
**Predicted under pi_4:** 0.5500 (var=0.0066)
**Predicted under pi_6:** 0.4950 (var=0.0080)

### Experiment 8
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: target is [1, 0, 0, 1, 1] vs [0, 1, 1, 0, 0]
    t1_mask1 = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t1_mask2 = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    t1_chose_target = (t1_mask1 & (data['response'] == 0)).sum() + (t1_mask2 & (data['response'] == 1)).sum()
    t1_total = t1_mask1.sum() + t1_mask2.sum()
    p1 = t1_chose_target / t1_total if t1_total > 0 else 0.5
    
    # Trial 2: target is [0, 1, 1, 1, 1] vs [1, 0, 0, 0, 0]
    t2_mask1 = (a_tuples == (0, 1, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    t2_mask2 = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 1, 1, 1))
    
    t2_chose_target = (t2_mask1 & (data['response'] == 0)).sum() + (t2_mask2 & (data['response'] == 1)).sum()
    t2_total = t2_mask1.sum() + t2_mask2.sum()
    p2 = t2_chose_target / t2_total if t2_total > 0 else 0.5
    
    return float(p1 - p2)
```

**Observed (real) value:** 0.0400 (var=0.0147)
**Predicted under pi_4:** -0.1453 (var=0.0248)
**Predicted under pi_6:** 0.0474 (var=0.0345)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_sums = data['option_b_ratings'].apply(sum)
    mask = b_sums == 4
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4758 (var=0.0084)
**Predicted under pi_4:** 0.6053 (var=0.0084)
**Predicted under pi_6:** 0.5037 (var=0.0055)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    wadd_target = {
        (1, 1, 0, 0, 0, 0): 1,
        (0, 0, 1, 1, 1, 1): 0,
        (1, 0, 1, 0, 0, 0): 1,
        (1, 0, 1, 1, 1, 1): 0,
        (0, 1, 0, 1, 1, 1): 0,
        (0, 1, 0, 0, 0, 0): 1
    }
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    targets = a_tuples.map(wadd_target)
    matches = (data['response'] == targets)
    
    return float(matches.mean())
```

**Observed (real) value:** 0.5108 (var=0.0021)
**Predicted under pi_4:** 0.6498 (var=0.0064)
**Predicted under pi_6:** 0.5702 (var=0.0183)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across almost all 10 experiments, Theory 2 (pi_6: Conflict-Induced Guessing with Evidence Threshold) predicts the observed metrics much better than Theory 1 (pi_4: Compensatory WADD with exponentiated validities). The observed data consistently shows choice proportions hovering around 0.5 on trials with strict heuristic conflict or weak overall evidence differences. Theory 1 fails because it assumes decision-makers aggregate all cues into a continuous subjective value, leading to strong deterministic predictions (e.g., in Exp 1, 2, 8, 9, 10) that do not match the empirical guessing behavior. Theory 2 correctly identifies that conflict or low evidence leads to uncertainty and 50/50 guessing. However, Theory 1's core premise of compensatory aggregation is fundamentally at odds with the pervasive guessing observed in the data. Instead of tweaking the WADD model, we should replace Theory 1 entirely.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Probabilistic Heuristic Mixture' theory. Instead of assuming any form of Compensatory/WADD integration, propose that on each trial, decision-makers probabilistically sample a single simple heuristic (e.g., Take The Best, Tallying, or random guessing) from a fixed distribution. When heuristics agree, the choice is deterministic; when they conflict, the mixture naturally averages out to ~0.5 choice probabilities without requiring an explicit 'conflict detection' mechanism or computing WADD as a backup (which Theory 2 assumes). This provides a computationally simpler, purely non-compensatory alternative to Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 8932,
  "candidates_token_count": 362,
  "total_token_count": 10368
}
```
