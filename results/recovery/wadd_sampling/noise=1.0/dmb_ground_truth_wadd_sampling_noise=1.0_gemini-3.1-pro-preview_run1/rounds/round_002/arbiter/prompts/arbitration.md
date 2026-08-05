# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_3
Strategy Mixture: Instead of adopting a single heuristic for the entire experiment, individuals probabilistically sample a decision strategy on each trial. Specifically, they mix between Take The Best (TTB) and Tallying, leading to an equal probability of choosing either option when the two heuristics strictly conflict.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Take The Best (TTB) prediction
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner is None:
        p_ttb_choice = np.array([0.5, 0.5])
    else:
        p_ttb_choice = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Tallying prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    # Strategy mixture
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_mixed = p_ttb * p_ttb_choice + (1.0 - p_ttb) * p_tally_choice
    
    # Apply epsilon lapse rate
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.92, 0.89, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To cleanly dissociate the Advocated Theory (WADD with subjective validity transformation) from the Competing Theory (a strategy mixture of TTB and Tallying), we design trials where TTB and Tallying perfectly agree, yet WADD predicts the exact opposite choice. By using a feature set with three highly valid cues and two low validity cues, we can construct options where Product A has the single most valid cue AND a greater total number of positive cues (strongly favored by both TTB and Tallying), while Product B has the second and third most valid cues. Because WADD exponentiates validities, the combined weight of the second and third cues surpasses the first cue even at high exponent values, causing WADD to strongly prefer Product B. This creates a stark qualitative and quantitative divergence: the Mixture model predicts near-certainty for A, while WADD predicts a strong preference for B.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option [0, 1, 1, 0, 0] over [1, 0, 0, 1, 1]. In these specific trials, the Mixture model (TTB + Tallying) strongly prefers [1, 0, 0, 1, 1] because it has the single most valid cue AND a greater total number of positive cues. However, the Advocated WADD model strongly prefers [0, 1, 1, 0, 0] because the exponentiated sum of the second and third most valid cues outweighs the first cue. This creates a maximal divergence between the two theories: the Mixture model predicts a value near 0, while WADD predicts a value near 1.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t2_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    choices_01100 = (data.loc[t1_mask, 'response'] == 1).sum() + (data.loc[t2_mask, 'response'] == 0).sum()
    total = t1_mask.sum() + t2_mask.sum()
    
    return float(choices_01100 / total) if total > 0 else 0.5

### RESULTS
- Predicted under pi_4 (simulated): 0.5500 (var=0.0066)
- Predicted under pi_3 (simulated): 0.0488 (var=0.0021)
- Observed on real data: 0.5212 (var=0.0057)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.98, 0.94, 0.9, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design systematically dissociates the Strategy Mixture theory (which mixes pure Take The Best and pure Tallying) from the Compensatory WADD theory (which weights features by exponentiated validities). By using three highly valid features and two less valid features, we can construct trials where TTB and Tallying completely agree (e.g., Option A has the highest validity cue AND the most cues overall), leading the Mixture model to predict a strong preference for Option A. However, Option B possesses the second and third most valid cues, whose combined exponentiated weight under WADD exceeds the highest cue plus the two lowest cues, leading WADD to strongly predict Option B. In other trials, TTB and Tallying conflict (Mixture model predicts ~50% choice), but WADD strongly favors one option due to the non-linear weighting of validities. This provides multiple distinct quantitative and qualitative divergence points.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in preference for the option with more features on two specific trial types. On Trial 1 ([1,0,0,1,1] vs [0,1,1,0,0]), the Strategy Mixture model strongly prefers the first option (~90% probability) since both Take The Best (TTB) and Tallying agree. However, WADD slightly prefers the second option due to the compensatory weights of the second and third validities. On Trial 2 ([0,1,1,1,1] vs [1,0,0,0,0]), TTB and Tallying conflict, resulting in a ~50% choice rate for the first option under the Mixture model, whereas WADD consistently prefers the first option because the summed exponentiated weights outcompete the top cue. The difference P(Target 1) - P(Target 2) is expected to be highly positive for the Mixture model and strongly negative for the WADD model, maximizing discriminability while remaining stable within subjects.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): -0.1453 (var=0.0248)
- Predicted under pi_3 (simulated): 0.4495 (var=0.0153)
- Observed on real data: 0.0400 (var=0.0147)

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
**Predicted under pi_3:** 0.5162 (var=0.0080)

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
**Predicted under pi_3:** 0.4919 (var=0.0060)

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
**Predicted under pi_3:** 0.5178 (var=0.0082)

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
**Predicted under pi_3:** 0.5044 (var=0.0045)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics consistently hover around 0.5. This indicates that subjects are largely indifferent or choosing at random on these carefully constructed conflict trials. THEORY 1 (pi_4) explicitly anticipates this by positing a subjective transformation of validities (raising to a power) that naturally balances the scales on conflict trials, pulling predictions towards 0.5. THEORY 1 predicts 0.55 in Exp 1 and -0.14 in Exp 2, which is much closer to the observed 0.52 and 0.04 than THEORY 2. THEORY 2 (pi_3), which mixes pure TTB and Tallying, predicts extreme values (0.04 in Exp 1, 0.45 in Exp 2) because both heuristics often agree strongly on these trials, failing to capture the observed indifference. Therefore, THEORY 2 is degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that can naturally explain the pervasive ~0.5 choice probabilities on conflict trials without relying on an exponentiated validity mechanism. For example, you could propose a 'Noisy Linear WADD' theory where subjects simply sum the raw validities but apply a very low inverse temperature (high decision noise) in their softmax choice, reflecting task difficulty or low effort. Alternatively, propose a 'Top-K Tallying' heuristic where subjects only count the number of positive features among the top 2 or 3 most valid cues, ignoring low-validity cues entirely; when the top cues are tied, they guess."
}
```

## Usage

```json
{
  "prompt_token_count": 6740,
  "candidates_token_count": 361,
  "total_token_count": 8843
}
```
