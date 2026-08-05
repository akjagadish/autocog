# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Conflict-Induced Guessing Theory posits that decision makers concurrently evaluate options using both a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). When these two strategies agree, the decision maker confidently chooses the favored option (subject to standard softmax noise). However, when the strategies conflict—or when one strategy fails to corroborate the other (e.g., Tallying is tied while TTB prefers one option)—the decision maker experiences cognitive conflict or ambiguity. Unable to easily resolve this conflict, they resort to uniform guessing. This explains the consistent ~50% choice rates observed on adversarial trials and trials with weak corroborating evidence without requiring extreme parameter values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
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
        elif b[j] > a[j]:
            ttb_winner = 1
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        tally_winner = 0
    elif b_wins > a_wins:
        tally_winner = 1
    else:
        tally_winner = None
        
    # Determine choice probabilities
    if ttb_winner != tally_winner or ttb_winner is None:
        # Conflict or lack of clear corroboration leads to guessing
        p_core = np.array([0.5, 0.5])
    else:
        # No conflict: both strategies agree
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
            
        beta = float(parameters["beta"])
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Threshold Tallying Theory posits that decision makers evaluate options by counting the number of features where one option strictly dominates the other (unweighted tallying). However, they only make a confident directional choice if the difference in winning features between the two options meets or exceeds a certain cognitive threshold. If the difference in evidence is below this threshold (e.g., a difference of 0 or 1), the decision maker finds the evidence too ambiguous or weak, leading to cognitive overload or uncertainty, and they resort to uniform guessing. This captures the persistent ~50% choice rates observed across many adversarial trials where the feature counts are closely matched.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Count strict feature-wise wins for each option
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    diff = abs(a_wins - b_wins)
    threshold = float(parameters["threshold"])
    
    # If the difference is below the threshold, the evidence is deemed too weak -> guessing
    if diff < threshold:
        p_core = np.array([0.5, 0.5])
    else:
        # Otherwise, make a choice based on the tally scores using softmax
        scores = np.array([float(a_wins), float(b_wins)])
        beta = float(parameters["beta"])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Conflict-Induced Guessing Theory from Threshold Tallying Theory, we use a 6-feature design. Conflict-Induced Guessing Theory predicts uniform guessing whenever Take The Best (TTB) and Tallying conflict, and confident choices when they agree. Threshold Tallying Theory predicts uniform guessing when the absolute difference in feature wins (Diff) is below a cognitive threshold (1.5 to 3.5), and confident choices when Diff is above this threshold. We construct two types of critical trials. In Type 1, one option wins on the single highest-validity cue (TTB winner) but loses on all remaining cues (Tallying loser). Here, TTB and Tallying conflict, so Conflict Theory predicts guessing; however, the Diff is 4 (5 wins vs 1 win), which exceeds the maximum threshold of 3.5, so Threshold Tallying predicts a highly confident choice. In Type 2, one option wins on the highest-validity cue and one other cue, while the alternative wins on only a single lower-validity cue. Here, TTB and Tallying agree, so Conflict Theory predicts a confident choice; however, the Diff is 1 (2 wins vs 1 win), which is strictly below the minimum threshold of 1.5, so Threshold Tallying predicts uniform guessing. These opposing predictions cleanly separate the two theories across their entire parameter spaces.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in choice consistency with the Tallying winner between trials where the absolute difference in feature wins is 1 versus 4. Under Conflict-Induced Guessing Theory, a difference of 4 corresponds to adversarial trials where TTB and Tallying conflict, leading to guessing (~50%), while a difference of 1 corresponds to trials where both strategies agree, leading to confident choices (~100%). Thus, the metric should be strongly positive (approx. 0.5). Under Threshold Tallying Theory, a difference of 4 exceeds the threshold, leading to confident choices (~100%), whereas a difference of 1 is below the threshold, leading to guessing (~50%). Thus, the metric should be strongly negative (approx. -0.5). This creates a massive, cleanly separated contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_diff_1 = []
    match_diff_4 = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        diff = abs(a_wins - b_wins)
        
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            continue
            
        match = 1 if row['response'] == tally_winner else 0
        
        if diff == 1:
            match_diff_1.append(match)
        elif diff == 4:
            match_diff_4.append(match)
            
    p1 = np.mean(match_diff_1) if match_diff_1 else 0.0
    p4 = np.mean(match_diff_4) if match_diff_4 else 0.0
    
    return float(p1 - p4)

### RESULTS
- Predicted under pi_4 (simulated): 0.3600 (var=0.0152)
- Predicted under pi_5 (simulated): -0.3534 (var=0.0167)
- Observed on real data: -0.0319 (var=0.0091)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=7):** [0.98, 0.88, 0.78, 0.68, 0.58, 0.54, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate Threshold Tallying Theory from Conflict-Induced Guessing Theory, we use a 7-feature design. Conflict-Induced Guessing Theory predicts guessing whenever Take The Best (TTB) and Tallying conflict, and confident choices when they agree. Threshold Tallying Theory ignores validities entirely, predicting uniform guessing when the absolute difference in feature wins (Diff) is below a cognitive threshold (1.5 to 3.5), and confident choices when Diff is above this threshold. We construct critical trials to perfectly cross these predictions. In 'TT-confident / CIG-guess' trials, one option wins on the single highest-validity cue (TTB winner) but loses on all remaining cues (Tallying loser). Here, TTB and Tallying conflict (CIG guesses), but the Diff is 6 (6 wins vs 1 win), which exceeds the maximum threshold of 3.5 (TT is highly confident). In 'TT-guess / CIG-confident' trials, one option wins on the highest-validity cue and a few others, while the alternative wins on just one fewer lower-validity cues. Here, TTB and Tallying agree (CIG is confident), but the Diff is exactly 1, which is strictly below the minimum threshold of 1.5 (TT guesses).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts predictions on trials where the two theories have diametrically opposed confidence levels. Trial 1 is a 'TT-confident / CIG-guess' trial where Threshold Tallying strongly predicts choosing B (due to a tally difference of 5), while Conflict-Induced Guessing predicts ~50% because the best cue favors A but tallying favors B. Trials 2 and 4 are 'TT-guess / CIG-confident' trials where TT predicts ~50% (tally difference of 1 is below threshold) while CIG strongly predicts A (both TTB and tallying favor A). By computing P(A | Trial 2) + P(A | Trial 4) - P(B | Trial 1), we expect a value near 0 for TT (0.5 + 0.5 - 1.0) and near 1.5 for CIG (1.0 + 1.0 - 0.5), yielding a robust and highly discriminative metric.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 1) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 6)
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 2) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 1)
    t4_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and sum(x) == 3) & data['option_b_ratings'].apply(lambda x: x[0] == 0 and sum(x) == 2)
    
    p_b_t1 = np.mean(data[t1_mask]['response'] == 1) if t1_mask.sum() > 0 else 0.5
    p_a_t2 = np.mean(data[t2_mask]['response'] == 0) if t2_mask.sum() > 0 else 0.5
    p_a_t4 = np.mean(data[t4_mask]['response'] == 0) if t4_mask.sum() > 0 else 0.5
    
    return float(p_a_t2 + p_a_t4 - p_b_t1)

### RESULTS
- Predicted under pi_4 (simulated): 1.1850 (var=0.0461)
- Predicted under pi_5 (simulated): 0.1562 (var=0.0557)
- Observed on real data: 0.4675 (var=0.0532)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                winner = 0
                break
            elif b[i] > a[i]:
                winner = 1
                break
        if winner is not None:
            matches.append(1 if resp == winner else 0)
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0012)
**Predicted under pi_4:** 0.5360 (var=0.0025)
**Predicted under pi_5:** 0.4294 (var=0.0045)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4981 (var=0.0023)
**Predicted under pi_4:** 0.4918 (var=0.0024)
**Predicted under pi_5:** 0.5400 (var=0.0056)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_aligned = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(4):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            if resp == ttb_pred:
                ttb_aligned += 1
            total += 1
            
    return ttb_aligned / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5033 (var=0.0016)
**Predicted under pi_4:** 0.4969 (var=0.0038)
**Predicted under pi_5:** 0.4740 (var=0.0053)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5054 (var=0.0018)
**Predicted under pi_4:** 0.5098 (var=0.0029)
**Predicted under pi_5:** 0.5369 (var=0.0040)

### Experiment 7
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (a_tuples == (1, 0, 0, 1, 1)) & (b_tuples == (0, 1, 1, 0, 0))
    t5_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1))
    
    resp_t1 = data.loc[t1_mask, 'response'].mean()
    resp_t5 = data.loc[t5_mask, 'response'].mean()
    
    val_t1 = float(resp_t1) if not pd.isna(resp_t1) else 0.5
    val_t5 = float(resp_t5) if not pd.isna(resp_t5) else 0.5
    
    return val_t5 - val_t1
```

**Observed (real) value:** -0.0133 (var=0.0243)
**Predicted under pi_4:** 0.7133 (var=0.0273)
**Predicted under pi_5:** 0.0200 (var=0.0429)

### Experiment 8
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    mask_t1 = (a_tuples == (0, 1, 1, 1, 0))
    mask_t2 = (a_tuples == (1, 0, 0, 0, 0))
    
    data_t1 = data[mask_t1]
    data_t2 = data[mask_t2]
    
    val1 = (data_t1['response'] == 0).mean() if len(data_t1) > 0 else 0.5
    val2 = (data_t2['response'] == 1).mean() if len(data_t2) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)
```

**Observed (real) value:** 0.4916 (var=0.0085)
**Predicted under pi_4:** 0.5147 (var=0.0057)
**Predicted under pi_5:** 0.4963 (var=0.0083)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics consistently hover around the guessing rate (~0.5 or 0 difference), even on trials where both Conflict-Induced Guessing (pi_4) and Threshold Tallying (pi_5) predict highly confident choices. For instance, in Experiment 1, subjects faced a choice between an option possessing only the single most valid cue and an option possessing all other lower-validity cues. Both theories failed to predict the observed ~50% choice rate here. This pervasive ~50% choice on such '1 vs. all others' adversarial trials strongly suggests that subjects are not using raw validities or simple tallying. Instead, they appear to be using exponentially decaying, rank-based weights (e.g., 1/2, 1/4, 1/8...). Under a 1/2^k weighting scheme, the weight of the highest-ranked cue (1/2) is mathematically almost identical to the sum of all remaining cues (1/4 + 1/8 + ... = 1/2). Thus, pitting the best cue against all others perfectly balances the evidence, naturally predicting the ~50% choice rates observed across these diverse adversarial designs without requiring explicit conflict-detection mechanisms.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Rank-Weighted Additive Theory'. This theory posits that decision-makers ignore the exact numerical validities provided and instead assign exponentially decaying weights based solely on the ordinal rank of the cues (e.g., w_k = 1 / 2^k). They then compute a weighted sum for each option and choose probabilistically (via softmax). Because the highest-ranked cue's weight equals the sum of all lower-ranked cues' weights, this theory naturally predicts the ~50% guessing rate observed on '1 vs all' adversarial trials without needing arbitrary thresholds or conflict-based guessing rules."
}
```

## Usage

```json
{
  "prompt_token_count": 7564,
  "candidates_token_count": 422,
  "total_token_count": 9887
}
```
