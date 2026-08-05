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

## THEORY 2 — pi_5
Top-K Tallying with Restricted Scope: Decision-makers simplify complex multi-attribute choices by focusing only on the top K most valid cues, ignoring all lower-validity information. By restricting K to a narrow band (1, 2, or 3) and allowing for a high degree of random guessing (lapse rate), the model produces consistent behavior across individuals while naturally capturing the pervasive ~0.5 choice probabilities on strict conflict trials where the top cues often tie or balance out.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(parameters["k"])
    
    # Ensure k doesn't exceed the number of available features
    k = min(k, len(val))
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally positive features among top K
    a_score = np.sum(a[top_k_cues])
    b_score = np.sum(b[top_k_cues])
    
    if a_score > b_score:
        p_core = np.array([1.0, 0.0])
    elif b_score > a_score:
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
**Validities (n_features=6):** [0.9, 0.88, 0.86, 0.84, 0.82, 0.8]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Rationale:** This design strictly dissociates the Advocated Theory (WADD with exponentiated validities) from the Competing Theory (Top-K Tallying with restricted scope K in {1, 2, 3}). By using 6 features with relatively close validities, we can construct trials where the Top-K Tallying model consistently predicts Option A (or a tie) regardless of whether K is 1, 2, or 3, because Option A dominates in the top 2 features. However, Option B possesses a larger number of slightly lower-validity features (cues 3, 4, 5, 6). Because WADD evaluates all features and weights them by validities raised to a power, the combined weight of Option B's multiple features easily surpasses the weight of Option A's top features. This creates a stark quantitative dissociation where Top-K Tallying predicts a strong preference for A (or a 50% split), while WADD predicts a strong preference for B.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
Focuses on Trial 1 and Trial 2, where Option B has 4 positive features (cues 3, 4, 5, 6) and Option A has 2 positive features (cues 1, 2 or 1, 3). The Competing Theory (Top-K Tallying, K in 1,2,3) predicts Option A will be chosen almost exclusively because Option A dominates in the top validity cues. However, the Advocated Theory (WADD with exponentiated validities) predicts Option B will be chosen because the combined weighted value of four lower-validity cues surpasses the two higher-validity cues. The metric simply computes the proportion of times Option B is chosen on these critical trials.

Source:
def metric(data: pd.DataFrame) -> float:
    b_sums = data['option_b_ratings'].apply(sum)
    mask = b_sums == 4
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.6053 (var=0.0084)
- Predicted under pi_5 (simulated): 0.3032 (var=0.0245)
- Observed on real data: 0.4758 (var=0.0084)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.92, 0.9, 0.88, 0.86, 0.84, 0.82]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Rationale:** This design strictly dissociates the Advocated Theory (Top-K Tallying with restricted scope K in {1, 2, 3}) from the Competing Theory (WADD with exponentiated validities). By using 6 features with closely spaced high validities, we construct two primary types of dissociation trials. First, we create trials where Option A dominates in the top 2 features but Option B possesses the remaining 4 features. Because Top-K Tallying strictly ignores features beyond the top 3, it consistently predicts Option A across all allowable values of K. However, WADD integrates all features; due to the close validities, the combined exponentiated weight of Option B's 4 features surpasses Option A's 2 features even at high gamma values, causing WADD to strongly predict Option B. Second, we design 'top-tie' trials where both options are identical on the top 3 features, but one option dominates on the lower 3. Top-K Tallying is entirely blind to this lower-tier difference and predicts a strict 50/50 tie, whereas WADD predicts a strong deterministic preference for the option with the lower-tier features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the deterministic predictions of the WADD model. For Trials 1 and 2, WADD predicts the option with 4 lower-validity features, whereas Top-K Tallying deterministically predicts the option with 2 higher-validity features (0% alignment). For Trials 3 through 6, WADD deterministically favors the option that strictly dominates on lower-validity features, whereas Top-K Tallying is blind to these features and predicts a 50/50 tie (50% alignment). Consequently, WADD will score > 0.5 on this metric (approaching 1.0 for high beta/low epsilon), while Top-K Tallying will score around 0.333 (or up to 0.5 under pure random guessing). This establishes a robust and consistent theoretical gap.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.6498 (var=0.0064)
- Predicted under pi_5 (simulated): 0.4331 (var=0.0048)
- Observed on real data: 0.5108 (var=0.0021)

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
**Predicted under pi_5:** 0.5554 (var=0.0210)

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
**Predicted under pi_5:** 0.4131 (var=0.0099)

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
**Predicted under pi_5:** 0.5609 (var=0.0198)

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
**Predicted under pi_5:** 0.5409 (var=0.0195)

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
**Predicted under pi_5:** 0.4744 (var=0.0821)

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
**Predicted under pi_5:** -0.0242 (var=0.2223)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric values consistently hover around 0.5. In Experiment 1, the observed value is 0.4758 (compared to pi_4's 0.6053 and pi_5's 0.3032). In Experiment 2, the observed value is 0.5108 (compared to pi_4's 0.6498 and pi_5's 0.4331). Similar patterns emerge in Experiments 3-7 where the observed metrics are all between 0.50 and 0.52. Both pi_4 and pi_5 attempt to extract deterministic preferences from these strict conflict trials (either via exponentiated WADD or Top-K Tallying), but the real data suggests subjects are evenly split or guessing. Neither theory adequately captures this robust regression to 0.5.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that explicitly accounts for this robust 50/50 split on strict conflict trials. The new theory could posit a 'Mixture of Heuristics' where the population is evenly divided between subjects using strict Take-The-Best and subjects using unweighted Tallying, which naturally balances out to ~0.5 on conflict trials. Alternatively, the theory could propose 'Conflict-Induced Guessing', where subjects compute a quick conflict signal (e.g., top-cue favors A, but overall count favors B) and resort to random guessing when the conflict is too high. This will better capture the pervasive 0.5 choice probabilities observed across all these experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 7565,
  "candidates_token_count": 375,
  "total_token_count": 9060
}
```
