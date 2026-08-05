# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_14" and "pi_15") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_14" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_15" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_14
Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_15
Adaptive Environment-Structured Strategy Selection: Decision-makers adapt their choice strategy based on the global statistical structure of the environment. Specifically, they evaluate the dispersion (e.g., standard deviation) of cue validities. In environments with high dispersion, they probabilistically favor a non-compensatory heuristic (Take-The-Best). In environments with low dispersion, they fall back to a simpler compensatory heuristic (Equal-weights Tallying). The mixture between these strategies is dynamically governed by a logistic function of the environment's validity dispersion, with independent confidence (temperature) parameters for each strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate dispersion of validities in the environment
    dispersion = float(np.std(val))
    
    # Logistic function for strategy weight
    # Higher dispersion -> higher probability of using Take-The-Best
    w_ttb = 1.0 / (1.0 + np.exp(-k * (dispersion - theta)))
    
    # Take-The-Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_diff = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            ttb_diff = 1.0
            break
        elif b[j] > a[j]:
            ttb_diff = -1.0
            break
            
    scores_ttb = np.array([ttb_diff, -ttb_diff])
    
    # Tallying component
    # Normalized by the number of cues to ensure it's on a comparable scale to TTB (-1 to 1)
    tally_diff = (np.sum(a) - np.sum(b)) / len(val)
    scores_tally = np.array([tally_diff, -tally_diff])
    
    # Convert scores to probabilities
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    p_ttb = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # Mixture prediction
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_14)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated 'Sequential Evidence Accumulation' (SCA) theory from the competing 'Adaptive Strategy Selection' (TTB + Tallying mixture) theory by creating scenarios where both components of the competing theory agree, but the advocated theory predicts the exact opposite choice. The competing theory is a mixture of Take-The-Best (which relies solely on the first discriminating cue) and Tallying (which relies solely on the unweighted sum of cues). By setting validities such that the top cue favors Option A, but the second and third cues favor Option B, and the fourth and fifth cues favor Option A, we create a situation where Option A wins on TTB (Cue 1) AND wins on Tallying (3 cues to 2). Thus, the competing theory strictly predicts a preference for Option A regardless of its mixture parameter. However, SCA uses a chance-corrected non-linear weighting (e.g., max(val - 0.5)^gamma). Because the second and third cues are very close in validity to the first, their combined compensatory weight can easily exceed the combined weight of the first, fourth, and fifth cues (which include near-chance cues that are heavily discounted). Therefore, if the threshold is not immediately crossed, SCA uniquely predicts a preference for Option B, a mathematical impossibility for the competing theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on the critical trials (Trial 1 and its mirror, Trial 2) where both components of the competing theory (Take-The-Best and Tallying) agree on the same option, but the advocated Sequential Evidence Accumulation theory predicts the opposite option due to its non-linear weighting of validities. The metric calculates the proportion of choices that align with the unique SCA prediction on these trials.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x))
    
    t1_mask = a_tuples == (1, 0, 0, 1, 1)
    t2_mask = a_tuples == (0, 1, 1, 0, 0)
    
    sca_choices = (data.loc[t1_mask, 'response'] == 1).sum() + (data.loc[t2_mask, 'response'] == 0).sum()
    total = t1_mask.sum() + t2_mask.sum()
    
    if total == 0:
        return 0.5
    return float(sca_choices / total)

### RESULTS
- Predicted under pi_14 (simulated): 0.5469 (var=0.0314)
- Predicted under pi_15 (simulated): 0.1881 (var=0.0158)
- Observed on real data: 0.1575 (var=0.0127)

## EXPERIMENT 2 (proposed by pi_15)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the advocated 'Adaptive Environment-Structured Strategy Selection' (Strategy Mixture) model from the competing 'Sequential Evidence Accumulation' (SCA) theory by exploiting their divergent handling of secondary cues. The advocated theory is a mixture of Take-The-Best (which relies exclusively on the first discriminating cue) and Tallying (which relies exclusively on the unweighted sum of cues). Across a subset of trials, we hold two factors strictly constant: the highest discriminating cue (which fixes the TTB choice) and the total number of cues favoring each option (which fixes the Tallying difference). Because neither component of the advocated theory is sensitive to the specific validities of the secondary cues, it predicts identically flat choice probabilities across all these trials. SCA, in contrast, accumulates evidence based on the non-linearly scaled validities of all evaluated cues. By systematically shifting the secondary cues supporting the top-cue winner from high validity to low validity, SCA predicts a systematic, graded change in choice probabilities, directly contradicting the flat-line prediction of the advocated theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the two trial pairs that produce the maximal divergence in evidence according to the Sequential Evidence Accumulation (SCA) theory, while maintaining strict equivalence under the advocated Strategy Mixture theory. In Trials 1 and 5, the Cue 0 winner is supported by Cue 1 (the strongest secondary cue). In Trials 4 and 8, the Cue 0 winner is supported by Cue 4 (the weakest secondary cue). Because both the highest discriminating cue (Cue 0) and the total tally of supporting cues (2 vs 3) are perfectly matched across these subsets, the advocated mixture of Take-The-Best and Tallying predicts a difference of exactly zero. In contrast, SCA accumulates evidence proportionally to non-linearly scaled validities, thus predicting a highly positive difference as the supporting secondary evidence drops from Cue 1 to Cue 4. By focusing exclusively on these two extremes rather than diluting the signal with intermediate trials, the metric maximizes the mean difference between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify which option is favored by the most valid cue (Cue 0)
    a_winner = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Determine if the subject chose the option favored by Cue 0
    chose_winner = ((a_winner) & (data['response'] == 0)) | ((~a_winner) & (data['response'] == 1))
    
    # Identify trials where the Cue 0 winner is supported by the second most valid cue (Cue 1)
    # In the experimental design, these correspond strictly to Trials 1 and 5.
    winner_has_cue1 = data.apply(
        lambda row: row['option_a_ratings'][1] == 1 if row['option_a_ratings'][0] == 1 else row['option_b_ratings'][1] == 1,
        axis=1
    )
    
    # Identify trials where the Cue 0 winner is supported by the least valid cue (Cue 4)
    # In the experimental design, these correspond strictly to Trials 4 and 8.
    winner_has_cue4 = data.apply(
        lambda row: row['option_a_ratings'][4] == 1 if row['option_a_ratings'][0] == 1 else row['option_b_ratings'][4] == 1,
        axis=1
    )
    
    # Calculate the choice probability for the Cue 0 winner in these extreme subsets
    p_winner_cue1 = chose_winner[winner_has_cue1].mean()
    p_winner_cue4 = chose_winner[winner_has_cue4].mean()
    
    # Return the maximal contrast
    return float(p_winner_cue1 - p_winner_cue4)

### RESULTS
- Predicted under pi_14 (simulated): 0.1675 (var=0.0910)
- Predicted under pi_15 (simulated): -0.0233 (var=0.0104)
- Observed on real data: 0.0150 (var=0.0125)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_matches = 0
    total = 0
    
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
            if resp == ttb_winner:
                ttb_matches += 1
            total += 1
            
    return float(ttb_matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.7417 (var=0.0032)
**Predicted under pi_14:** 0.5127 (var=0.0109)
**Predicted under pi_15:** 0.5448 (var=0.0623)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

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
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        matches.append(row['response'] == pred)
        
    return float(np.mean(matches)) if len(matches) > 0 else 0.5
```

**Observed (real) value:** 0.2881 (var=0.0100)
**Predicted under pi_14:** 0.4814 (var=0.0153)
**Predicted under pi_15:** 0.4821 (var=0.0882)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_winner(a, b):
        for i in range(len(a)):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return -1

    extreme_comp = []
    extreme_noncomp = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_win = get_ttb_winner(a, b)
        if ttb_win == -1:
            continue
            
        ttb_chosen = 1 if resp == ttb_win else 0
        
        # Tallying margin for the TTB winner
        margin = sum(a) - sum(b) if ttb_win == 0 else sum(b) - sum(a)
            
        # Extreme compensatory trials (TTB winner is opposed by many cues)
        if margin <= -2:
            extreme_comp.append(ttb_chosen)
        # Non-compensatory trials (TTB winner is supported by overall cue sum)
        elif margin >= 1:
            extreme_noncomp.append(ttb_chosen)
            
    if not extreme_comp or not extreme_noncomp:
        return 0.0
        
    return float(np.mean(extreme_noncomp) - np.mean(extreme_comp))
```

**Observed (real) value:** 0.1389 (var=0.0317)
**Predicted under pi_14:** 0.1328 (var=0.0973)
**Predicted under pi_15:** 0.3350 (var=0.0833)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_p_a(a_list, b_list):
        a_tup = tuple(a_list)
        b_tup = tuple(b_list)
        mask = (data['option_a_ratings'].apply(tuple) == a_tup) & (data['option_b_ratings'].apply(tuple) == b_tup)
        if mask.sum() == 0:
            return 0.5
        return 1.0 - data.loc[mask, 'response'].mean()

    # Trial 1: highest validity cue favors A, lowest validity cue favors B
    p_a_t1 = get_p_a([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    
    # Trial 4: 4th validity cue favors A, lowest validity cue favors B
    p_a_t4 = get_p_a([0, 0, 0, 1, 0], [0, 0, 0, 0, 1])
    
    return float(p_a_t1 - p_a_t4)
```

**Observed (real) value:** 0.0000 (var=0.0233)
**Predicted under pi_14:** 0.1583 (var=0.0462)
**Predicted under pi_15:** 0.0150 (var=0.0277)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_first_diff(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i, (val_a, val_b) in enumerate(zip(a, b)):
            if val_a != val_b:
                return i
        return -1

    # Calculate which cue is the highest valid discriminating cue
    first_diffs = data.apply(get_first_diff, axis=1)
    
    # Probability of choosing Option A
    chose_a = 1 - data['response']
    
    # Mean P(Choose A) when Cue 1 (index 0) is the highest discriminating cue
    mask_cue1 = (first_diffs == 0)
    p_a_cue1 = chose_a[mask_cue1].mean() if mask_cue1.sum() > 0 else 0.0
    
    # Mean P(Choose A) when Cue 3 (index 2) is the highest discriminating cue
    mask_cue3 = (first_diffs == 2)
    p_a_cue3 = chose_a[mask_cue3].mean() if mask_cue3.sum() > 0 else 0.0
    
    # The metric is the difference in P(Choose A) between Cue 1 and Cue 3 trials.
    return float(p_a_cue1 - p_a_cue3)
```

**Observed (real) value:** -0.4333 (var=0.0244)
**Predicted under pi_14:** -0.0207 (var=0.0253)
**Predicted under pi_15:** 0.0167 (var=0.0111)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify if option A has the highest validity cue (feature 0)
    # Feature 0 has validity 0.95, so Take The Best (TTB) always favors the option with a 1 here.
    a_has_f0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    
    # Identify if the subject chose A or B
    chose_a = (data['response'] == 0)
    chose_b = (data['response'] == 1)
    
    # Check if the chosen option is the one with feature 0 (the TTB winner)
    ttb_chosen = (a_has_f0 & chose_a) | (~a_has_f0 & chose_b)
    
    # Return the overall proportion of choosing the TTB winner
    return float(ttb_chosen.mean())
```

**Observed (real) value:** 0.4946 (var=0.0018)
**Predicted under pi_14:** 0.6085 (var=0.0142)
**Predicted under pi_15:** 0.6833 (var=0.0323)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    target_trials = [
        ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
        ((1, 0, 0, 1, 0), (0, 1, 1, 0, 0)),
        ((1, 0, 1, 0, 0), (0, 1, 0, 1, 0)),
        ((1, 0, 0, 0, 1), (0, 1, 0, 1, 0)),
        ((0, 1, 1, 0, 0), (1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))
    ]
    
    mad_sum = 0.0
    count = 0
    for a_tup, b_tup in target_trials:
        mask = (data['a_tup'] == a_tup) & (data['b_tup'] == b_tup)
        if mask.sum() > 0:
            p_a = 1.0 - data[mask]['response'].mean()
            mad_sum += abs(p_a - 0.5)
            count += 1
            
    return float(mad_sum / count) if count > 0 else 0.0
```

**Observed (real) value:** 0.3633 (var=0.0090)
**Predicted under pi_14:** 0.0961 (var=0.0088)
**Predicted under pi_15:** 0.1367 (var=0.0091)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Convert ratings to hashable tuples
    t_keys = data['option_a_ratings'].apply(tuple)
    
    # Focus on the highly conflicting trials where SCA is likely to fail to reach its evidence threshold
    t6 = (1, 0, 1, 1, 0, 0)
    t7 = (1, 0, 1, 0, 0, 0)
    t8 = (1, 0, 0, 0, 0, 0)
    
    mask = t_keys.isin([t6, t7, t8])
    if not mask.any():
        return 0.0
        
    # Calculate the mean response for each of these trial types
    subset = data[mask]
    t_keys_subset = t_keys[mask]
    means = subset.groupby(t_keys_subset)['response'].mean()
    
    # Calculate 'determinism': how far the mean response is from 0.5 (random guessing)
    determinism = (means - 0.5).abs().mean()
    
    return float(determinism)
```

**Observed (real) value:** 0.3389 (var=0.0140)
**Predicted under pi_14:** 0.0578 (var=0.0108)
**Predicted under pi_15:** 0.0800 (var=0.0130)

### Experiment 11
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Calculate tallies (sum of positive cues) for each option
    a_tallies = data['option_a_ratings'].apply(lambda x: sum(x))
    b_tallies = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter for trials where the tallies are unequal (trials 1, 2, 6, 7)
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    filtered_data = data[unequal_mask]
    a_tally_filtered = a_tallies[unequal_mask]
    b_tally_filtered = b_tallies[unequal_mask]
    
    # Determine which option has the higher tally (0 for A, 1 for B)
    higher_tally_option = (b_tally_filtered > a_tally_filtered).astype(int)
    
    # Calculate the proportion of choices that match the higher tally option
    match = (filtered_data['response'] == higher_tally_option).mean()
    
    return float(match)
```

**Observed (real) value:** 0.8892 (var=0.0052)
**Predicted under pi_14:** 0.4408 (var=0.0114)
**Predicted under pi_15:** 0.5408 (var=0.0797)

### Experiment 12
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Trial 4: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0]
    mask = (
        ((a_str == '00111') & (b_str == '11000')) |
        ((a_str == '10001') & (b_str == '01100'))
    )
    
    if mask.sum() == 0:
        return 0.0
    
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.4900 (var=0.0032)
**Predicted under pi_14:** 0.5500 (var=0.0172)
**Predicted under pi_15:** 0.3808 (var=0.0116)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_key(row):
        a = tuple(int(x) for x in row['option_a_ratings'])
        b = tuple(int(x) for x in row['option_b_ratings'])
        return str(a) + str(b)
    
    data['trial_str'] = data.apply(get_key, axis=1)
    
    t5 = str((1, 0, 1, 0, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t6 = str((0, 0, 1, 1, 1, 0)) + str((0, 0, 0, 0, 0, 1))
    
    t7 = str((1, 0, 1, 1, 1, 0)) + str((0, 1, 0, 0, 0, 0))
    t8 = str((0, 0, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    t9 = str((1, 0, 1, 1, 1, 1)) + str((0, 1, 0, 0, 0, 0))
    t10 = str((0, 1, 1, 1, 1, 1)) + str((1, 0, 0, 0, 0, 0))
    
    data['chose_A'] = (data['response'] == 0).astype(float)
    subj_trial = data.groupby(['subject_id', 'trial_str'])['chose_A'].mean().unstack()
    
    diff2 = (subj_trial[t5] - subj_trial[t6]).abs().mean() if t5 in subj_trial and t6 in subj_trial else 0
    diff3 = (subj_trial[t7] - subj_trial[t8]).abs().mean() if t7 in subj_trial and t8 in subj_trial else 0
    diff4 = (subj_trial[t9] - subj_trial[t10]).abs().mean() if t9 in subj_trial and t10 in subj_trial else 0
    
    return float(diff2 + diff3 + diff4)
```

**Observed (real) value:** 0.3556 (var=0.0405)
**Predicted under pi_14:** 0.5556 (var=0.0879)
**Predicted under pi_15:** 0.9111 (var=0.2983)

### Experiment 14
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract whether Option B has the most valid feature (index 0)
    b_has_best_cue = data['option_b_ratings'].apply(lambda x: x[0])
    
    # response == 0 means Option A was chosen
    chose_a = 1.0 - data['response']
    
    # Calculate the choice probability for A when B does NOT have the best cue
    p_a_when_b_lacks_best = chose_a[b_has_best_cue == 0].mean()
    
    # Calculate the choice probability for A when B HAS the best cue
    p_a_when_b_has_best = chose_a[b_has_best_cue == 1].mean()
    
    # The difference reflects sensitivity to cue validities
    return float(p_a_when_b_lacks_best - p_a_when_b_has_best)
```

**Observed (real) value:** 0.0067 (var=0.0065)
**Predicted under pi_14:** 0.1500 (var=0.0465)
**Predicted under pi_15:** 0.3262 (var=0.0827)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where Option A has lower tally (2 vs 3) but higher validity under both TTB and WADD
    is_t1 = (data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    if not is_t1.any():
        return 0.0
    # response == 0 means Option A was chosen
    return float((data.loc[is_t1, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1240 (var=0.0162)
**Predicted under pi_14:** 0.6540 (var=0.0501)
**Predicted under pi_15:** 0.5720 (var=0.0688)

### Experiment 16
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    # Focus purely on trials where the tally difference is exactly 1
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.5
        
    df = data[mask]
    tally_a_filtered = tally_a[mask]
    tally_b_filtered = tally_b[mask]
    
    # Check if the subject chose the option with the higher tally
    chose_a = df['response'] == 0
    higher_tally_a = tally_a_filtered > tally_b_filtered
    
    chose_higher = (chose_a == higher_tally_a)
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.2000 (var=0.0131)
**Predicted under pi_14:** 0.4808 (var=0.0306)
**Predicted under pi_15:** 0.5417 (var=0.1062)

### Experiment 17
**Design**
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    mask = (tally_a == 2) & (tally_b == 3)
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.8633 (var=0.0110)
**Predicted under pi_14:** 0.3900 (var=0.0568)
**Predicted under pi_15:** 0.5183 (var=0.0742)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tallies = data['option_a_ratings'].apply(sum)
    b_tallies = data['option_b_ratings'].apply(sum)
    
    unequal_mask = a_tallies != b_tallies
    if not unequal_mask.any():
        return 0.5
        
    unequal_data = data[unequal_mask]
    a_higher = a_tallies[unequal_mask] > b_tallies[unequal_mask]
    b_higher = b_tallies[unequal_mask] > a_tallies[unequal_mask]
    
    chose_a = unequal_data['response'] == 0
    chose_b = unequal_data['response'] == 1
    
    matched = (a_higher & chose_a) | (b_higher & chose_b)
    
    return float(matched.mean())
```

**Observed (real) value:** 0.6222 (var=0.0019)
**Predicted under pi_14:** 0.4383 (var=0.0344)
**Predicted under pi_15:** 0.5025 (var=0.0859)

### Experiment 19
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    mask = abs(tally_a - tally_b) == 1
    if not mask.any():
        return 0.0
        
    lower_tally_is_b = (tally_b[mask] < tally_a[mask]).astype(int)
    responses = data.loc[mask, 'response']
    
    chose_lower_tally = (responses == lower_tally_is_b).mean()
    return float(chose_lower_tally)
```

**Observed (real) value:** 0.5092 (var=0.0035)
**Predicted under pi_14:** 0.4750 (var=0.0259)
**Predicted under pi_15:** 0.3979 (var=0.0637)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    
    diff_1_mask = abs(tally_a - tally_b) == 1
    subset = data[diff_1_mask]
    
    if len(subset) == 0:
        return 0.5
        
    chose_higher_tally = ((tally_a[diff_1_mask] > tally_b[diff_1_mask]) & (subset['response'] == 0)) | ((tally_b[diff_1_mask] > tally_a[diff_1_mask]) & (subset['response'] == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.6858 (var=0.0032)
**Predicted under pi_14:** 0.4379 (var=0.0077)
**Predicted under pi_15:** 0.5142 (var=0.0687)

### Experiment 21
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_a = data['option_a_ratings'].apply(sum)
    tally_b = data['option_b_ratings'].apply(sum)
    diff = (tally_a - tally_b).abs()
    
    mask = diff >= 2
    if not mask.any():
        return 0.5
    
    subset = data[mask]
    ta = tally_a[mask]
    tb = tally_b[mask]
    resp = subset['response']
    
    chose_higher_tally = ((ta > tb) & (resp == 0)) | ((tb > ta) & (resp == 1))
    
    return float(chose_higher_tally.mean())
```

**Observed (real) value:** 0.3050 (var=0.0041)
**Predicted under pi_14:** 0.4775 (var=0.0208)
**Predicted under pi_15:** 0.5196 (var=0.0712)

### Experiment 22
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t2_mask = (a_str == '10000') & (b_str == '01110')
    t3_mask = (a_str == '10000') & (b_str == '01111')
    t7_mask = (a_str == '01110') & (b_str == '10000')
    
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t7 = (data.loc[t7_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    if pd.isna(p_a_t3): p_a_t3 = 0.5
    if pd.isna(p_a_t7): p_a_t7 = 0.5
    
    return float(p_a_t2 + p_a_t3 - p_a_t7)
```

**Observed (real) value:** 0.8700 (var=0.0323)
**Predicted under pi_14:** 0.3583 (var=0.2623)
**Predicted under pi_15:** 0.5500 (var=0.7611)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    t10_mask = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 1, 1))
    t5_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 1, 0, 0, 0))
    
    p_a_t10 = (data.loc[t10_mask, 'response'] == 0).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t10): p_a_t10 = 0.5
    if pd.isna(p_a_t5): p_a_t5 = 0.5
    
    return float(p_a_t10 - p_a_t5)
```

**Observed (real) value:** -0.6578 (var=0.0661)
**Predicted under pi_14:** -0.1444 (var=0.0633)
**Predicted under pi_15:** -0.5400 (var=0.0440)

### Experiment 24
**Design**
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where cue 1 and cue 2 conflict
    conflict = (a_ratings[:, 0] != b_ratings[:, 0]) & \
               (a_ratings[:, 1] != b_ratings[:, 1]) & \
               (a_ratings[:, 0] != a_ratings[:, 1])
               
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    
    a_has_cue1 = a_ratings[:, 0] > b_ratings[:, 0]
    b_has_cue1 = b_ratings[:, 0] > a_ratings[:, 0]
    
    # Check if the chosen option is the one favored by cue 1
    chose_cue1 = (a_has_cue1 & (responses == 0)) | (b_has_cue1 & (responses == 1))
    
    return float(np.mean(chose_cue1[conflict]))
```

**Observed (real) value:** 0.6108 (var=0.0044)
**Predicted under pi_14:** 0.5554 (var=0.0078)
**Predicted under pi_15:** 0.8071 (var=0.0139)

### Experiment 25
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5 where Option A has the single best cue and Option B has all other 5 cues.
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 1, 1, 1))
    
    if t5_mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen in this specific trial
    return float((data.loc[t5_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.8600 (var=0.0198)
**Predicted under pi_14:** 0.6533 (var=0.0332)
**Predicted under pi_15:** 0.5783 (var=0.1046)

### Experiment 26
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    sum_a = a.sum(axis=1)
    sum_b = b.sum(axis=1)
    
    # HVO (High Validity Option) is the one with the best cue (index 0)
    # We filter for trials where the HVO is outnumbered in simple tally
    cond_a = (a[:, 0] == 1) & (b[:, 0] == 0) & (sum_a < sum_b)
    cond_b = (b[:, 0] == 1) & (a[:, 0] == 0) & (sum_b < sum_a)
    
    hvo_chosen = np.sum((cond_a & (resp == 0)) | (cond_b & (resp == 1)))
    total = np.sum(cond_a | cond_b)
    
    return float(hvo_chosen / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5813 (var=0.0017)
**Predicted under pi_14:** 0.6217 (var=0.0282)
**Predicted under pi_15:** 0.6440 (var=0.0786)

### Experiment 27
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where options A and B are identical on the two highest validity cues (cues 0 and 1).
    # In the experimental design, these are trials 2, 4, and 6.
    # For these trials, the only differences lie in the chance-level (0.5 validity) cues (cues 2, 3, and 4),
    # and option B always has more positive features on these cues than option A.
    is_target = data.apply(lambda row: row['option_a_ratings'][0] == row['option_b_ratings'][0] and 
                                       row['option_a_ratings'][1] == row['option_b_ratings'][1], axis=1)
    
    target_data = data[is_target]
    if len(target_data) == 0:
        return 0.5
    
    # Return the proportion of times the subject chose option B (response == 1)
    return target_data['response'].mean()
```

**Observed (real) value:** 0.6267 (var=0.0044)
**Predicted under pi_14:** 0.5204 (var=0.0072)
**Predicted under pi_15:** 0.7308 (var=0.0149)

### Experiment 28
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
    t4 = data[(data['a_str'] == '10000') & (data['b_str'] == '01111')]
    # Trial 5: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
    t5 = data[(data['a_str'] == '10111') & (data['b_str'] == '01000')]
    
    p_a_t4 = 1.0 - t4['response'].mean() if len(t4) > 0 else 0.5
    p_a_t5 = 1.0 - t5['response'].mean() if len(t5) > 0 else 0.5
    
    return float(p_a_t5 - p_a_t4)
```

**Observed (real) value:** -0.6337 (var=0.0446)
**Predicted under pi_14:** 0.1200 (var=0.0830)
**Predicted under pi_15:** 0.1747 (var=0.0875)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 28 experiments, both Theory 1 (Sequential Evidence Accumulation) and Theory 2 (Strategy Selection Mixture) struggle to capture the full spectrum of human decision-making, frequently predicting the wrong magnitude or even the wrong direction of effects. For instance, in Experiment 1, pi_15 is much closer to the observed data (0.1575) than pi_14 (0.5469). Conversely, in Experiment 5, pi_14 captures the observed effect (0.1389) perfectly, while pi_15 overestimates it. More importantly, both theories fail dramatically on experiments like Experiment 7 and Experiment 28, where the observed metrics are strongly negative (-0.4333 and -0.6337, respectively), yet both models predict values near zero or positive. This indicates that neither a purely sequential threshold model nor a simple TTB/Tallying mixture can account for the nuanced ways subjects weigh conflicting cues, particularly when dealing with extreme compensatory structures or specific feature contrasts. A fundamentally new theoretical approach is required to resolve these deep inconsistencies.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory based on 'Attention-Weighted Feature Contrasts'. Instead of accumulating evidence sequentially or mixing two rigid heuristics, decision-makers might evaluate options by computing cue-wise differences and weighting them dynamically based on both the cue's objective validity and the overall conflict in the trial. When cues heavily conflict (e.g., one option has the best cue, the other has all remaining cues), attention is drawn disproportionately to the sheer number of features (a compensatory shift). This theory should feature a continuous integration of cue differences modulated by an attention mechanism that scales the non-linearity of weights depending on the dispersion of the current trial's feature matrix, allowing it to gracefully capture both the strong non-compensatory TTB-like choices and the paradoxical compensatory reversals seen in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 21425,
  "candidates_token_count": 434,
  "total_token_count": 23394
}
```
