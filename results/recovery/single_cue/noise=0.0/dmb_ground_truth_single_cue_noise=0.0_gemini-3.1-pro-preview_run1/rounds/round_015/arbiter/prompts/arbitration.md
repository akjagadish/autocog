# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_17" and "pi_16") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_17" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_16" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_17
Rank-Dependent Continuous WADD with Additive Tie-Breaking: Decision-makers use a continuous Weighted Additive (WADD) strategy where subjective weights are determined by a combination of an exponential decay over cue rank (capturing TTB and Tallying) and a small additive linear boost for lower-ranked cues (capturing inverse-validity tie-breaking).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    n = len(a)
    ranks = np.arange(n)
    
    # Compute a single set of unnormalized subjective weights
    # Exponential decay for primary TTB/Tallying behavior
    # Small linear boost for lower-validity cues (tie-breaking)
    raw_weights = np.exp(-gamma * ranks) + delta * ranks
    w_combined = raw_weights / np.sum(raw_weights)
    
    # Continuous WADD evaluation
    score_a = np.sum(a * w_combined)
    score_b = np.sum(b * w_combined)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_16
Context-Dependent Dual-Process Mixture of TTB and Tallying with Inverse Validity Tie-Breaking: Decision-makers rely on a mixture of Take-The-Best (TTB) and Tallying, but the mixture weight is dynamically determined by the environment. When cue validities are highly dispersed (measured by the standard deviation of the validities), subjects predominantly use TTB; when validities are similar, they rely on Tallying. When Tallying results in a tie, subjects resolve it using an inverse-validity weighting mechanism, heavily favoring options with positive features among the lower-validity (or more recently processed) cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    disp_slope = float(parameters["disp_slope"])
    disp_threshold = float(parameters["disp_threshold"])
    w_tie = float(parameters["w_tie"])
    gamma = float(parameters["gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_ttb = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying with Inverse Validity Tie-Breaker ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # w_tie < 1.0 ensures the tie-breaker only dictates choice when a_wins == b_wins
    score_a_tally = a_wins + w_tie * tie_score_a
    score_b_tally = b_wins + w_tie * tie_score_b
    
    scores_tally = np.array([score_a_tally, score_b_tally])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    probs_tally = e_tally / np.sum(e_tally)
    
    # --- Strategy 2: Take-The-Best (TTB) ---
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # --- Mixture Model ---
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_tally
    
    # --- Lapse Rate ---
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_17)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Rank-Dependent Continuous WADD) from the Competing Theory (Context-Dependent Mixture of TTB and Tallying with Inverse Validity Tie-Breaking), we exploit their different uses of feature indices versus explicit validities. The Advocated Theory assigns weights based purely on spatial rank (left-to-right), combining an exponential decay (favoring the left) and a linear boost (favoring the right). It completely ignores the explicitly provided validities. The Competing Theory uses the actual validities to determine the mixture weight and the inverse-validity tie-breaker, while its TTB component scans left-to-right. By setting the validities to monotonically decrease, we align the left-to-right TTB bias with the highest validities, but force the Competing Theory's inverse-validity tie-breaker to strongly favor the right-most (lowest validity) features. We include 'Compensatory' trials to test the strength of the left-most feature against multiple right-most features, and 'Tally Tie' trials where the left-to-right TTB component conflicts with the right-favoring inverse-validity tie-breaker in the Competing Theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 2, Option A has features [0, 0, 0, 1, 1] and Option B has [1, 1, 1, 0, 0]. The Competing Theory will almost exclusively choose Option B, because Option B wins both under Take-The-Best (it possesses the highest-validity feature 0) and under Tallying (it has 3 positive features compared to A's 2). Thus, the Competing Theory's choice rate for A will be near zero (driven only by the random lapse rate). Conversely, the Advocated Theory (Rank-Dependent WADD) includes a linear boost (`delta * ranks`) that increases the weight of lower-validity (right-most) features. Option A possesses the two lowest-validity features (ranks 3 and 4), which receive the maximum linear boost. For a wide range of parameters (specifically when the `delta` parameter is moderately high), this linear boost overcomes the exponential decay of the higher-validity features, causing the Advocated Theory to frequently choose Option A. This creates a large and reliable divergence in the choice rate of Option A between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    mask = (data['option_a_ratings'].apply(tuple) == (0, 0, 0, 1, 1)) & \
           (data['option_b_ratings'].apply(tuple) == (1, 1, 1, 0, 0))
    if mask.sum() == 0:
        return 0.0
    return float((data.loc[mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_17 (simulated): 0.5925 (var=0.0689)
- Predicted under pi_16 (simulated): 0.1787 (var=0.0231)
- Observed on real data: 0.8625 (var=0.0128)

## EXPERIMENT 2 (proposed by pi_16)

### DESIGN
**Validities (n_features=5):** [0.95, 0.55, 0.85, 0.5, 0.9]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Context-Dependent Mixture of TTB and Tallying with Inverse Validity Tie-Breaking) from the Competing Theory (Rank-Dependent Continuous WADD), we exploit their differing dependencies on explicit validities versus spatial rank. The Competing Theory completely ignores explicit validities, relying entirely on spatial rank (an exponential decay favoring left-most features and a linear boost favoring right-most features). The Advocated Theory uses the explicit validities for its tie-breaking mechanism, strongly favoring features with lower validities. By using a non-monotonic validity array where low validities and high validities are interleaved, we create 'Tally Tie' trials where the Advocated Theory's inverse-validity tie-breaker makes deterministic predictions that conflict with the Competing Theory's rank-based weights. We also include 'Compensatory' trials to further differentiate the mixture model's TTB component from continuous rank-based WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric leverages trials where the Advocated Theory makes unanimous predictions across both of its sub-strategies (TTB and Tallying with Tie-Breaker), whereas the Competing Theory is deeply conflicted depending on its parameters. In Trial 2, both TTB and the Inverse-Validity Tie-Breaker favor option B, making the Advocated Theory strongly predict B. However, for the Competing Theory (Rank-Dependent WADD), a high delta parameter will cause it to favor option A because A's feature is further to the right. Similarly, in Trial 3, both TTB and the Tie-Breaker favor option A, while the Competing Theory with a high delta will favor option B. By summing the probability of choosing B in Trial 2 and A in Trial 3, we obtain a metric that will be consistently high for the Advocated Theory (low variance) and significantly lower for the Competing Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 2: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
    t2_mask = (a_str == '00100') & (b_str == '01000')
    
    # Trial 3: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
    t3_mask = (a_str == '01010') & (b_str == '00101')
    
    p_B_t2 = data.loc[t2_mask, 'response'].mean() if t2_mask.sum() > 0 else 0.5
    p_A_t3 = 1.0 - data.loc[t3_mask, 'response'].mean() if t3_mask.sum() > 0 else 0.5
    
    return float(p_B_t2 + p_A_t3)

### RESULTS
- Predicted under pi_17 (simulated): 0.7133 (var=0.0945)
- Predicted under pi_16 (simulated): 1.4933 (var=0.1122)
- Observed on real data: 1.3533 (var=0.0357)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_ttb = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction: first discriminating cue
        ttb_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction: majority of discriminating cues
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only consider trials where the two heuristics deterministically disagree
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if row['response'] == ttb_winner:
                match_ttb += 1
            total += 1
            
    if total == 0:
        return 0.5
    return match_ttb / total
```

**Observed (real) value:** 0.2775 (var=0.0072)
**Predicted under pi_17:** 0.2121 (var=0.0249)
**Predicted under pi_16:** 0.2404 (var=0.0425)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            if row['response'] == tally_pref:
                tally_match += 1
            total += 1
            
    return float(tally_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8178 (var=0.0246)
**Predicted under pi_17:** 0.7794 (var=0.0293)
**Predicted under pi_16:** 0.7650 (var=0.0614)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a_sum = sum(row['option_a_ratings'])
        b_sum = sum(row['option_b_ratings'])
        if a_sum == 2 and b_sum == 3:
            if row['response'] == 0:
                wadd_consistent += 1
            total += 1
        elif a_sum == 3 and b_sum == 2:
            if row['response'] == 1:
                wadd_consistent += 1
            total += 1
    if total == 0:
        return 0.5
    return wadd_consistent / total
```

**Observed (real) value:** 0.1400 (var=0.0117)
**Predicted under pi_17:** 0.2150 (var=0.0303)
**Predicted under pi_16:** 0.2850 (var=0.0784)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_t1 = [0, 0, 1, 1, 1]
    B_t1 = [1, 1, 0, 0, 0]
    A_t2 = [1, 1, 0, 0, 0]
    B_t2 = [0, 0, 1, 1, 1]
    
    consistencies = []
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        r = row['response']
        
        # Trial 1: Tallying prefers A (3 wins vs 2), WADD prefers B (1.90 vs 1.65)
        if a == A_t1 and b == B_t1:
            consistencies.append(1 if r == 0 else 0)
        # Trial 2: Tallying prefers B (3 wins vs 2), WADD prefers A (1.90 vs 1.65)
        elif a == A_t2 and b == B_t2:
            consistencies.append(1 if r == 1 else 0)
            
    if not consistencies:
        return 0.5
    return float(np.mean(consistencies))
```

**Observed (real) value:** 0.8354 (var=0.0165)
**Predicted under pi_17:** 0.7023 (var=0.0710)
**Predicted under pi_16:** 0.6800 (var=0.0875)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        for i in range(len(a)):
            if a[i] > b[i]:
                if resp == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if resp == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.2194 (var=0.0145)
**Predicted under pi_17:** 0.3025 (var=0.0249)
**Predicted under pi_16:** 0.2720 (var=0.0569)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Extract the highest validity feature (feature 0) for both options
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    b0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Calculate the total number of feature-wise wins for each option
    a_wins = data.apply(lambda row: sum(a > b for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    b_wins = data.apply(lambda row: sum(b > a for a, b in zip(row['option_a_ratings'], row['option_b_ratings'])), axis=1)
    
    # Identify critical trials where the option favored by the most valid cue (feature 0) 
    # is actually the option with FEWER total winning features.
    # This perfectly dissociates Take The Best (which follows feature 0) 
    # from Tallying (which follows the total number of wins).
    critical = ((a0 == 1) & (a_wins < b_wins)) | ((b0 == 1) & (b_wins < a_wins))
    
    if not critical.any():
        return 0.5
        
    crit_data = data[critical]
    
    # The choice predicted by TTB is exactly the option that has a 1 on feature 0.
    # Since feature 0 always discriminates in this design, B[0] == 1 means TTB chooses B (1),
    # and B[0] == 0 means TTB chooses A (0).
    ttb_choice = crit_data['option_b_ratings'].apply(lambda x: x[0])
    
    # Return the proportion of times the subject's response matches the TTB prediction on these critical trials.
    # TTB will score near 1.0, while Tallying will score near 0.0.
    return float((crit_data['response'] == ttb_choice).mean())
```

**Observed (real) value:** 0.6650 (var=0.0076)
**Predicted under pi_17:** 0.5029 (var=0.0545)
**Predicted under pi_16:** 0.3008 (var=0.0780)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert lists to strings for hashable comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t1_mask = (a_str == '11000') & (b_str == '00111')
    # Trial 2: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 0, 0]
    t2_mask = (a_str == '10000') & (b_str == '01100')
    
    # Response 0 means option A was chosen
    p_a_t1 = (data[t1_mask]['response'] == 0).mean()
    p_a_t2 = (data[t2_mask]['response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.0
    if pd.isna(p_a_t2): p_a_t2 = 0.0
    
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** -0.3850 (var=0.0268)
**Predicted under pi_17:** -0.1312 (var=0.0425)
**Predicted under pi_16:** 0.0012 (var=0.0130)

### Experiment 10
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Identify "tie" trials where Tallying sees an equal number of wins (2 vs 2)
    t2_mask = (a_keys == (1, 0, 1, 0, 0)) & (b_keys == (0, 1, 0, 1, 0))
    t6_mask = (a_keys == (0, 1, 0, 0, 1)) & (b_keys == (1, 0, 0, 1, 0))
    t8_mask = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 0, 1))
    
    tie_mask = t2_mask | t6_mask | t8_mask
    tie_data = data[tie_mask]
    
    if len(tie_data) == 0:
        return 0.0
        
    def subject_score(sub_df):
        a = sub_df['option_a_ratings'].apply(tuple)
        b = sub_df['option_b_ratings'].apply(tuple)
        
        m2 = (a == (1, 0, 1, 0, 0)) & (b == (0, 1, 0, 1, 0))
        m6 = (a == (0, 1, 0, 0, 1)) & (b == (1, 0, 0, 1, 0))
        m8 = (a == (0, 1, 1, 0, 0)) & (b == (1, 0, 0, 0, 1))
        
        score = 0.0
        count = 0
        for m in [m2, m6, m8]:
            if m.sum() > 0:
                prop_a = (sub_df.loc[m, 'response'] == 0).mean()
                score += abs(prop_a - 0.5)
                count += 1
        return score / count if count > 0 else 0.0
        
    return float(tie_data.groupby('subject_id').apply(subject_score).mean())
```

**Observed (real) value:** 0.2700 (var=0.0052)
**Predicted under pi_17:** 0.2239 (var=0.0131)
**Predicted under pi_16:** 0.2172 (var=0.0120)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Isolate trials where Tallying is perfectly tied
        if a_wins == b_wins:
            # Determine TTB prediction (first discriminating feature)
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_pred = 0
                    break
                elif b[i] > a[i]:
                    ttb_pred = 1
                    break
            else:
                continue
                
            ttb_match.append(1 if row['response'] == ttb_pred else 0)
            
    if len(ttb_match) == 0:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.4567 (var=0.0102)
**Predicted under pi_17:** 0.7342 (var=0.0223)
**Predicted under pi_16:** 0.5992 (var=0.0233)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tie_mask = (a_wins == b_wins)
    if not np.any(tie_mask):
        return 0.5
        
    a_ties = a_ratings[tie_mask]
    b_ties = b_ratings[tie_mask]
    responses = data['response'].values[tie_mask]
    
    diff = a_ties - b_ties
    idx = np.argmax(diff != 0, axis=1)
    first_diffs = diff[np.arange(len(diff)), idx]
    ttb_choices = (first_diffs < 0).astype(int)
    
    return float(np.mean(responses == ttb_choices))
```

**Observed (real) value:** 0.4967 (var=0.0079)
**Predicted under pi_17:** 0.4328 (var=0.0118)
**Predicted under pi_16:** 0.5306 (var=0.0316)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_is_11000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    b_is_00110 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0))
    t5_mask = a_is_11000 & b_is_00110
    if not t5_mask.any():
        return 0.5
    
    # response == 0 means choice A
    return float((data.loc[t5_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.1250 (var=0.0066)
**Predicted under pi_17:** 0.3575 (var=0.0447)
**Predicted under pi_16:** 0.4612 (var=0.0872)

### Experiment 14
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Isolate Trial 3, which is a perfect tie under Tallying (A wins 2, B wins 2)
    # but has a massive Weighted Additive Difference (WADD) favoring Option A (0.225).
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t3_mask = (data['a_str'] == '11000') & (data['b_str'] == '00110')
    
    if not t3_mask.any():
        return 0.5
        
    # Return the empirical probability of choosing Option A on Trial 3
    return float((data.loc[t3_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.2062 (var=0.0240)
**Predicted under pi_17:** 0.3769 (var=0.0836)
**Predicted under pi_16:** 0.3785 (var=0.0951)

### Experiment 15
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: A=[1, 0, 0, 0, 1] vs B=[0, 1, 1, 0, 0]
    t2_mask = (a_tuples == (1, 0, 0, 0, 1))
    # Trial 8: A=[0, 1, 1, 0, 0] vs B=[1, 0, 0, 0, 1]
    t8_mask = (a_tuples == (0, 1, 1, 0, 0))
    
    val = 0.0
    if t2_mask.any():
        val += (data.loc[t2_mask, 'response'] == 0).mean()
    if t8_mask.any():
        val += (data.loc[t8_mask, 'response'] == 1).mean()
        
    return float(val)
```

**Observed (real) value:** 1.6900 (var=0.0225)
**Predicted under pi_17:** 1.6033 (var=0.0824)
**Predicted under pi_16:** 1.4683 (var=0.1083)

### Experiment 16
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    thp_alignments = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus exclusively on Tally-Tie trials
        if a_wins == b_wins:
            thp_weights = np.arange(1, len(a) + 1)
            thp_a = np.sum(a * thp_weights)
            thp_b = np.sum(b * thp_weights)
            
            # Check if the subject's choice aligns with the Top-Heavy Penalty preference
            if thp_b > thp_a:
                thp_alignments.append(1.0 if row['response'] == 1 else 0.0)
            elif thp_a > thp_b:
                thp_alignments.append(1.0 if row['response'] == 0 else 0.0)
                
    if not thp_alignments:
        return 0.5
    return float(np.mean(thp_alignments))
```

**Observed (real) value:** 0.5337 (var=0.0084)
**Predicted under pi_17:** 0.5906 (var=0.0418)
**Predicted under pi_16:** 0.6044 (var=0.0181)

### Experiment 17
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.vstack(data['option_a_ratings'].values)
    B = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(A > B, axis=1)
    b_wins = np.sum(B > A, axis=1)
    
    ties = (a_wins == b_wins)
    
    if not np.any(ties):
        return 0.5
        
    weights = np.arange(1, A.shape[1] + 1)
    recency_a = np.sum(A * weights, axis=1)
    recency_b = np.sum(B * weights, axis=1)
    
    expected_response = np.where(recency_a > recency_b, 0, 1)
    actual_response = data['response'].values
    
    match = (expected_response == actual_response)
    valid = ties & (recency_a != recency_b)
    
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(match[valid]))

```

**Observed (real) value:** 0.7422 (var=0.0077)
**Predicted under pi_17:** 0.6092 (var=0.0454)
**Predicted under pi_16:** 0.6303 (var=0.0456)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    consistent_choices = 0
    total_eligible = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a == tally_b:
            weights = np.arange(1, len(a) + 1)
            recency_a = np.sum(a * weights)
            recency_b = np.sum(b * weights)
            
            if recency_a != recency_b:
                recency_choice = 0 if recency_a > recency_b else 1
                if row['response'] == recency_choice:
                    consistent_choices += 1
                total_eligible += 1
                
    if total_eligible == 0:
        return 0.5
    return float(consistent_choices / total_eligible)

```

**Observed (real) value:** 0.5025 (var=0.0037)
**Predicted under pi_17:** 0.5058 (var=0.0537)
**Predicted under pi_16:** 0.5650 (var=0.0382)

### Experiment 19
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['trial_type'] = data.apply(lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])), axis=1)
    data['is_tie'] = data.apply(lambda x: sum(x['option_a_ratings']) == sum(x['option_b_ratings']), axis=1)
    
    ties = data[data['is_tie']]
    if len(ties) == 0:
        return 0.0
        
    grouped = ties.groupby(['subject_id', 'trial_type'])['response'].apply(lambda x: np.mean(x == 0)).reset_index()
    subj_devs = grouped.groupby('subject_id')['response'].apply(lambda x: np.mean(np.abs(x - 0.5)))
    
    return float(np.mean(subj_devs))
```

**Observed (real) value:** 0.2442 (var=0.0046)
**Predicted under pi_17:** 0.2317 (var=0.0103)
**Predicted under pi_16:** 0.2275 (var=0.0112)

### Experiment 20
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def check_tie(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a > b) == np.sum(b > a)
        
    is_tie = data.apply(check_tie, axis=1)
    tie_data = data[is_tie].copy()
    
    if len(tie_data) == 0:
        return 0.0
        
    tie_data['trial_id'] = tie_data.apply(
        lambda x: tuple(x['option_a_ratings']) + tuple(x['option_b_ratings']), axis=1
    )
    
    tie_data['chose_a'] = (tie_data['response'] == 0).astype(float)
    
    means = tie_data.groupby(['subject_id', 'trial_id'])['chose_a'].mean()
    
    return float(np.abs(means - 0.5).mean())
```

**Observed (real) value:** 0.3800 (var=0.0052)
**Predicted under pi_17:** 0.2556 (var=0.0050)
**Predicted under pi_16:** 0.2315 (var=0.0124)

### Experiment 21
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify tie trials where both options have the same number of positive features
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    ties = data[a_sums == b_sums].copy()
    
    if len(ties) == 0:
        return 0.0
        
    # Create a string representation for the pair to group by unique trial types
    ties['pair_str'] = ties['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + \
                       ties['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
                       
    # Calculate proportion of choosing A (response == 0) for each subject and pair
    p_A = ties.groupby(['subject_id', 'pair_str'])['response'].apply(lambda x: (x == 0).mean())
    
    # Calculate absolute deviation from 0.5
    dev = (p_A - 0.5).abs().mean()
    
    return float(dev)
```

**Observed (real) value:** 0.1694 (var=0.0026)
**Predicted under pi_17:** 0.2087 (var=0.0078)
**Predicted under pi_16:** 0.1856 (var=0.0105)

### Experiment 22
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def is_tie(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a > b) == np.sum(b > a)
    
    tie_mask = data.apply(is_tie, axis=1)
    tie_data = data[tie_mask].copy()
    
    if len(tie_data) == 0:
        return 0.0
        
    tie_data['trial_id'] = tie_data.apply(
        lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])),
        axis=1
    )
    
    subject_trial_means = tie_data.groupby(['subject_id', 'trial_id'])['response'].mean()
    abs_dev = np.abs(subject_trial_means - 0.5)
    
    return float(abs_dev.mean())
```

**Observed (real) value:** 0.2308 (var=0.0031)
**Predicted under pi_17:** 0.2304 (var=0.0076)
**Predicted under pi_16:** 0.2512 (var=0.0105)

### Experiment 23
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = (a > b).astype(float)
        b_wins = (b > a).astype(float)
        
        tally_a = np.sum(a_wins)
        tally_b = np.sum(b_wins)
        
        if tally_a == tally_b:
            tie_count += 1
            val_a = np.sum(validities * a_wins)
            val_b = np.sum(validities * b_wins)
            
            if val_a > val_b and row['response'] == 0:
                match_count += 1
            elif val_b > val_a and row['response'] == 1:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
    return float(match_count / tie_count)
```

**Observed (real) value:** 0.2394 (var=0.0086)
**Predicted under pi_17:** 0.3641 (var=0.0164)
**Predicted under pi_16:** 0.3588 (var=0.0413)

### Experiment 24
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val_favored = 0
    val_count = 0
    strict_correct = 0
    strict_count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # T1: Tally Tie. A has higher validity (0.95+0.85 > 0.75+0.65)
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            val_favored += 1 if resp == 0 else 0
            val_count += 1
        elif a == (0, 0, 1, 1, 0) and b == (1, 1, 0, 0, 0):
            val_favored += 1 if resp == 1 else 0
            val_count += 1
            
        # T2: Tally Tie. A has higher validity (0.85+0.75 > 0.65+0.55)
        elif a == (0, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            val_favored += 1 if resp == 0 else 0
            val_count += 1
        elif a == (0, 0, 0, 1, 1) and b == (0, 1, 1, 0, 0):
            val_favored += 1 if resp == 1 else 0
            val_count += 1
            
        # T5: Tally Tie. A has higher validity (0.95 > 0.85)
        elif a == (1, 0, 0, 0, 0) and b == (0, 1, 0, 0, 0):
            val_favored += 1 if resp == 0 else 0
            val_count += 1
        elif a == (0, 1, 0, 0, 0) and b == (1, 0, 0, 0, 0):
            val_favored += 1 if resp == 1 else 0
            val_count += 1
            
        # T6: Tally Tie. B has higher validity (0.85+0.75 > 0.95+0.55)
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 0, 0):
            val_favored += 1 if resp == 1 else 0
            val_count += 1
        elif a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 0, 1):
            val_favored += 1 if resp == 0 else 0
            val_count += 1
            
        # T3: Strict Win (A wins 3-2)
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            strict_correct += 1 if resp == 0 else 0
            strict_count += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            strict_correct += 1 if resp == 1 else 0
            strict_count += 1
            
        # T4: Strict Win (B wins 3-2)
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 1):
            strict_correct += 1 if resp == 1 else 0
            strict_count += 1
        elif a == (0, 1, 0, 1, 1) and b == (1, 0, 1, 0, 0):
            strict_correct += 1 if resp == 0 else 0
            strict_count += 1

    if val_count == 0 or strict_count == 0:
        return 0.0
        
    strict_acc = strict_correct / strict_count
    val_rate = val_favored / val_count
    
    w = max(0.0, strict_acc - 0.5) * 2.0
    return float((val_rate - 0.5) * w)

```

**Observed (real) value:** -0.1124 (var=0.0074)
**Predicted under pi_17:** -0.0538 (var=0.0073)
**Predicted under pi_16:** -0.0579 (var=0.0096)

### Experiment 25
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    ties = (a_wins == b_wins)
    if not np.any(ties):
        return 0.5
        
    responses = data['response'].values[ties]
    
    a_f4 = a_ratings[ties, 3]
    b_f4 = b_ratings[ties, 3]
    
    chosen_f4 = np.where(responses == 0, a_f4, b_f4)
    
    return float(np.mean(chosen_f4 == 1))
```

**Observed (real) value:** 0.8230 (var=0.0090)
**Predicted under pi_17:** 0.6673 (var=0.0183)
**Predicted under pi_16:** 0.6785 (var=0.0274)

### Experiment 26
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    aligned_choices = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Only look at Tally Tie trials
        if a_wins == b_wins:
            score_a = np.sum(a * weights)
            score_b = np.sum(b * weights)
            
            if score_a > score_b:
                aligned_choices.append(row['response'] == 0)
            elif score_b > score_a:
                aligned_choices.append(row['response'] == 1)
                
    if len(aligned_choices) == 0:
        return 0.5
        
    return float(np.mean(aligned_choices))
```

**Observed (real) value:** 0.6750 (var=0.0048)
**Predicted under pi_17:** 0.6188 (var=0.0335)
**Predicted under pi_16:** 0.6204 (var=0.0309)

### Experiment 27
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of ratings for A and B to identify tally tie trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for tie trials
    tie_mask = sum_a == sum_b
    tie_data = data[tie_mask]
    
    if len(tie_data) == 0:
        return 0.5
        
    # Return the proportion of times Option B was chosen on tie trials
    return float(tie_data['response'].mean())
```

**Observed (real) value:** 0.8183 (var=0.0179)
**Predicted under pi_17:** 0.5683 (var=0.0455)
**Predicted under pi_16:** 0.6454 (var=0.0506)

### Experiment 28
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.vstack(data['option_a_ratings'].values)
    b = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    
    tie_trials = a_wins == b_wins
    
    if np.sum(tie_trials) == 0:
        return 0.5
        
    return float(np.mean(data['response'].values[tie_trials]))
```

**Observed (real) value:** 0.6731 (var=0.0071)
**Predicted under pi_17:** 0.5941 (var=0.0326)
**Predicted under pi_16:** 0.5716 (var=0.0415)

### Experiment 29
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = data['option_a_ratings'].apply(tuple)
    
    # Identify the critical 'Tally Tie' trials where the two theories diverge.
    mask1 = a_ratings == (1, 1, 0, 0, 0)
    mask2 = a_ratings == (0, 0, 1, 0, 1)
    mask3 = a_ratings == (0, 1, 1, 0, 0)
    
    # Calculate choices aligned with the Advocated Theory (Inverse Validity tie-breaker)
    # Trial 1: Advocated favors A (0), Competing favors B (1)
    score1 = np.sum((data['response'] == 0) & mask1)
    # Trial 2: Advocated favors B (1), Competing favors A (0)
    score2 = np.sum((data['response'] == 1) & mask2)
    # Trial 3: Advocated favors A (0), Competing favors B (1)
    score3 = np.sum((data['response'] == 0) & mask3)
    
    total = np.sum(mask1) + np.sum(mask2) + np.sum(mask3)
    
    if total == 0:
        return 0.5
    return float((score1 + score2 + score3) / total)

```

**Observed (real) value:** 0.8556 (var=0.0083)
**Predicted under pi_17:** 0.2539 (var=0.0211)
**Predicted under pi_16:** 0.7567 (var=0.0238)

### Experiment 30
**Design**
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_tally_tie(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a > b) == np.sum(b > a)
        
    tie_mask = data.apply(is_tally_tie, axis=1)
    tie_data = data[tie_mask]
    
    if len(tie_data) == 0:
        return 0.5
        
    return float(tie_data['response'].mean())
```

**Observed (real) value:** 0.7893 (var=0.0105)
**Predicted under pi_17:** 0.4093 (var=0.0347)
**Predicted under pi_16:** 0.7540 (var=0.0143)

### Experiment 31
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    adv_matches = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Take-The-Best (TTB) prediction
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Advocated Theory prediction (Tallying + Inverse Validity Tie-Breaker)
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        if sum_a > sum_b:
            adv_pred = 0
        elif sum_b > sum_a:
            adv_pred = 1
        else:
            # Tie breaker: sum of indices (higher index = lower validity = preferred by IV)
            idx_a = np.sum(np.arange(len(a)) * a)
            idx_b = np.sum(np.arange(len(b)) * b)
            if idx_a > idx_b:
                adv_pred = 0
            elif idx_b > idx_a:
                adv_pred = 1
            else:
                adv_pred = None
                
        # Only evaluate on trials where TTB and Advocated Theory disagree
        if ttb_pred is not None and adv_pred is not None and ttb_pred != adv_pred:
            conflict_trials += 1
            if row['response'] == adv_pred:
                adv_matches += 1
                
    if conflict_trials == 0:
        return 0.5
    return float(adv_matches / conflict_trials)
```

**Observed (real) value:** 0.6000 (var=0.0032)
**Predicted under pi_17:** 0.6114 (var=0.0387)
**Predicted under pi_16:** 0.6292 (var=0.0490)

### Experiment 32
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match_count = 0
    total = len(data)
    if total == 0:
        return 0.0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = 0
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if row['response'] == ttb_choice:
            ttb_match_count += 1
            
    return float(ttb_match_count) / total
```

**Observed (real) value:** 0.2742 (var=0.0047)
**Predicted under pi_17:** 0.2796 (var=0.0424)
**Predicted under pi_16:** 0.3408 (var=0.0624)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Both theories capture certain aspects of the observed data but fail significantly on specific experiments. Theory 1 (pi_17) performs better on Experiments 1, 8, 9, and 15, where its rank-dependent continuous WADD mechanism with additive tie-breaking captures the preference for lower-ranked cues. However, it fails dramatically on Experiments 2, 29, and 30, where Theory 2 (pi_16) excels. Theory 2's explicit use of validities for tie-breaking and its context-dependent mixture nicely captures the variance in those experiments, but it fundamentally mispredicts the strong deviations observed in Experiment 1 and 8. Because both theories have mutually exclusive blind spots, a new theory is needed to reconcile these mechanisms.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that integrates the continuous, compensatory nature of WADD with context-dependent heuristics. Specifically, the new theory should be a Dual-Process mixture of a Validity-Weighted Additive strategy and Tallying. Instead of purely spatial rank or pure inverse-validity tie-breaking, the subjective weights should explicitly combine the provided cue validities with a spatial recency effect (boosting features processed later/right-most). The mixture between WADD and Tallying could be driven by the dispersion of validities, allowing the model to act like Tallying when validities are similar, but reverting to a recency-boosted WADD when validities are highly dispersed. This hybrid approach will capture the continuous trade-offs seen in Exp 1 and 15, while maintaining the discrete tallying ties seen in Exp 29 and 30."
}
```

## Usage

```json
{
  "prompt_token_count": 24681,
  "candidates_token_count": 380,
  "total_token_count": 25944
}
```
