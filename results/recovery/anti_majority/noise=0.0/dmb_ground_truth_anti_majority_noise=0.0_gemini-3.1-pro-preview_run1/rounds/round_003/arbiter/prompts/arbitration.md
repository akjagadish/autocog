# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Averaging Model: Decision-makers evaluate options not by summing the evidence (additive) or relying on a single best cue (lexicographic), but by averaging the subjective validities of all positive features present in an option. This mechanism naturally accounts for the 'dilution effect', where adding a positive but low-validity feature to an option can actually decrease its overall subjective value and choice probability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities to subjective weights
    w = val ** gamma
    
    # Count positive features
    k_a = np.sum(a)
    k_b = np.sum(b)
    
    # Calculate average validity of positive features (0 if no positive features)
    score_a = np.sum(w * a) / k_a if k_a > 0 else 0.0
    score_b = np.sum(w * b) / k_b if k_b > 0 else 0.0
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
Decision-makers integrate all available features to evaluate options, weighting each feature according to its subjective importance. This Weighted Additive (WADD) strategy computes a compensatory score for each option by summing the feature values multiplied by their normalized subjective weights. The subjective weights are modeled as the provided feature validities raised to a power 'gamma', which allows the model to smoothly interpolate between Tallying (gamma = 0, equal weights) and Take The Best (large gamma, where the highest validity cue dominates). Choice probabilities are then generated via a softmax function over the options' scores, subject to random lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute subjective weights by raising validities to the power of gamma
    w = val ** gamma
    # Normalize weights to decouple gamma (cue dominance) from beta (choice determinism)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Calculate Weighted Additive (WADD) scores for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.6, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Averaging model from the Weighted Additive (WADD) model, we exploit the 'dilution effect' unique to the Averaging model. In the Averaging model, the subjective value of an option is the average validity of its positive features. This leads to a counterintuitive prediction: adding a positive feature with a low validity to an option that already has high-validity features will *decrease* its overall score. In contrast, WADD computes a sum, meaning that adding any positive feature strictly increases the option's score. We design trials where Option A has a small number of high-validity features, and Option B has the exact same high-validity features PLUS several additional low-validity features. WADD will universally prefer Option B (more evidence), while the Averaging model will prefer Option A (higher average evidence quality).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
In trials where Option A's positive features are a strict subset of Option B's positive features, the WADD model will strictly prefer Option B because it has all the evidence of Option A plus additional evidence (higher sum). Conversely, the Averaging model will prefer Option A because the additional features in Option B are of lower validity, thereby diluting the average subjective value. Measuring the proportion of times Option A is chosen on these specific subset trials provides a stark contrast: the Averaging model predicts a value near 1, whereas WADD predicts a value near 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Option A's features are a strict subset of Option B's features
    is_subset = np.all(a_ratings <= b_ratings, axis=1) & (np.sum(a_ratings, axis=1) < np.sum(b_ratings, axis=1))
    
    subset_data = data[is_subset]
    if len(subset_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen
    return float(np.mean(subset_data['response'] == 0))

### RESULTS
- Predicted under pi_5 (simulated): 0.7136 (var=0.0101)
- Predicted under pi_3 (simulated): 0.4372 (var=0.0121)
- Observed on real data: 0.8678 (var=0.0153)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from the Averaging model, this design systematically pits the sum of evidence against the average quality of evidence across options with different numbers of positive features. In each trial, Option A possesses fewer but higher-validity features, whereas Option B possesses a larger number of lower-validity features. The WADD model, which integrates evidence additively, consistently predicts a preference for Option B because the total sum of subjective weights is greater. Conversely, the Averaging model, which computes the mean validity of positive features, consistently predicts a strong preference for Option A because the inclusion of lower-validity features in Option B dilutes its average value. By avoiding the simple 'A is a subset of B' dilution effect explored previously and instead contrasting entirely distinct feature sets, we provide a robust test of additive integration versus averaging.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the difference in the probability of choosing Option B when it has 3 features versus when it has 2 features. Under the Averaging model, adding a lower-validity feature dilutes Option B's average score, strictly decreasing its choice probability (a negative difference). Under the Weighted Additive (WADD) model, an extra feature increases Option B's total sum of evidence, which either increases its choice probability or decreases it much less severely than Averaging (resulting in a more positive or less negative difference).

Source:
def metric(data: pd.DataFrame) -> float:
    b_sum = data['option_b_ratings'].apply(sum)
    p_b_3 = data.loc[b_sum == 3, 'response'].mean()
    p_b_2 = data.loc[b_sum == 2, 'response'].mean()
    return float(p_b_3 - p_b_2)

### RESULTS
- Predicted under pi_5 (simulated): -0.0631 (var=0.0065)
- Predicted under pi_3 (simulated): -0.1687 (var=0.0254)
- Observed on real data: -0.1200 (var=0.0258)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    ttb_winner = np.zeros(len(data))
    
    for i in range(len(data)):
        winner = -1
        for j in range(5):
            if diff[i, j] > 0:
                winner = 0
                break
            elif diff[i, j] < 0:
                winner = 1
                break
        ttb_winner[i] = winner
        
    return float(np.mean(data['response'].values == ttb_winner))
```

**Observed (real) value:** 0.7675 (var=0.0218)
**Predicted under pi_5:** 0.7315 (var=0.0170)
**Predicted under pi_3:** 0.7538 (var=0.0288)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    strict_mask = a_wins != b_wins
    if not np.any(strict_mask):
        return 0.5
        
    tally_preds = (b_wins > a_wins).astype(int)
    responses = np.array(data['response'].tolist())
    
    match = (tally_preds[strict_mask] == responses[strict_mask])
    return float(np.mean(match))
```

**Observed (real) value:** 0.2552 (var=0.0312)
**Predicted under pi_5:** 0.2874 (var=0.0114)
**Predicted under pi_3:** 0.3233 (var=0.0094)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_choice = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice != -1:
            matches.append(1 if resp == ttb_choice else 0)
            
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0095)
**Predicted under pi_5:** 0.7281 (var=0.0104)
**Predicted under pi_3:** 0.7373 (var=0.0120)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Focus on diagnostic trials where one option has more positive features than the other.
    # In this specific design, these are exactly the trials where the single highest-validity
    # discriminating cue favors the option with FEWER total features.
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_filt = sum_a[mask]
    sum_b_filt = sum_b[mask]
    
    # Determine which option has the greater number of positive features
    # 0 for A, 1 for B
    compensatory_choice = (sum_b_filt > sum_a_filt).astype(int)
    
    # Calculate the proportion of choices aligning with the compensatory (WADD-like) option
    match = (filtered_data['response'] == compensatory_choice).mean()
    return float(match)

```

**Observed (real) value:** 0.2888 (var=0.0207)
**Predicted under pi_5:** 0.2784 (var=0.0094)
**Predicted under pi_3:** 0.3234 (var=0.0244)

### Experiment 7
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # The validities are strictly decreasing from cue 0 to cue 3.
    # So the Take-The-Best (TTB) choice is simply determined by the first cue 
    # (from index 0 to 3) where the two options differ.
    def get_ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(4):
            if a[i] > b[i]: return 0
            if b[i] > a[i]: return 1
        return 0
        
    data['ttb_choice'] = data.apply(get_ttb_choice, axis=1)
    data['is_ttb'] = (data['response'] == data['ttb_choice']).astype(float)
    
    # Create a hashable string representation of the trial pair to group by
    data['trial_str'] = data.apply(lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])), axis=1)
    
    # Calculate the proportion of TTB-consistent choices for each unique trial type
    trial_means = data.groupby('trial_str')['is_ttb'].mean()
    
    # Return the standard deviation of these proportions across the 10 trial types
    return float(trial_means.std())
```

**Observed (real) value:** 0.3309 (var=0.0110)
**Predicted under pi_5:** 0.0660 (var=0.0014)
**Predicted under pi_3:** 0.0645 (var=0.0053)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_cues'] = data['option_a_ratings'].apply(sum)
    data['b_cues'] = data['option_b_ratings'].apply(sum)
    
    def ttb_favors_a(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]: return True
            if b[i] > a[i]: return False
        return False

    data['ttb_a'] = data.apply(ttb_favors_a, axis=1)
    
    df_conflict = data[data['ttb_a']].copy()
    
    high_mask = df_conflict['b_cues'] >= 3
    low_mask = df_conflict['b_cues'] <= 2
    
    p_high = df_conflict.loc[high_mask, 'response'].mean()
    p_low = df_conflict.loc[low_mask, 'response'].mean()
    
    if pd.isna(p_high): p_high = 0.5
    if pd.isna(p_low): p_low = 0.5
    
    return float(p_high - p_low)
```

**Observed (real) value:** -0.1823 (var=0.0238)
**Predicted under pi_5:** -0.0304 (var=0.0102)
**Predicted under pi_3:** -0.0536 (var=0.0101)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the eight experiments, neither the Averaging model (THEORY 1) nor the Weighted Additive model (THEORY 2) fully captures the decision-making behavior. THEORY 1 successfully predicts the 'dilution effect' in Experiment 1 where THEORY 2 fails, and it aligns better with Experiments 4 and 6. However, both models completely fail to capture the large variance across trial types in Experiment 7 and the strong negative effect in Experiment 8. THEORY 2 (WADD) completely misses the subset dilution effect in Experiment 1. Because both theories fail on the later experiments and WADD struggles with key phenomena like dilution, WADD should be replaced with a fundamentally different approach.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory based on a heuristic or sequential feature-comparison process rather than holistic integration. For example, a 'Lexicographic with Fallback' (or Take-The-Best mixed with Tallying) model, where subjects compare options feature-by-feature in order of validity. If the difference on the most valid features is not decisive enough (or subject to a probabilistic threshold), the decision-maker falls back to a simple tally (counting) or averaging of remaining features. This could explain both the strong adherence to high-validity cues in some trials and the dilution/tallying-like effects when lower-validity cues are added, bridging the gap seen in Experiments 7 and 8."
}
```

## Usage

```json
{
  "prompt_token_count": 7951,
  "candidates_token_count": 332,
  "total_token_count": 9057
}
```
