# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Threshold-Gated Compensatory Model: Subjects sort cues by their validities and evaluate the numerical gaps between adjacent cues in this sorted order. If the validities are closely clustered, subjects integrate them additively, functioning as a weighted additive model. However, if a gap exceeds a subjective threshold, all subsequent lower-ranked cues are heavily discounted or completely ignored, triggering a shift toward a non-compensatory, semi-lexicographic heuristic. The threshold is sensitive enough to trigger discrete strategy shifts frequently.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gap_threshold = float(parameters["gap_threshold"])
    discount_factor = float(parameters["discount_factor"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort validities descending to evaluate gaps
    order = np.argsort(-validities)
    sorted_vals = validities[order]
    
    w_sorted = np.zeros_like(sorted_vals)
    if len(sorted_vals) > 0:
        w_sorted[0] = sorted_vals[0]
        current_discount = 1.0
        
        for i in range(1, len(sorted_vals)):
            # If the gap between adjacent sorted validities exceeds the threshold, apply discount
            if (sorted_vals[i-1] - sorted_vals[i]) > gap_threshold:
                current_discount *= discount_factor
            w_sorted[i] = sorted_vals[i] * current_discount
            
    # Map subjective weights back to original feature order
    w = np.zeros_like(validities)
    w[order] = w_sorted
    
    # Option scores are the sum of feature values weighted by the threshold-gated subjective weights
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=7):** [0.92, 0.91, 0.9, 0.85, 0.85, 0.85, 0.85]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Threshold-Gated Compensatory Model from the Non-linear Subjective Weighting (NLSW) Model by exploiting the difference between a discrete threshold-triggered discount and continuous non-linear scaling. We use a 7-feature design with validities clustered into a top tier [0.92, 0.91, 0.90] and a bottom tier [0.85, 0.85, 0.85, 0.85]. The critical Trial 1 pits Option A (possessing features 2 and 3) against Option B (possessing feature 1 and all four bottom-tier features). Under the NLSW model, Option B mathematically must win across the entire parameter space: for small gamma, the sheer quantity of features in Option B dominates; for large gamma, Option B's single highest feature (0.92) strictly dominates Option A's features. There is no gamma exponent where features 2 and 3 can simultaneously overcome feature 1 AND the sum of features 4-7. In stark contrast, the Threshold-Gated model identifies the gap between 0.90 and 0.85 and applies a discrete discount to all bottom-tier features. Consequently, Option A's un-discounted sum (1.81) easily outweighs Option B's feature 1 (0.92) plus the heavily suppressed bottom-tier features. Thus, the Advocated model uniquely predicts a preference for Option A, a choice structurally impossible under the Competing model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Trial 1 is specifically designed to completely dissociate the two theories. Option A possesses features 2 and 3, while Option B possesses feature 1 and features 4 through 7. Under the Non-linear Subjective Weighting model, Option B strictly dominates Option A across all valid combinations of gamma: at low gamma, Option B's sheer quantity of features wins, and at high gamma, Option B's top feature (0.92) dominates Option A's best feature (0.91). Therefore, NLSW predicts that choosing A is mathematically impossible except through lapse noise. In contrast, the Threshold-Gated Compensatory Model can detect the gap between the top tier (0.90) and bottom tier (0.85), applying a heavy discount to features 4-7. This discrete suppression allows Option A's un-discounted features to outscore Option B. By measuring the choice rate for Option A on Trial 1, we cleanly isolate this structural divergence.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Isolate Trial 1 where Option A has features 2 & 3, and Option B has feature 1 plus all bottom-tier features
    mask = (a_tuples == (0, 1, 1, 0, 0, 0, 0)) & (b_tuples == (1, 0, 0, 1, 1, 1, 1))
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.0
        
    # Return the proportion of times Option A was chosen
    return float((subset['response'] == 0).mean())

### RESULTS
- Predicted under pi_6 (simulated): 0.2233 (var=0.0293)
- Predicted under pi_4 (simulated): 0.1200 (var=0.0084)
- Observed on real data: 0.3300 (var=0.0469)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** This design quantitatively dissociates the Non-linear Subjective Weighting Model (Advocated) from the Threshold-Gated Compensatory Model (Competing) by exploiting the difference between a continuous power-function transformation and discrete, threshold-triggered discounting. The validities are spaced evenly with uniform gaps of 0.05. Because the Threshold-Gated model relies on gap thresholds, uniform gaps force it into an all-or-nothing regime: it must either apply no discounts (collapsing into a pure Weighted Additive rule) or apply a heavy, recursive discount at every single step (collapsing toward Take The Best). It cannot smoothly amplify the top cues while maintaining moderate weights for the middle cues. In contrast, the Non-linear Subjective Weighting model uses a continuous exponent (gamma), allowing it to naturally generate intermediate compensatory trade-offs. By pitting the highest validity cue against varying coalitions of middle- and lower-validity cues across the trials, the Advocated model predicts a smooth, graded shift in choice probabilities that the Competing model's rigid step-function mechanics cannot replicate without severe quantitative misfit.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference between the rate of choosing Option A in Trial 1 and the rate of choosing Option B in Trial 3. In the Competing (Threshold-Gated) model, because all validity gaps are exactly 0.05, subjects fall into two discrete categories: pure linear (WADD) or heavily discounted (TTB-like). For both of these sub-populations, the mathematical difference in option values is nearly perfectly symmetric between Trial 1 and Trial 3, meaning `a_t1` and `b_t3` will be identical (either both high for TTB, or both low for WADD). Thus, the Competing model predicts this difference will be exactly 0 with near-zero variance. In contrast, the Advocated (Non-linear Subjective Weighting) model applies a continuous power function. Because Trial 3 has a larger base difference in validities than Trial 1, the non-linear transformation breaks the symmetry, leading to a consistently positive difference (Option A wins Trial 3 more strongly than Option B wins Trial 1 at high gammas). This provides a highly stable, low-variance contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (1, 0, 0, 0, 0, 1))
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(int(v) for v in x) == (0, 1, 1, 1, 1, 0))
    
    a_t1 = float((data.loc[is_t1, 'response'] == 0).mean()) if is_t1.any() else 0.5
    b_t3 = float((data.loc[is_t3, 'response'] == 1).mean()) if is_t3.any() else 0.5
    
    return a_t1 - b_t3

### RESULTS
- Predicted under pi_6 (simulated): -0.0167 (var=0.0087)
- Predicted under pi_4 (simulated): 0.0483 (var=0.0090)
- Observed on real data: 0.0670 (var=0.0234)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        
        # In the compensatory trials designed for this experiment,
        # the option favored by TTB always has fewer total positive cues
        # than the option favored by WADD.
        if sum_a != sum_b:
            ttb_choice = 0 if sum_a < sum_b else 1
            ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.4175 (var=0.0635)
**Predicted under pi_6:** 0.4000 (var=0.0781)
**Predicted under pi_4:** 0.5029 (var=0.0626)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None and response == ttb_choice:
            ttb_matches += 1
        total += 1
        
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4317 (var=0.0770)
**Predicted under pi_6:** 0.3485 (var=0.0545)
**Predicted under pi_4:** 0.4823 (var=0.0320)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Define the critical trials where PHS and WADD strongly diverge.
    # In Trial 1, Option A has the single highest validity cue and more total cues,
    # so PHS (TTB + Tallying) strongly predicts Option A (response 0).
    # Option B has multiple high-validity cues that outweigh A's cues in a weighted sum,
    # so WADD strongly predicts Option B (response 1).
    # Trial 2 is the exact symmetric swap.
    t1_A = (1, 0, 0, 0, 1, 1, 1)
    t2_A = (0, 1, 1, 1, 0, 0, 0)
    
    is_t1 = data['option_a_ratings'].apply(tuple) == t1_A
    is_t2 = data['option_a_ratings'].apply(tuple) == t2_A
    
    match_phs = 0
    match_phs += (data.loc[is_t1, 'response'] == 0).sum()
    match_phs += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    if total == 0:
        return 0.5
    return float(match_phs / total)

```

**Observed (real) value:** 0.2317 (var=0.0651)
**Predicted under pi_6:** 0.1321 (var=0.0099)
**Predicted under pi_4:** 0.2029 (var=0.0250)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_1 = data['option_b_ratings'].apply(lambda x: x[1])
    critical_mask = (a_0 == 1) & (b_1 == 1)
    if critical_mask.sum() == 0:
        return 0.5
    return float(data.loc[critical_mask, 'response'].mean())
```

**Observed (real) value:** 0.8158 (var=0.0450)
**Predicted under pi_6:** 0.7712 (var=0.0431)
**Predicted under pi_4:** 0.7850 (var=0.0505)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a[0] == 1:
            other = b
        else:
            other = a
            
        if other == (0, 1, 1, 1, 0):
            return 'strong'
        elif other == (0, 0, 0, 1, 1):
            return 'weak'
        return 'other'

    data['trial_type'] = data.apply(get_trial_type, axis=1)
    
    def is_ttb_choice(row):
        a_has_cue = row['option_a_ratings'][0] == 1
        return 1 if (row['response'] == 0 and a_has_cue) or (row['response'] == 1 and not a_has_cue) else 0

    data['ttb_choice'] = data.apply(is_ttb_choice, axis=1)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        strong_rate = subj_df[subj_df['trial_type'] == 'strong']['ttb_choice'].mean()
        weak_rate = subj_df[subj_df['trial_type'] == 'weak']['ttb_choice'].mean()
        if pd.notna(strong_rate) and pd.notna(weak_rate):
            subj_diffs.append(weak_rate - strong_rate)
            
    return float(np.mean(subj_diffs)) if subj_diffs else 0.0
```

**Observed (real) value:** 0.4767 (var=0.1713)
**Predicted under pi_6:** 0.1058 (var=0.0307)
**Predicted under pi_4:** 0.2833 (var=0.0876)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    overrides = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        sum_a = sum(a)
        sum_b = sum(b)
        tally_winner = 0 if sum_a > sum_b else (1 if sum_b > sum_a else None)
        
        if ttb_winner is not None and tally_winner is not None and ttb_winner == tally_winner:
            chosen = row['response']
            if chosen != ttb_winner:
                overrides.append(1)
            else:
                overrides.append(0)
                
    if not overrides:
        return 0.0
    return float(np.mean(overrides))
```

**Observed (real) value:** 0.4729 (var=0.0463)
**Predicted under pi_6:** 0.2744 (var=0.0489)
**Predicted under pi_4:** 0.3604 (var=0.0143)

### Experiment 9
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 3: Cue 2+3 (A) vs Cue 1 (B)
    is_t3 = (a_keys == (0, 1, 1, 0, 0)) & (b_keys == (1, 0, 0, 0, 0))
    # Trial 1: Cue 2 (A) vs Cue 3+4 (B)
    is_t1 = (a_keys == (0, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 0))
    
    # Probability of choosing the single higher-ranked cue in each trial
    p_cue1_t3 = (data.loc[is_t3, 'response'] == 1).mean()
    p_cue2_t1 = (data.loc[is_t1, 'response'] == 0).mean()
    
    if pd.isna(p_cue1_t3) or pd.isna(p_cue2_t1):
        return 0.0
        
    return float(p_cue1_t3 - p_cue2_t1)
```

**Observed (real) value:** 0.1325 (var=0.1076)
**Predicted under pi_6:** 0.4775 (var=0.0861)
**Predicted under pi_4:** 0.3100 (var=0.0608)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def subj_diff(df):
        t1_mask = df['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
        t2_mask = df['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 0))
        p_b_t1 = df[t1_mask]['response'].mean()
        p_b_t2 = df[t2_mask]['response'].mean()
        return p_b_t1 - p_b_t2
        
    diffs = data.groupby('subject_id').apply(subj_diff).dropna()
    return float(diffs.mean())
```

**Observed (real) value:** 0.5243 (var=0.1374)
**Predicted under pi_6:** 0.4100 (var=0.1272)
**Predicted under pi_4:** 0.5233 (var=0.0964)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 10 experiments, Theory 2 (pi_4: Non-linear Subjective Weighting) consistently outperforms Theory 1 (pi_6: Threshold-Gated Compensatory Model). pi_4 provides much more accurate predictions for Experiments 2, 4, 5, 6, 7, 8, 9, and 10. The gap-threshold mechanism of pi_6 falls short because it relies on a discrete cutoff that either heavily discounts or completely ignores lower-ranked cues, failing to capture the graded, continuous nature of subjective cue weighting observed in the data. However, while pi_4 is superior, it still struggles to fully capture the magnitude of effects in Experiments 7 and 9, suggesting that a single continuous power function might not perfectly capture the heterogeneity of subject strategies.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Probabilistic Strategy Mixture Model'. Instead of assuming a single integration mechanism (like gap-discounting or a non-linear power function), this theory posits that subjects maintain a repertoire of distinct decision strategies (e.g., pure Take-The-Best and pure Weighted Additive). On any given trial, a subject probabilistically selects which strategy to apply. The probability of selecting the non-compensatory strategy (TTB) over the compensatory one (WADD) could be an individual-level trait parameter, potentially modulated by the dispersion of the cue validities in the environment. This mixture approach can naturally produce the intermediate response rates and high between-subject variances observed in the data, offering a structurally different alternative to the continuous power transformation of Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 9607,
  "candidates_token_count": 363,
  "total_token_count": 10941
}
```
