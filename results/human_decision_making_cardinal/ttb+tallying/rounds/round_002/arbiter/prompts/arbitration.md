# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Dynamic Strategy Selection (Threshold Model): Decision-makers dynamically select between a compensatory Weighted Additive (WADD) strategy and a non-compensatory Tallying strategy on a trial-by-trial basis. The choice of strategy depends on the maximum cardinal difference between the options across all features. If this difference exceeds a certain threshold, the decision-maker is more likely to use WADD to account for the large magnitude; otherwise, they default to the simpler Tallying heuristic. This is modeled as a probabilistic mixture where the weight of WADD is a soft step function (sigmoid) of the maximum feature difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD score computation
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a_wadd = np.sum(a * val) / sum_val
        score_b_wadd = np.sum(b * val) / sum_val
    else:
        score_a_wadd = np.sum(a)
        score_b_wadd = np.sum(b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # Tallying score computation
    score_a_tally = float(np.sum(a > b))
    score_b_tally = float(np.sum(b > a))
    scores_tally = np.array([score_a_tally, score_b_tally])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    threshold = float(parameters["threshold"])
    steepness = float(parameters["steepness"])
    
    # Dynamic strategy selection based on max cardinal difference
    max_diff = np.max(np.abs(a - b))
    w_wadd = 1.0 / (1.0 + np.exp(-steepness * (max_diff - threshold)))
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[7, 7, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 2: A=[6, 6, 6, 6, 6]  B=[7, 7, 5, 5, 5]
  trial 3: A=[8, 8, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 4: A=[6, 6, 6, 6, 6]  B=[8, 8, 5, 5, 5]
  trial 5: A=[10, 10, 5, 5, 5]  B=[6, 6, 6, 6, 6]
  trial 6: A=[6, 6, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  trial 7: A=[10, 10, 2, 2, 2]  B=[2, 2, 3, 3, 3]
  trial 8: A=[2, 2, 3, 3, 3]  B=[10, 10, 2, 2, 2]
  trial 9: A=[9, 5, 5, 5, 5]  B=[1, 6, 6, 6, 6]
  trial 10: A=[1, 6, 6, 6, 6]  B=[9, 5, 5, 5, 5]

**Rationale:** To dissociate the purely compensatory Weighted Additive (WADD) theory from the Dynamic Strategy Selection (Threshold) model, we manipulate the maximum cardinal difference ('max_diff') between options across features. The Threshold model predicts that subjects will use Tallying when the maximum feature difference is small (falling below their threshold) and switch to WADD only when a feature difference is large enough to cross the threshold. In contrast, the pure WADD model integrates validity-weighted cardinal differences on every trial, regardless of the maximum difference. We construct a 5-feature design where WADD and Tallying consistently make opposing predictions: one option wins on the two most valid cues, while the other option wins on the three less valid cues. By parametrically varying the magnitude of the differences (max_diff = 1, 2, 4, and 8), we can observe whether subjects' choices flip from Tallying to WADD as the difference grows (supporting the Threshold model) or remain consistent with WADD throughout (supporting the advocated pure WADD theory).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
The experimental design contrasts WADD and Tallying by having them make opposing predictions on every trial. The Threshold model predicts that subjects will use Tallying (WADD-inconsistent) on trials with small maximum feature differences (max_diff <= 2) and switch to WADD (WADD-consistent) on trials with large maximum feature differences (max_diff >= 4). The pure WADD theory predicts WADD-consistent choices across all trials regardless of max_diff. Therefore, the difference in the proportion of WADD-consistent choices between large and small max_diff trials will be near zero for the pure WADD model, but significantly positive for the Threshold model.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # WADD consistently predicts the option with the higher rating on the first feature
    wadd_predicts_b = (a_ratings[:, 0] < b_ratings[:, 0]).astype(int)
    wadd_consistent = (data['response'].values == wadd_predicts_b).astype(float)
    
    # Calculate the maximum cardinal difference across features for each trial
    max_diff = np.max(np.abs(a_ratings - b_ratings), axis=1)
    
    # Define large and small max_diff trials based on the experimental design
    large_diff_mask = max_diff >= 4
    small_diff_mask = max_diff <= 2
    
    if np.sum(large_diff_mask) == 0 or np.sum(small_diff_mask) == 0:
        return 0.0
        
    large_diff_wadd_prop = np.mean(wadd_consistent[large_diff_mask])
    small_diff_wadd_prop = np.mean(wadd_consistent[small_diff_mask])
    
    return float(large_diff_wadd_prop - small_diff_wadd_prop)

### RESULTS
- Predicted under pi_3 (simulated): 0.1156 (var=0.0072)
- Predicted under pi_4 (simulated): 0.3141 (var=0.0617)
- Observed on real data: 0.0368 (var=0.0185)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[5, 5, 4, 4, 4]  B=[4, 4, 5, 5, 5]
  trial 2: A=[6, 6, 4, 4, 4]  B=[4, 4, 6, 6, 6]
  trial 3: A=[7, 7, 3, 3, 3]  B=[3, 3, 7, 7, 7]
  trial 4: A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]
  trial 5: A=[9, 9, 1, 1, 1]  B=[1, 1, 9, 9, 9]
  trial 6: A=[10, 10, 0, 0, 0]  B=[0, 0, 10, 10, 10]
  trial 7: A=[5, 5, 5, 4, 4]  B=[4, 4, 4, 5, 5]
  trial 8: A=[10, 10, 10, 0, 0]  B=[0, 0, 0, 10, 10]

**Rationale:** To quantitatively dissociate the Dynamic Strategy Selection (Threshold) model from the pure Weighted Additive (WADD) model, we construct a 5-feature design where the two models make diverging predictions depending on the maximum cardinal difference between the options. We design conflict trials where Option A wins on the two most valid features (sum of validities = 1.9), while Option B wins on the three least valid features (sum of validities = 1.6). Because WADD integrates validity-weighted magnitudes, it consistently prefers Option A on these trials regardless of the scale of the difference. Tallying, conversely, simply counts feature wins and consistently prefers Option B (3 wins to 2). By parametrically varying the magnitude of the difference (max_diff from 1 to 10), we can test the Threshold model's prediction: subjects will prefer Option B (Tallying) when the difference is small and below their threshold, but switch to Option A (WADD) when the difference is large. The pure WADD model predicts a consistent preference for Option A across all magnitudes.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the proportion of Option B choices between trials with a small maximum feature difference (<= 2) and trials with a large maximum feature difference (>= 8), restricting to trials where the Tallying heuristic favors Option B. The advocated Dynamic Strategy Selection (Threshold) model predicts a strong preference for B on small-difference trials (due to Tallying) and a strong preference for A on large-difference trials (due to WADD), yielding a large positive contrast. The pure WADD model consistently prefers A on both trial types, and while its choice probabilities may shift slightly due to the magnitude of the difference affecting softmax noise, the contrast will be substantially smaller than in the Threshold model.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    small_diff_b = []
    large_diff_b = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        max_diff = np.max(np.abs(a - b))
        tally_b = np.sum(b > a) > np.sum(a > b)
        
        if tally_b:
            if max_diff <= 2:
                small_diff_b.append(row['response'])
            elif max_diff >= 8:
                large_diff_b.append(row['response'])
                
    m_small = float(np.mean(small_diff_b)) if small_diff_b else 0.0
    m_large = float(np.mean(large_diff_b)) if large_diff_b else 0.0
    
    return m_small - m_large


### RESULTS
- Predicted under pi_3 (simulated): 0.1417 (var=0.0182)
- Predicted under pi_4 (simulated): 0.5850 (var=0.0597)
- Observed on real data: -0.0571 (var=0.0222)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[4, 1, 1, 1]  B=[3, 2, 2, 2]
  A=[2, 3, 3, 3]  B=[3, 1, 1, 1]
  A=[5, 5, 1, 1]  B=[5, 4, 2, 2]
  A=[3, 2, 4, 4]  B=[3, 3, 1, 1]
  A=[4, 2, 1, 1]  B=[3, 3, 2, 2]
  A=[1, 4, 4, 4]  B=[2, 2, 2, 2]
  A=[5, 4, 1, 1]  B=[5, 3, 2, 2]
  A=[2, 2, 4, 4]  B=[2, 3, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = None
        # Features are implicitly ordered by validity (highest to lowest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_match += 1
            total += 1
    return ttb_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.3308 (var=0.0607)
**Predicted under pi_3:** 0.1652 (var=0.0103)
**Predicted under pi_4:** 0.1710 (var=0.0109)

### Experiment 4
**Design**
  A=[5, 2, 2, 2, 2]  B=[2, 5, 5, 5, 2]
  A=[2, 5, 5, 5, 2]  B=[5, 2, 2, 2, 2]
  A=[5, 5, 2, 2, 2]  B=[5, 2, 5, 5, 5]
  A=[4, 4, 4, 2, 2]  B=[4, 4, 2, 5, 5]
  A=[3, 2, 5, 5, 5]  B=[3, 5, 2, 2, 2]
  A=[4, 3, 3, 3, 3]  B=[3, 4, 4, 4, 4]
  A=[1, 5, 1, 1, 1]  B=[1, 1, 5, 5, 5]
  A=[5, 1, 5, 5, 5]  B=[1, 5, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def tally_predicts(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(x > y for x, y in zip(a, b))
        b_wins = sum(y > x for x, y in zip(a, b))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        return -1
    predictions = data.apply(tally_predicts, axis=1)
    return float(np.mean(data['response'] == predictions))
```

**Observed (real) value:** 0.7270 (var=0.0429)
**Predicted under pi_3:** 0.8744 (var=0.0071)
**Predicted under pi_4:** 0.8544 (var=0.0084)

### Experiment 5
**Design**
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 10]
  A=[5, 5, 0, 0]  B=[4, 4, 10, 10]
  A=[0, 0, 10, 10]  B=[2, 2, 0, 0]
  A=[10, 10, 0, 0]  B=[9, 9, 1, 1]
  A=[1, 1, 1, 1]  B=[4, 0, 0, 0]
  A=[0, 2, 2, 2]  B=[10, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # trial 1: A=[10, 0, 0, 0]  B=[0, 1, 1, 1] -> WADD prefers A (0), Tallying prefers B
        if a[0] == 10 and a[1] == 0 and b[0] == 0:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 2: A=[0, 8, 0, 0]  B=[1, 0, 1, 1] -> WADD prefers A (0), Tallying prefers B
        elif a[0] == 0 and a[1] == 8 and b[0] == 1:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 3: A=[1, 1, 1, 0]  B=[0, 0, 0, 10] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 0 and b[3] == 10:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 7: A=[1, 1, 1, 1]  B=[4, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 1 and b[0] == 4:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 8: A=[0, 2, 2, 2]  B=[10, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 0 and a[1] == 2 and b[0] == 10:
            wadd_consistent += (resp == 1)
            total += 1
            
    return wadd_consistent / total if total > 0 else 0.5
```

**Observed (real) value:** 0.6446 (var=0.0854)
**Predicted under pi_3:** 0.8840 (var=0.0047)
**Predicted under pi_4:** 0.6873 (var=0.0203)

### Experiment 6
**Design**
  A=[5, 5, 5, 5, 5]  B=[10, 4, 4, 4, 4]
  A=[9, 9, 4, 4, 4]  B=[5, 5, 5, 5, 5]
  A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  A=[8, 8, 2, 2, 5]  B=[2, 2, 8, 8, 5]
  A=[3, 3, 3, 3, 3]  B=[2, 2, 2, 2, 10]
  A=[1, 9, 9, 9, 9]  B=[10, 8, 8, 8, 8]
  A=[7, 2, 7, 2, 7]  B=[6, 8, 6, 8, 6]
  A=[5, 6, 5, 6, 5]  B=[6, 5, 6, 5, 6]
  A=[4, 8, 4, 8, 4]  B=[5, 2, 5, 2, 5]
  A=[8, 8, 8, 0, 0]  B=[9, 9, 9, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    responses = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_diff = np.sum(a > b) - np.sum(b > a)
        wadd_diff = np.sum(a * v) - np.sum(b * v)
        
        if tally_diff > 0 and wadd_diff < 0:
            responses.append(row['response'])
        elif tally_diff < 0 and wadd_diff > 0:
            responses.append(1 - row['response'])
            
    if len(responses) == 0:
        return 0.5
    return float(np.mean(responses))
```

**Observed (real) value:** n/a (var=n/a)
**Predicted under pi_3:** 0.8067 (var=0.0051)
**Predicted under pi_4:** 0.5704 (var=0.0814)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data across Experiments 1 and 2 severely contradict the Dynamic Strategy Selection (Threshold) theory (pi_4). In Experiment 1, the observed metric (0.0368) shows almost no shift towards WADD on large max_diff trials, whereas pi_4 predicted a strong shift (0.3141). In Experiment 2, the observed metric is slightly negative (-0.0571), completely invalidating pi_4's prediction of a massive positive shift (0.5850). While the pure WADD model (pi_3) is closer to the data in these experiments, it overpredicts WADD-consistency in Experiment 5 (predicting 0.8840 vs. observed 0.6446), suggesting that large cardinal differences do not dominate choice as much as a linear WADD model assumes.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Diminishing Returns WADD' (or Non-linear WADD) theory. Instead of dynamically switching between Tallying and WADD based on a threshold, subjects evaluate options by applying a concave utility transformation (e.g., a power function with an exponent < 1) to the cardinal feature values before computing the validity-weighted sum. This naturally compresses large cardinal differences, causing the model to behave somewhat like a hybrid of WADD and Tallying without relying on an arbitrary max_diff threshold. This should better capture the reduced impact of large magnitudes seen in Experiment 5 while avoiding the incorrect threshold-based predictions in Experiments 1 and 2."
}
```

## Usage

```json
{
  "prompt_token_count": 7388,
  "candidates_token_count": 361,
  "total_token_count": 8740
}
```
