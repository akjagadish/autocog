# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Soft Take-The-Best with Power-Law Decay: Decision-makers evaluate options by ranking features according to their subjective validity and integrating evidence across all cues. However, instead of the importance of cues decaying exponentially with their rank, it decays according to a power law. This 'fatter tail' enables a strong accumulation of evidence on multiple lower-ranked cues to compensate for a loss on top-ranked cues, capturing compensatory behavior in scenarios where many weak cues oppose a few strong ones.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Soft TTB expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(v)[::-1]
    
    alpha = float(parameters["alpha"])
    # Power-law decay based on rank (rank 1, 2, 3...)
    weights = 1.0 / ((np.arange(len(v)) + 1.0) ** alpha)
    
    # Binary wins on ordered features
    a_wins = (a[order] > b[order]).astype(float)
    b_wins = (b[order] > a[order]).astype(float)
    
    score_a = np.sum(a_wins * weights)
    score_b = np.sum(b_wins * weights)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.5]

**rating_max:** 1

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To dissociate Soft Take-The-Best (with Power-Law Decay) from Weighted Tallying, we exploit how they assign importance to features. Both models ignore the magnitude of feature differences and focus purely on 'wins'. However, Weighted Tallying weights each win linearly by the raw validity of the feature, whereas Soft TTB weights each win by a power-law transformation of the feature's rank (e.g., 1, 1/2^a, 1/3^a). By clustering the top validities very close to each other (e.g., 0.95, 0.90, 0.85, 0.80), Weighted Tallying treats them as nearly equal, meaning a combination of the 2nd and 3rd most valid features easily outweighs the 1st. In contrast, Soft TTB enforces a strict rank-based drop-off regardless of the raw validity values, meaning the 1st feature can still dominate the 2nd and 3rd combined. The trials pit a single high-rank win against multiple slightly lower-rank wins to strongly dissociate the predictions of the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the overall proportion of trials where the subject chose Option A. The experiment is specifically designed so that Option A always relies on a single or very few top-ranked cues, while Option B relies on a greater number of slightly lower-validity cues. Due to the linear versus power-law weighting of the two theories, Soft TTB will predominantly choose Option A, while Weighted Tallying will predominantly choose Option B. This simple proportion perfectly captures the opposing predictions.

Source:
def metric(data: pd.DataFrame) -> float:
    # In all designed trials, Option A has fewer wins but on higher-ranked cues,
    # whereas Option B has more wins but on slightly lower-validity cues.
    # Weighted Tallying treats the clustered validities as nearly linear,
    # so the sum of multiple lower-ranked wins (Option B) easily outweighs Option A.
    # Soft TTB uses a power-law decay on rank, allowing the top-ranked cue(s) of Option A to dominate.
    # Thus, Soft TTB predicts choosing A (response == 0) much more frequently than Weighted Tallying.
    return float((data['response'] == 0).mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.7006 (var=0.0367)
- Predicted under pi_4 (simulated): 0.1760 (var=0.0121)
- Observed on real data: 0.3400 (var=0.0733)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.85, 0.8, 0.75, 0.7]

**rating_max:** 1

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To dissociate Weighted Tallying from Soft Take-The-Best (with Power-Law Decay), we exploit the difference between linear validity weighting and rank-based power-law weighting. We use a set of 5 features with a gentle, linear decline in validities (0.90 down to 0.70). Weighted Tallying uses these exact validities, treating them as relatively comparable in importance. Soft TTB, however, ranks them and applies a power-law decay (e.g., 1, 1/2, 1/3), creating a steep drop-off in importance regardless of the raw validity values. We construct trials where one option wins on the 1st-ranked feature and the lowest-ranked feature, while the other wins on the 2nd and 3rd-ranked features. Weighted Tallying prefers the combination of the 2nd and 3rd features (0.85 + 0.80 = 1.65 > 0.90 + 0.70 = 1.60), whereas Soft TTB strongly prefers the 1st-ranked feature due to the heavy penalty on lower ranks. By using binary ratings, we isolate the weighting mechanism from magnitude effects.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
In the chosen experimental design, Weighted Tallying consistently prefers the option that wins on a larger number of slightly less valid features, whereas Soft Take-The-Best (with power-law decay) strongly prefers the option that wins on the single most valid feature (feature 1). By calculating the proportion of trials where the subject chooses the option that loses on feature 1, we perfectly separate the two theories: Weighted Tallying will yield a high value (close to 1), and Soft TTB will yield a low value (close to 0).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract the first feature (highest validity) for options A and B
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Weighted Tallying favors the combination of multiple lower-ranked features
    # over the single highest-ranked feature in this specific experimental design.
    # Therefore, WT predicts choosing the option that LOSES on feature 1.
    # If A > B on feature 1, B loses on feature 1, so WT predicts 1 (Option B).
    # If A < B on feature 1, A loses on feature 1, so WT predicts 0 (Option A).
    wt_predicted_response = (a_f1 > b_f1).astype(int)
    
    # Calculate the proportion of trials where the subject's choice aligns with WT
    match = (data['response'] == wt_predicted_response)
    return float(match.mean())

### RESULTS
- Predicted under pi_5 (simulated): 0.2817 (var=0.0536)
- Predicted under pi_4 (simulated): 0.7714 (var=0.0087)
- Observed on real data: 0.6080 (var=0.0943)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[5, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[2, 5, 2, 2]  B=[3, 1, 3, 3]
  A=[4, 4, 1, 5]  B=[5, 5, 2, 0]
  A=[0, 0, 5, 5]  B=[1, 1, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        if a[0] == 1:
            matches.append(r == 0)
        elif a[0] == 5:
            matches.append(r == 1)
        elif a[0] == 2:
            matches.append(r == 1)
        elif a[0] == 4:
            matches.append(r == 1)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7250 (var=0.0281)
**Predicted under pi_5:** 0.6097 (var=0.0228)
**Predicted under pi_4:** 0.8592 (var=0.0096)

### Experiment 4
**Design**
  A=[5, 5, 5, 5]  B=[0, 6, 6, 6]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 5, 5, 4]
  A=[8, 4, 4, 4]  B=[5, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[0, 4, 4, 4], B=[10, 0, 0, 0]
    # WADD prefers B (response=1) because 10*0.9 = 9.0 > 4*(0.8+0.6+0.5) = 7.6
    # Tallying prefers A (response=0) because A wins on 3 features vs B's 1
    
    # Identify trial 2 by checking the first element of option A
    is_trial_2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 4, 4, 4))
    trial_data = data[is_trial_2]
    
    if len(trial_data) == 0:
        return 0.5
        
    return float(trial_data['response'].mean())
```

**Observed (real) value:** 0.2600 (var=0.1924)
**Predicted under pi_5:** 0.6358 (var=0.1058)
**Predicted under pi_4:** 0.1537 (var=0.0208)

### Experiment 5
**Design**
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 10, 0, 0]  B=[0, 0, 1, 1]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 8]
  A=[9, 9, 0, 0]  B=[10, 0, 1, 1]

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
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            matches.append(1 if row['response'] == tally_pref else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7440 (var=0.0257)
**Predicted under pi_5:** 0.6803 (var=0.0220)
**Predicted under pi_4:** 0.8658 (var=0.0049)

### Experiment 6
**Design**
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[10, 10, 0, 0]  B=[0, 0, 2, 2]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[2, 2, 0, 0]  B=[0, 0, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    eqw_predicts_a = a_sums > b_sums
    eqw_predicts_b = b_sums > a_sums
    
    match_a = eqw_predicts_a & (data['response'] == 0)
    match_b = eqw_predicts_b & (data['response'] == 1)
    
    valid_trials = eqw_predicts_a | eqw_predicts_b
    if not valid_trials.any():
        return 0.5
        
    return float((match_a | match_b).sum() / valid_trials.sum())
```

**Observed (real) value:** 0.3680 (var=0.0278)
**Predicted under pi_5:** 0.3635 (var=0.0081)
**Predicted under pi_4:** 0.2812 (var=0.0030)

### Experiment 7
**Design**
  A=[1, 1, 4, 4, 4]  B=[4, 4, 1, 1, 1]
  A=[5, 5, 2, 2, 2]  B=[2, 2, 4, 4, 4]
  A=[4, 2, 4, 2, 3]  B=[2, 4, 2, 4, 3]
  A=[2, 3, 4, 4, 2]  B=[4, 3, 2, 2, 4]
  A=[4, 2, 3, 3, 3]  B=[2, 4, 3, 3, 3]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Trial 1: A=[1, 1, 4, 4, 4], B=[4, 4, 1, 1, 1]
    # Tallying prefers A (3 vs 2). Weighted Tallying prefers B (1.7 vs 1.5).
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 4)
    
    # Trial 2: A=[5, 5, 2, 2, 2], B=[2, 2, 4, 4, 4]
    # Tallying prefers B (3 vs 2). Weighted Tallying prefers A (1.7 vs 1.5).
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 2)
    
    # Proportion of times choosing B (response == 1) in Trial 1
    t1_data = data.loc[t1_mask, 'response']
    val1 = t1_data.mean() if len(t1_data) > 0 else 0.5
    
    # Proportion of times choosing A (response == 0) in Trial 2
    t2_data = data.loc[t2_mask, 'response']
    val2 = (1.0 - t2_data.mean()) if len(t2_data) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)
```

**Observed (real) value:** 0.4200 (var=0.2436)
**Predicted under pi_5:** 0.8221 (var=0.0280)
**Predicted under pi_4:** 0.7405 (var=0.0148)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where A=[1, 1, 0, 0, 0] and B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.5
    trial_1_data = data[is_trial_1]
    # Calculate the proportion of times Option A was chosen (response == 0)
    prob_a = (trial_1_data['response'] == 0).mean()
    return float(prob_a)
```

**Observed (real) value:** 0.2800 (var=0.2016)
**Predicted under pi_5:** 0.7274 (var=0.0701)
**Predicted under pi_4:** 0.7411 (var=0.0266)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither Soft Take-The-Best (Theory 1) nor Weighted Tallying (Theory 2) adequately captures the observed data across all experiments, particularly failing on Experiments 7 and 8. For instance, in Experiment 8, both theories predicted a strong preference for Option A (around 72-74%), but the observed data showed a strong preference for Option B (A chosen only 28% of the time). Option B wins on a greater number of less valid features (3 wins vs 2 wins). This implies that participants are likely ignoring the validities or weights of the features and simply counting the number of winning features (a pure Tallying heuristic). While Weighted Tallying (Theory 2) captures some of this by giving weight to lower-ranked features, it still over-values the top-ranked features compared to human participants. Theory 1 (Soft TTB) is too extreme in its reliance on top-ranked features. Therefore, a new theory is needed to replace Theory 1.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Pure Tallying' (or Equal Weight) heuristic theory. In this theory, decision-makers completely ignore the provided validities of the cues. Instead, they simply count the number of features on which Option A is strictly better than Option B, and vice-versa. The option with the higher total count of winning features is chosen. This unweighted counting mechanism will naturally explain the strong preference for options with more winning features (even if those features are of lower validity), accurately capturing the observed behavior in Experiments 1, 7, and 8 where both current weighted models fail."
}
```

## Usage

```json
{
  "prompt_token_count": 6977,
  "candidates_token_count": 373,
  "total_token_count": 8419
}
```
