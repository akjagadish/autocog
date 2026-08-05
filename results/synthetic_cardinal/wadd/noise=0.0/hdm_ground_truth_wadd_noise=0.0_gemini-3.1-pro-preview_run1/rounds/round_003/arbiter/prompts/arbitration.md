# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People evaluate options by taking a weighted sum of their feature values, where the weights correspond to the subjective validities of the cues (Weighted Additive rule). This allows for fully compensatory decision making, where many small advantages can outweigh a single large advantage, and is sensitive to the cardinal magnitude of feature differences, unlike simple counting (Tallying) or one-reason heuristics (Take The Best).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    if len(val) != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Compute the weighted sum of feature values for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    return int(np.argmax(probabilities))


## THEORY 2 — pi_5
Tallying (Majority of Confirming Dimensions) heuristic: People evaluate multi-attribute options by counting the number of features on which each option is strictly superior to the other. They ignore both the validities of the features and the cardinal magnitudes of the differences, choosing the option that wins on the most features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Count strict superiority
    tally_a = np.sum(a > b)
    tally_b = np.sum(b > a)
    scores = np.array([tally_a, tally_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[1, 5, 8, 8]  B=[10, 4, 7, 7]
  trial 2: A=[10, 10, 9, 9]  B=[0, 0, 10, 10]
  trial 3: A=[10, 0, 10, 0]  B=[0, 10, 0, 10]
  trial 4: A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 5: A=[4, 4, 4, 4]  B=[10, 0, 2, 2]
  trial 6: A=[10, 0, 2, 2]  B=[4, 4, 4, 4]
  trial 7: A=[5, 5, 5, 5]  B=[0, 10, 10, 10]
  trial 8: A=[10, 10, 10, 0]  B=[0, 0, 0, 10]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) rule from the Tallying heuristic. Tallying ignores both the validities of the features and the cardinal magnitudes of their values, simply counting the number of features on which an option is strictly superior. WADD, conversely, computes a validity-weighted sum of the cardinal feature values. The trials are constructed to include stark preference reversals (where one option wins on more features but the other has a massive magnitude advantage on a highly valid feature), as well as orthogonal indifference points. In some trials, Tallying predicts a tie (equal number of winning features) while WADD predicts a strong preference due to cardinal differences. In other trials, WADD predicts a tie (weighted sums are equal) but Tallying predicts a strong preference (one option wins on 3 out of 4 features).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1, Tallying strongly prefers Option A (which wins on 3 out of 4 features), whereas WADD strongly prefers Option B (due to its massive advantage on the most valid feature). In Trial 4, the preference is completely reversed: Tallying prefers Option B (wins on 3 features), while WADD prefers Option A (huge advantage on the highest validity feature). By computing the difference in the probability of choosing Option B between Trial 1 and Trial 4, we get a metric that is highly positive for WADD (high - low) and highly negative for Tallying (low - high).

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1: A=[1, 5, 8, 8]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    # Identify Trial 4: A=[10, 0, 0, 0]
    is_trial_4 = data['option_a_ratings'].apply(lambda x: x[0] == 10 and x[1] == 0 and x[2] == 0 and x[3] == 0)
    
    p_b_t1 = data.loc[is_trial_1, 'response'].mean()
    p_b_t4 = data.loc[is_trial_4, 'response'].mean()
    
    if pd.isna(p_b_t1): p_b_t1 = 0.5
    if pd.isna(p_b_t4): p_b_t4 = 0.5
    
    return float(p_b_t1 - p_b_t4)

### RESULTS
- Predicted under pi_3 (simulated): 0.8600 (var=0.2404)
- Predicted under pi_5 (simulated): -0.6250 (var=0.0434)
- Observed on real data: 0.8200 (var=0.3076)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[2, 6, 6, 6]  B=[10, 5, 5, 5]
  trial 2: A=[10, 4, 4, 4]  B=[2, 5, 5, 5]
  trial 3: A=[10, 10, 2, 2]  B=[2, 2, 4, 4]
  trial 4: A=[0, 6, 6, 6]  B=[10, 1, 1, 2]
  trial 5: A=[10, 1, 1, 2]  B=[0, 6, 6, 6]
  trial 6: A=[6, 6, 2, 2]  B=[4, 4, 10, 10]

**Rationale:** This design strictly dissociates Tallying (a sign-based, unweighted heuristic) from the Weighted Additive (WADD) rule (a magnitude-based, validity-weighted heuristic). To do this, we use 4 features with dispersed validities and a cardinal rating scale (0-10). The trials are carefully designed to capture three quantitative divergences: 1) Preference reversals, where one option wins on the majority of features by a small margin (favored by Tallying), but the other wins on a single high-validity feature by a massive margin (favored by WADD); 2) Tallying indifference, where options win on an equal number of features (leading Tallying to predict 50/50 guessing), but one option has a much larger cardinal advantage on its winning features (leading WADD to strongly prefer it); 3) WADD indifference, where the validity-weighted sums of the two options are exactly equal (causing WADD to predict 50/50 guessing), but one option wins on 3 out of 4 features (leading Tallying to strongly prefer it).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses strictly on the 'preference reversal' trials (Trials 1 and 2) where Tallying and WADD have opposite strict preferences. On Trial 1, Tallying prefers A (wins on 3 out of 4 features) while WADD strongly prefers B (wins on the most valid feature by a large margin). Trial 2 is the symmetric opposite. By calculating the proportion of Tallying-consistent choices on these specific trials, we expect a value near 1 for data generated by Tallying and a value near 0 for data generated by WADD, providing maximal discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Extract the first and second feature of option A to identify trial types
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    a_1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    # Trial 1: A=[2, 6, 6, 6], B=[10, 5, 5, 5] -> A[0] == 2
    # Tallying prefers A (response 0), WADD prefers B (response 1)
    mask_t1 = (a_0 == 2)
    
    # Trial 2: A=[10, 4, 4, 4], B=[2, 5, 5, 5] -> A[0] == 10 and A[1] == 4
    # Tallying prefers B (response 1), WADD prefers A (response 0)
    mask_t2 = (a_0 == 10) & (a_1 == 4)
    
    # Count choices consistent with Tallying
    t1_tally_choices = (data.loc[mask_t1, 'response'] == 0).sum()
    t2_tally_choices = (data.loc[mask_t2, 'response'] == 1).sum()
    
    total_relevant_trials = mask_t1.sum() + mask_t2.sum()
    
    if total_relevant_trials == 0:
        return 0.5
        
    return (t1_tally_choices + t2_tally_choices) / total_relevant_trials

### RESULTS
- Predicted under pi_3 (simulated): 0.2800 (var=0.2016)
- Predicted under pi_5 (simulated): 0.8387 (var=0.0117)
- Observed on real data: 0.2600 (var=0.1924)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[4, 2, 2, 2, 2]  B=[3, 4, 4, 4, 4]
  A=[2, 4, 4, 4, 4]  B=[4, 2, 2, 2, 2]
  A=[3, 4, 2, 2, 2]  B=[3, 3, 4, 4, 4]
  A=[5, 2, 4, 4, 4]  B=[5, 4, 2, 2, 2]
  A=[1, 1, 4, 2, 2]  B=[1, 1, 3, 4, 4]
  A=[2, 2, 2, 4, 4]  B=[2, 2, 4, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = None
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                ttb_pred = 0
                break
            elif val_b > val_a:
                ttb_pred = 1
                break
        if ttb_pred is not None:
            matches.append(row['response'] == ttb_pred)
    if not matches:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.0733 (var=0.0124)
**Predicted under pi_3:** 0.0467 (var=0.0112)
**Predicted under pi_5:** 0.1956 (var=0.0083)

### Experiment 4
**Design**
  A=[5, 2, 2, 2]  B=[4, 4, 4, 4]
  A=[2, 5, 5, 5]  B=[4, 3, 3, 3]
  A=[5, 2, 3, 3]  B=[4, 4, 3, 3]
  A=[2, 5, 3, 3]  B=[4, 4, 3, 3]
  A=[3, 5, 2, 2]  B=[3, 4, 4, 4]
  A=[3, 2, 5, 5]  B=[3, 4, 4, 4]
  A=[3, 3, 5, 2]  B=[3, 3, 4, 4]
  A=[3, 3, 2, 5]  B=[3, 3, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pred = 0 if a_wins > b_wins else 1
            tally_match.append(1.0 if row['response'] == tally_pred else 0.0)
            
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.8350 (var=0.0340)
**Predicted under pi_3:** 0.8750 (var=0.0231)
**Predicted under pi_5:** 0.7833 (var=0.0179)

### Experiment 5
**Design**
  A=[5, 5, 6, 6]  B=[10, 10, 0, 0]
  A=[0, 0, 8, 8]  B=[2, 2, 0, 0]
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[0, 10, 0, 0]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert response to probability of choosing A (where response == 0)
    is_a = 1.0 - data['response']
    
    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (5, 5, 6, 6) and b == (10, 10, 0, 0): return 1
        if a == (0, 0, 8, 8) and b == (2, 2, 0, 0): return 2
        if a == (10, 0, 0, 0) and b == (0, 2, 2, 2): return 3
        if a == (1, 1, 1, 1) and b == (0, 0, 0, 10): return 4
        if a == (10, 0, 0, 0) and b == (7, 1, 1, 1): return 5
        if a == (0, 10, 0, 0) and b == (1, 0, 1, 1): return 6
        return 0
        
    tt = data.apply(get_trial_type, axis=1)
    
    # WADD predicts A on trials 2, 3, 5, 6 and B on trials 1, 4.
    # Tallying predicts B on 3, 5, 6, A on 4, and tie on 1, 2.
    wadd_pro_trials = is_a[tt.isin([2, 3, 5, 6])]
    wadd_anti_trials = is_a[tt.isin([1, 4])]
    
    wadd_pro = wadd_pro_trials.mean() if len(wadd_pro_trials) > 0 else 0.0
    wadd_anti = wadd_anti_trials.mean() if len(wadd_anti_trials) > 0 else 0.0
    
    return float(wadd_pro - wadd_anti)
```

**Observed (real) value:** 0.3600 (var=0.1554)
**Predicted under pi_3:** 0.3400 (var=0.1219)
**Predicted under pi_5:** -0.4212 (var=0.0201)

### Experiment 6
**Design**
  A=[0, 3, 3, 3, 3]  B=[10, 0, 0, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 3, 3, 3, 3]
  A=[0, 0, 8, 8, 8]  B=[10, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[0, 0, 8, 8, 8]
  A=[2, 2, 2, 2, 2]  B=[10, 10, 0, 0, 0]
  A=[10, 10, 0, 0, 0]  B=[2, 2, 2, 2, 2]
  A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  A=[6, 6, 6, 6, 6]  B=[10, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    valid_trials = tally_a != tally_b
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (tally_b[valid_trials] > tally_a[valid_trials]).astype(int)
    actual_responses = data['response'].values[valid_trials]
    
    return float(np.mean(tally_pred == actual_responses))
```

**Observed (real) value:** 0.5250 (var=0.0619)
**Predicted under pi_3:** 0.5725 (var=0.0682)
**Predicted under pi_5:** 0.8004 (var=0.0097)

### Experiment 7
**Design**
  A=[10, 0, 0]  B=[0, 6, 6]
  A=[0, 10, 10]  B=[10, 5, 5]
  A=[9, 2, 2]  B=[2, 7, 7]
  A=[5, 8, 2]  B=[8, 0, 7]
  A=[5, 10, 0]  B=[9, 4, 0]
  A=[7, 5, 0]  B=[5, 8, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (10, 0, 0))
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (9, 2, 2))
    mask = is_t1 | is_t3
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.7000 (var=0.2000)
**Predicted under pi_3:** 0.6800 (var=0.1776)
**Predicted under pi_5:** 0.7662 (var=0.0208)

### Experiment 8
**Design**
  A=[0, 0, 10, 10]  B=[8, 8, 0, 0]
  A=[9, 9, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 0, 5, 5]
  A=[0, 0, 10, 8]  B=[10, 0, 0, 2]
  A=[2, 2, 8, 8]  B=[9, 5, 2, 2]
  A=[8, 8, 2, 2]  B=[2, 2, 10, 10]
  A=[0, 0, 8, 8]  B=[8, 8, 0, 0]
  A=[8, 0, 0, 4]  B=[0, 0, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    eqw_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        if a == (0, 0, 10, 10):
            eqw_choices += (resp == 0)
            total += 1
        elif a == (9, 9, 0, 0):
            eqw_choices += (resp == 1)
            total += 1
        elif a == (2, 2, 8, 8):
            eqw_choices += (resp == 0)
            total += 1
        elif a == (8, 8, 2, 2):
            eqw_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.5
    return float(eqw_choices / total)
```

**Observed (real) value:** 0.6850 (var=0.1870)
**Predicted under pi_3:** 0.5200 (var=0.2296)
**Predicted under pi_5:** 0.4925 (var=0.0048)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the eight experiments, Theory 1 (Weighted Additive, WADD) consistently outperforms Theory 2 (Tallying). WADD closely matches the observed data in nearly all experiments (e.g., Exp 1, 2, 5, 6, 7), successfully predicting preference reversals and indifference points that rely on cardinal feature magnitudes and cue validities. Tallying completely fails to capture the magnitude-driven preference reversals, predicting effects in the opposite direction of the empirical data. However, in Experiment 8, neither WADD nor Tallying perfectly captures the observed behavior. In Exp 8, subjects often choose the option with the higher unweighted sum of feature values, even when the alternative is superior on the most valid cues. This suggests that while subjects are sensitive to cardinal magnitudes (refuting Tallying), they may not always perfectly weight these magnitudes by cue validities (challenging WADD). Because Tallying is fundamentally degenerate across the board, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with an 'Equal Weights' (EQW) theory. The EQW theory posits that subjects evaluate options by taking an unweighted sum of their cardinal feature values, ignoring the provided cue validities. This distinguishes it from Tallying (which ignores magnitudes) and WADD (which weights by validities). An EQW model would simply sum the feature values for each option and apply a softmax choice rule over these sums. This could better capture the variance in Experiment 8 where subjects favored options with higher total sums despite lower cue validities."
}
```

## Usage

```json
{
  "prompt_token_count": 7336,
  "candidates_token_count": 356,
  "total_token_count": 8896
}
```
