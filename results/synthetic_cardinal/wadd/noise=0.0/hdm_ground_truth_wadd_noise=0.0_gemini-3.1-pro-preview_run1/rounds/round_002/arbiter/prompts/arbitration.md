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


## THEORY 2 — pi_4
Equal Weight heuristic: People evaluate multi-attribute options by summing the cardinal values of all features, treating every feature as equally important regardless of its objective or subjective validity. This allows decision makers to account for the magnitude of feature differences (unlike Tallying or Take The Best) while avoiding the cognitive complexity of weighting each feature by its validity (unlike Weighted Additive).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Equal Weight expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Equal Weight sums the cardinal feature values, ignoring validities.
    score_a = np.sum(a)
    score_b = np.sum(b)
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=3):** [0.9, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[10, 0, 0]  B=[0, 6, 6]
  trial 2: A=[0, 10, 10]  B=[10, 5, 5]
  trial 3: A=[9, 2, 2]  B=[2, 7, 7]
  trial 4: A=[5, 8, 2]  B=[8, 0, 7]
  trial 5: A=[5, 10, 0]  B=[9, 4, 0]
  trial 6: A=[7, 5, 0]  B=[5, 8, 0]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) rule from the Equal Weight (EQW) heuristic. Both rules sum cardinal feature values, but WADD weights them by validities whereas EQW ignores validities and treats all features equally. By using highly dispersed validities (0.9, 0.6, 0.5) and a large rating scale (0-10), we construct three types of dissociating trials: 1) Preference reversals, where one option has a higher unweighted sum but a lower validity-weighted sum; 2) EQW indifference, where unweighted sums are equal but weighted sums strongly favor one option; and 3) WADD indifference, where validity-weighted sums are perfectly balanced but unweighted sums favor one option. These trial types ensure that the two theories predict distinctly different choice patterns and indifference points.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on trials 1 and 3 where WADD and EQW have strong, opposing preference reversals. On these trials, WADD assigns a higher validity-weighted score to Option A, predicting a response near 0. Conversely, EQW assigns a higher unweighted sum to Option B, predicting a response near 1. The metric computes the proportion of Option B choices on these specific trials, yielding a value close to 0 for WADD and close to 1 for EQW.

Source:
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (10, 0, 0))
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (9, 2, 2))
    mask = is_t1 | is_t3
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6800 (var=0.1776)
- Predicted under pi_4 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.7000 (var=0.2000)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 10, 10]  B=[8, 8, 0, 0]
  trial 2: A=[9, 9, 0, 0]  B=[0, 0, 10, 10]
  trial 3: A=[10, 0, 0, 0]  B=[0, 0, 5, 5]
  trial 4: A=[0, 0, 10, 8]  B=[10, 0, 0, 2]
  trial 5: A=[2, 2, 8, 8]  B=[9, 5, 2, 2]
  trial 6: A=[8, 8, 2, 2]  B=[2, 2, 10, 10]
  trial 7: A=[0, 0, 8, 8]  B=[8, 8, 0, 0]
  trial 8: A=[8, 0, 0, 4]  B=[0, 0, 7, 10]

**Rationale:** To distinctively separate Equal Weight (EQW) from Weighted Additive (WADD) without duplicating previous 3-feature designs, this experiment uses 4 features with a wide spread of validities [0.9, 0.8, 0.6, 0.5] and a rating maximum of 10. We construct trials that systematically manipulate the locus of the cardinal advantage. In 'preference reversal' trials, one option has a large advantage on low-validity features (maximizing the unweighted sum for EQW), while the other has a moderate advantage on high-validity features (maximizing the weighted sum for WADD). We also include trials where one theory predicts exact indifference while the other predicts a strong preference. This provides a robust quantitative dissociation across multiple distinct trial types.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the Equal Weight (EQW) theory on the specific subset of 'preference reversal' trials where EQW and Weighted Additive (WADD) predict strictly opposite choices. For EQW, this metric will be close to 1.0, whereas for WADD, it will be close to 0.0.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.5200 (var=0.2296)
- Predicted under pi_4 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.6850 (var=0.1870)

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
**Predicted under pi_4:** 0.0000 (var=0.0000)

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
**Predicted under pi_4:** 1.0000 (var=0.0000)

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
**Predicted under pi_4:** 0.5000 (var=0.0000)

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
**Predicted under pi_4:** 0.7500 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, Theory 1 (Weighted Additive, pi_3) consistently outperforms Theory 2 (Equal Weight, pi_4). In Experiments 1, 3, 4, 5, and 6, the predictions of pi_3 are remarkably close to the observed data, whereas pi_4 repeatedly predicts extreme, deterministic behavior (e.g., 1.0 or 0.0) that fails to capture the nuances of human decision-making. The observed data clearly indicates that subjects do not completely ignore feature validities, which is the core assumption of Equal Weight. Because Equal Weight fails to capture the empirical reality that subjects are sensitive to validities and magnitudes, it should be discarded in favor of a new theoretical competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Equal Weight with 'Tallying' (also known as Majority of Confirming Dimensions). Under Tallying, decision makers simply count the number of features on which each option is strictly superior to the other, ignoring both the validities of the features and the cardinal magnitudes of the differences. This provides a distinct, non-compensatory heuristic baseline that accounts for feature-by-feature comparisons rather than summing values."
}
```

## Usage

```json
{
  "prompt_token_count": 5865,
  "candidates_token_count": 277,
  "total_token_count": 7124
}
```
