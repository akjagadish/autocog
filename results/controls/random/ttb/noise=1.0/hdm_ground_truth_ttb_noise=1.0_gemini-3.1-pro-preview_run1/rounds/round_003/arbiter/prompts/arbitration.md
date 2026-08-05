# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
In complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. Thus, behavior resembles zero-intelligence random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Single Random Feature heuristic: On each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all. An epsilon parameter allows blending this strategy with pure random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    epsilon = float(parameters["epsilon"])
    n_features = stim.shape[1]
    
    wins = (stim[0] > stim[1]).sum()
    losses = (stim[0] < stim[1]).sum()
    ties = (stim[0] == stim[1]).sum()
    
    p_A_core = (wins + 0.5 * ties) / n_features
    p_B_core = (losses + 0.5 * ties) / n_features
    
    probs = np.array([p_A_core, p_B_core])
    return (1.0 - epsilon) * probs + epsilon * 0.5

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 5

**Trial pairs (n=8):**
  trial 1: A=[5, 5, 5, 5]  B=[0, 0, 0, 0]
  trial 2: A=[5, 5, 5, 0]  B=[0, 0, 0, 5]
  trial 3: A=[5, 5, 0, 0]  B=[0, 0, 5, 5]
  trial 4: A=[5, 0, 0, 0]  B=[0, 5, 5, 5]
  trial 5: A=[0, 0, 0, 0]  B=[5, 5, 5, 5]
  trial 6: A=[3, 3, 3, 3]  B=[3, 3, 3, 3]
  trial 7: A=[4, 2, 4, 2]  B=[2, 4, 2, 4]
  trial 8: A=[1, 4, 1, 4]  B=[4, 1, 4, 1]

**Rationale:** To quantitatively dissociate Random Guessing from the Single Random Feature heuristic, we systematically vary the proportion of features on which Option A strictly dominates Option B. The Random Guessing theory predicts a flat 50% probability of choosing A across all trials, completely ignoring the feature values. The Single Random Feature theory assumes the subject randomly samples one feature and decides based on that feature alone, predicting that the probability of choosing A will scale linearly with the number of features on which A is superior to B (e.g., 100% if A wins all features, 75% if A wins 3 out of 4, 25% if A wins 1 out of 4). By including trials that range from A dominating all features to B dominating all features, alongside tie trials, we can perfectly distinguish the flat prediction of Random Guessing from the monotonic step-wise prediction of Single Random Feature.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chose the option that dominates on a strict majority of features. Under Random Guessing, this proportion will be exactly 0.5 (since they ignore features). Under the Single Random Feature heuristic, subjects are far more likely to choose the option with more feature wins (e.g., 100% on 4-0 trials, 75% on 3-1 trials), leading to a score substantially higher than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wins_a = (a_ratings > b_ratings).sum(axis=1)
    wins_b = (b_ratings > a_ratings).sum(axis=1)
    
    mask = wins_a != wins_b
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    wa = wins_a[mask]
    wb = wins_b[mask]
    
    correct = ((wa > wb) & (responses == 0)) | ((wb > wa) & (responses == 1))
    return float(np.mean(correct))

### RESULTS
- Predicted under pi_4 (simulated): 0.5071 (var=0.0052)
- Predicted under pi_5 (simulated): 0.6933 (var=0.0177)
- Observed on real data: 0.4917 (var=0.0039)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[8, 8, 8, 8, 8]  B=[2, 2, 2, 2, 2]
  trial 2: A=[8, 8, 8, 8, 2]  B=[2, 2, 2, 2, 8]
  trial 3: A=[8, 8, 8, 2, 2]  B=[2, 2, 2, 8, 8]
  trial 4: A=[8, 8, 5, 2, 2]  B=[2, 2, 5, 8, 8]
  trial 5: A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]
  trial 6: A=[8, 2, 2, 2, 2]  B=[2, 8, 8, 8, 8]
  trial 7: A=[2, 2, 2, 2, 2]  B=[8, 8, 8, 8, 8]
  trial 8: A=[8, 8, 8, 8, 5]  B=[2, 2, 2, 2, 5]

**Rationale:** To quantitatively dissociate the Single Random Feature (SRF) heuristic from the Random Guessing theory, we use a 5-feature design and systematically manipulate the number of features on which Option A wins, loses, or ties against Option B. The Random Guessing theory predicts a flat 50% probability of choosing A across all trials, completely ignoring the feature values. The SRF theory assumes the subject randomly samples exactly one feature and decides based on that feature alone (guessing if tied). Thus, SRF predicts that the probability of choosing A will scale linearly with the number of features on which A beats B, plus half the number of tied features. We construct a gradient of choice probabilities predicted by SRF (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0) by varying the win/loss/tie counts, while Random Guessing rigidly predicts 0.5 for every single trial.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the covariance between the subject's choice and the theoretical advantage of Option A under the Single Random Feature (SRF) theory. For the Random Guessing theory, choices are independent of the options' features, yielding an expected value of 0. For the SRF theory, subjects are more likely to choose Option A when it wins on more features, yielding a strictly positive expected value. The vectorized implementation ensures robust and fast execution.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    # Calculate the SRF predicted probability of choosing A
    wins = np.sum(a > b, axis=1)
    ties = np.sum(a == b, axis=1)
    p_A = (wins + 0.5 * ties) / a.shape[1]
    
    # Map response to +1 for choosing A (response=0) and -1 for choosing B (response=1)
    choice_sign = 1 - 2 * data['response']
    
    # Calculate the covariance-like product between choice and SRF advantage
    # Random Guessing expected value = 0
    # SRF expected value > 0
    return float(np.mean(choice_sign * (p_A - 0.5)))

### RESULTS
- Predicted under pi_4 (simulated): -0.0018 (var=0.0012)
- Predicted under pi_5 (simulated): 0.0975 (var=0.0042)
- Observed on real data: -0.0008 (var=0.0015)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 4, 4]  B=[5, 3, 3]
  A=[5, 0, 0]  B=[2, 1, 1]
  A=[2, 5, 0]  B=[3, 1, 1]
  A=[0, 4, 4]  B=[2, 3, 3]
  A=[0, 5, 5]  B=[3, 4, 4]
  A=[4, 1, 0]  B=[1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_pred = (b_wins > a_wins).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.4917 (var=0.0021)
**Predicted under pi_4:** 0.5052 (var=0.0023)
**Predicted under pi_5:** 0.5938 (var=0.0040)

### Experiment 4
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[10, 2, 2]  B=[8, 3, 3]
  A=[9, 5, 4]  B=[1, 5, 8]
  A=[5, 5, 5]  B=[0, 8, 8]
  A=[1, 9, 9]  B=[8, 8, 8]
  A=[9, 10, 1]  B=[10, 1, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        resp = row['response']
        # Trial 1: A=[2, 6, 6], B=[10, 5, 5]
        if a[0] == 2:
            wadd_match.append(1 if resp == 1 else 0)
        # Trial 5: A=[1, 9, 9], B=[8, 8, 8]
        elif a[0] == 1:
            wadd_match.append(1 if resp == 1 else 0)
    if not wadd_match:
        return 0.5
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.5000 (var=0.0073)
**Predicted under pi_4:** 0.4900 (var=0.0087)
**Predicted under pi_5:** 0.4238 (var=0.0118)

### Experiment 5
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[5, 10, 10]  B=[6, 0, 0]
  A=[8, 2, 10]  B=[8, 3, 0]
  A=[0, 8, 8]  B=[1, 0, 0]
  A=[10, 0, 0]  B=[9, 9, 9]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[2, 10, 10]  B=[3, 2, 2]
  A=[7, 8, 0]  B=[7, 7, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = np.zeros(len(data), dtype=int) - 1
    for i in range(len(data)):
        for j in range(a_ratings.shape[1]):
            if diff[i, j] > 0:
                ttb_choices[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_choices[i] = 1
                break
                
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    match = (responses[valid] == ttb_choices[valid])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4829 (var=0.0017)
**Predicted under pi_4:** 0.4860 (var=0.0027)
**Predicted under pi_5:** 0.4506 (var=0.0046)

### Experiment 6
**Design**
  A=[6, 0, 0]  B=[5, 10, 10]
  A=[10, 2, 0]  B=[10, 1, 10]
  A=[1, 10, 10]  B=[2, 0, 0]
  A=[5, 5, 5]  B=[6, 0, 0]
  A=[0, 8, 8]  B=[1, 1, 1]
  A=[7, 7, 7]  B=[7, 8, 0]
  A=[4, 9, 0]  B=[4, 10, 0]
  A=[0, 0, 10]  B=[0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (6, 0, 0) and b == (5, 10, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (10, 2, 0) and b == (10, 1, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (1, 10, 10) and b == (2, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (5, 5, 5) and b == (6, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 8, 8) and b == (1, 1, 1):
            ttb_match += (resp == 1)
            count += 1
        elif a == (7, 7, 7) and b == (7, 8, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (4, 9, 0) and b == (4, 10, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 0, 10) and b == (0, 1, 0):
            ttb_match += (resp == 1)
            count += 1
    if count == 0:
        return 0.5
    return ttb_match / count

```

**Observed (real) value:** 0.5017 (var=0.0022)
**Predicted under pi_4:** 0.5185 (var=0.0025)
**Predicted under pi_5:** 0.4773 (var=0.0032)

### Experiment 7
**Design**
  A=[10, 10, 10]  B=[0, 0, 0]
  A=[0, 0, 0]  B=[10, 10, 10]
  A=[9, 8, 7]  B=[1, 2, 3]
  A=[1, 2, 3]  B=[9, 8, 7]
  A=[10, 0, 10]  B=[0, 10, 0]
  A=[0, 10, 0]  B=[10, 0, 10]
  A=[8, 2, 5]  B=[2, 8, 5]
  A=[2, 8, 5]  B=[8, 2, 5]
  A=[5, 5, 5]  B=[5, 5, 5]
  A=[6, 4, 8]  B=[4, 6, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    correct = np.where(sum_a > sum_b, 0, 1)
    
    return float(np.mean(data.loc[mask, 'response'] == correct[mask]))
```

**Observed (real) value:** 0.4768 (var=0.0049)
**Predicted under pi_4:** 0.5019 (var=0.0036)
**Predicted under pi_5:** 0.6692 (var=0.0134)

### Experiment 8
**Design**
  A=[10, 10, 10, 10]  B=[0, 0, 0, 0]
  A=[8, 8, 8, 8]  B=[2, 2, 2, 2]
  A=[6, 6, 6, 6]  B=[4, 4, 4, 4]
  A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  A=[4, 4, 4, 4]  B=[6, 6, 6, 6]
  A=[2, 2, 2, 2]  B=[8, 8, 8, 8]
  A=[0, 0, 0, 0]  B=[10, 10, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 10, 10, 0]
  A=[0, 10, 10, 0]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[9, 2, 2, 2]  B=[2, 9, 9, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for each unique trial pair
    trial_keys = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    
    # Calculate the probability of choosing option A (response == 0) for each unique trial
    p_choose_A = (data['response'] == 0).groupby(trial_keys).mean()
    
    # Return the variance of these probabilities across different trial types
    return float(np.var(p_choose_A))
```

**Observed (real) value:** 0.0005 (var=0.0001)
**Predicted under pi_4:** 0.0004 (var=0.0001)
**Predicted under pi_5:** 0.0313 (var=0.0014)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, the observed data consistently and overwhelmingly supports Theory 1 (Random Guessing). The metrics evaluating choice proportions, covariances, and variance of choice probabilities repeatedly center around 0.5, 0.0, and 0.0 respectively. Theory 2 (Single Random Feature) predicts systematic deviations from pure chance based on the distribution of feature wins, which are entirely absent in the empirical data. Subjects appear to be completely ignoring the stimulus features in this paradigm.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since subjects are completely ignoring the stimulus features and their aggregate behavior is indistinguishable from random guessing, proposing another feature-based heuristic will likely fail. Instead, propose a 'Response Inertia and Alternation' theory. This theory posits that while subjects ignore the features (producing 50% aggregate choice probabilities), their trial-by-trial choices are driven by sequential dependencies\u2014either perseveration (repeating the previous choice) or alternation (switching choices). This provides a mechanistic alternative to pure zero-memory random guessing that can be tested using sequential choice metrics."
}
```

## Usage

```json
{
  "prompt_token_count": 6612,
  "candidates_token_count": 250,
  "total_token_count": 7727
}
```
