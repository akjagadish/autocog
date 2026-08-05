# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Tallying (Equal Weights) Theory with Very Soft Softmax: Decision-makers evaluate options by assigning equal weight to all features, simply counting the number of positive features for each option. However, their choices are not highly deterministic; they employ a 'very soft' decision rule where the difference in tallies yields only a mild preference (e.g., ~55/45 splits) rather than an overwhelming one, reflecting high levels of noise, uncertainty, or guessing in human decision-making in this domain.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Equal Weights expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Linear Weighted Additive Theory: Decision makers evaluate options by summing the features of each option, weighted linearly by their predictive validity (specifically, the validity's edge over chance, validity - 0.5). This represents a moderate integration strategy that avoids the extreme dominance of top cues seen in log-odds (Bayesian) weighting, while still differentiating cue importance unlike Equal-Weight/Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Linear WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to linear weights (edge over chance)
    weights = val - 0.5
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** This design systematically dissociates the Tallying (Equal Weights) theory from the Linear Weighted Additive (Linear WADD) theory. Tallying simply counts the number of positive features, ignoring validities, and predicts choices based on the difference in tallies. Linear WADD weights features by their edge over chance (validity - 0.5). By using validities of [0.95, 0.85, 0.75, 0.65, 0.55], the linear weights are [0.45, 0.35, 0.25, 0.15, 0.05]. The trials are constructed to pit these two integration strategies against each other. For example, in some trials, one option wins on a sheer majority of features (strongly favored by Tallying), while the other option wins on fewer, but highly valid features, such that the linear sum of validities favors it (Linear WADD). Other trials include ties in Tallying where Linear WADD strongly predicts one option, or ties in Linear WADD where Tallying strongly predicts one option.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The advocated theory (Tallying with a very soft softmax) predicts that choices will be highly stochastic and relatively close to 50/50 across all trial types, resulting in a low variance (and standard deviation) of choice proportions across the 8 unique trial types. In contrast, the competing theory (Linear WADD) predicts strong preferences in many trials (e.g., heavily favoring A in Trial 8 and B in Trial 7), leading to a much wider spread of choice probabilities and a significantly higher standard deviation across trial types.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create a hashable key for each unique trial type
    keys = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x)) + '_' + \
           data['option_b_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
           
    # Calculate the mean response (proportion of choosing B) for each trial type
    means = data.groupby(keys)['response'].mean()
    
    # Return the standard deviation of these choice proportions
    return float(np.std(means.values))

### RESULTS
- Predicted under pi_5 (simulated): 0.0456 (var=0.0016)
- Predicted under pi_4 (simulated): 0.2004 (var=0.0045)
- Observed on real data: 0.0851 (var=0.0027)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design aims to quantitatively dissociate Linear Weighted Additive (Linear WADD) from Tallying (Equal Weights) with a very soft softmax. Tallying simply counts the number of positive features and applies a low-beta softmax, resulting in choices that lean towards the option with more positive features but remain close to 50/50. Linear WADD, in contrast, applies a linear weight (validity - 0.5) to each feature and can have a much higher beta, leading to strong, deterministic predictions when the weighted sums differ. By using a specific validity spread [0.95, 0.85, 0.75, 0.65, 0.55], we create trials where Tallying predicts a clear advantage for one option (e.g., Option B has 3 features vs. Option A's 1 feature) while Linear WADD predicts a tie, or where Tallying predicts a tie (both have 2 features) but Linear WADD strongly predicts Option A due to higher validities. Additionally, we include reversal trials where Option B wins the tally but Option A wins the weighted linear sum.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses Option A specifically on 'reversal trials'. Reversal trials are designed such that Option B has a higher total number of positive features (favored by Tallying), but Option A has a higher sum of validities (favored by Linear WADD). Because Tallying employs a soft decision rule, it will choose Option A less than 50% of the time on these trials. Conversely, Linear WADD will systematically prefer Option A, yielding a choice proportion strictly above 50% (and often much higher, depending on the beta parameter). By isolating these conflicting trials, we can cleanly discriminate between the two decision-making strategies.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D arrays
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    # Validities are [0.95, 0.85, 0.75, 0.65, 0.55]
    # Linear WADD weights (validity - 0.5) are:
    weights = np.array([0.45, 0.35, 0.25, 0.15, 0.05])
    
    # Calculate WADD differences
    wadd_A = A.dot(weights)
    wadd_B = B.dot(weights)
    wadd_diff = wadd_A - wadd_B
    
    # Calculate Tallying (Equal Weights) differences
    tally_A = A.sum(axis=1)
    tally_B = B.sum(axis=1)
    tally_diff = tally_A - tally_B
    
    # Identify "reversal" trials where WADD predicts A but Tallying predicts B
    mask = (wadd_diff > 0.01) & (tally_diff < -0.01)
    
    if np.sum(mask) == 0:
        return 0.5
        
    # Return the proportion of times subjects chose Option A (response == 0)
    chose_A = (data['response'].values[mask] == 0)
    return float(np.mean(chose_A))


### RESULTS
- Predicted under pi_5 (simulated): 0.4389 (var=0.0074)
- Predicted under pi_4 (simulated): 0.6750 (var=0.0115)
- Observed on real data: 0.4900 (var=0.0163)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    
    first_diff_idx = np.argmax(non_zero, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_choice = np.where(first_diff_val > 0, 0, 1)
    has_diff = np.any(non_zero, axis=1)
    
    if not np.any(has_diff):
        return 0.5
        
    match = (data['response'].values[has_diff] == ttb_choice[has_diff])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4083 (var=0.0075)
**Predicted under pi_5:** 0.4098 (var=0.0039)
**Predicted under pi_4:** 0.3348 (var=0.0120)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    preds = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    valid = preds != -1
    if not valid.any():
        return 0.5
        
    matches = (data['response'].values[valid] == preds[valid])
    return float(matches.mean())
```

**Observed (real) value:** 0.6125 (var=0.0092)
**Predicted under pi_5:** 0.5663 (var=0.0046)
**Predicted under pi_4:** 0.6719 (var=0.0072)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total_relevant = 0
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: WADD prefers A, Tallying prefers B
        if a == [1, 0, 0, 0] and b == [0, 1, 1, 1]:
            wadd_choices += (1 if resp == 0 else 0)
            total_relevant += 1
        # Trial 3: WADD prefers A, Tallying prefers B
        elif a == [0, 1, 0, 0] and b == [0, 0, 1, 1]:
            wadd_choices += (1 if resp == 0 else 0)
            total_relevant += 1
        # Trial 6: WADD prefers B, Tallying prefers A
        elif a == [0, 0, 1, 1] and b == [0, 1, 0, 0]:
            wadd_choices += (1 if resp == 1 else 0)
            total_relevant += 1
        # Trial 8: WADD prefers B, Tallying prefers A
        elif a == [0, 1, 1, 1] and b == [1, 0, 0, 0]:
            wadd_choices += (1 if resp == 1 else 0)
            total_relevant += 1
            
    if total_relevant == 0:
        return 0.5
    return wadd_choices / total_relevant
```

**Observed (real) value:** 0.3858 (var=0.0151)
**Predicted under pi_5:** 0.4217 (var=0.0087)
**Predicted under pi_4:** 0.5429 (var=0.0054)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            if row['response'] == 0:
                consistent += 1
            total += 1
        elif b_wins > a_wins:
            if row['response'] == 1:
                consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return consistent / total
```

**Observed (real) value:** 0.5610 (var=0.0074)
**Predicted under pi_5:** 0.5860 (var=0.0059)
**Predicted under pi_4:** 0.4190 (var=0.0031)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design pits a single extremely valid feature (0.99) in Option A
    # against multiple moderately valid features in Option B.
    # Under Log-Odds WADD, the 0.99 feature gets a massive weight (~4.6) which dominates any combination of the rest.
    # Under Linear WADD, the 0.99 feature gets a weight of 0.49, which is easily outvoted by combinations (e.g., 0.85+0.80 -> 0.35+0.30=0.65).
    # Thus, Log-Odds strongly predicts choosing Option A (response == 0),
    # whereas Linear WADD strongly predicts choosing Option B (response == 1).
    return float((data['response'] == 0).mean())
```

**Observed (real) value:** 0.4092 (var=0.0156)
**Predicted under pi_5:** 0.4158 (var=0.0052)
**Predicted under pi_4:** 0.3135 (var=0.0106)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_and_chose_high_val(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Pairs where Linear WADD favors the combination of lower-validity cues (B) 
        # and Log-Odds WADD favors the single higher-validity cue (A)
        pairs = [
            ((1, 0, 0, 0, 0), (0, 1, 0, 1, 0)),
            ((0, 1, 0, 0, 0), (0, 0, 1, 1, 0)),
            ((1, 0, 0, 0, 1), (0, 1, 0, 1, 1))
        ]
        
        for p_a, p_b in pairs:
            if a == p_a and b == p_b:
                return 1 if row['response'] == 0 else 0
            elif a == p_b and b == p_a:
                return 1 if row['response'] == 1 else 0
        return None
        
    vals = data.apply(is_target_and_chose_high_val, axis=1).dropna()
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())
```

**Observed (real) value:** 0.4533 (var=0.0077)
**Predicted under pi_5:** 0.4467 (var=0.0038)
**Predicted under pi_4:** 0.4633 (var=0.0047)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, Theory 1 (Tallying with Very Soft Softmax) consistently and accurately predicts the observed metrics, often coming within 0.01 to 0.04 of the true values. Theory 2 (Linear Weighted Additive) systematically fails to capture the data, consistently overestimating the degree to which participants integrate feature validities and underestimating the high level of stochasticity (or reliance on simple feature counting) in their choices. Because Theory 1 is already exceptionally strong, Theory 2's core assumption of linear validity weighting appears fundamentally misaligned with human behavior in this paradigm.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Validity-Biased Tallying' or 'Lexicographic Fallback' theory. Since pure Tallying with high noise (Theory 1) is very successful, a strong competitor would be a theory that posits participants primarily tally features but apply a small lexicographic bonus (e.g., breaking ties using the most valid cue) or use a mixture model where a majority of trials are decided by tallying and a small minority by Take-The-Best. This will test whether there is any residual sensitivity to validity that pure Tallying misses."
}
```

## Usage

```json
{
  "prompt_token_count": 7851,
  "candidates_token_count": 281,
  "total_token_count": 8935
}
```
