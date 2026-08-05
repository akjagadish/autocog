# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Weighted Additive (WADD) Theory with Log-Odds: People compare two options by considering all available features, weighting each feature by its subjective validity transformed into log-odds (Bayesian weights), and summing these weighted values to form an overall valuation for each option. This normative integration allows multiple lower-validity cues to jointly outweigh a single high-validity cue, while the non-linear log-odds transformation ensures that the highest-validity cues maintain sufficient relative influence. Response noise is modeled via a softmax over the WADD scores and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to log-odds weights for Bayesian integration.
    # Clip validities to avoid division by zero or log(0) for extreme values.
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.99, 0.85, 0.8, 0.75, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To dissociate WADD with Log-Odds from Linear WADD, we exploit the non-linear transformation of log-odds. A highly valid feature (e.g., 0.99) has a massive log-odds weight (approx 4.6), which easily dominates the sum of several moderate features (e.g., 0.85, 0.80, 0.75). In contrast, the Linear WADD theory weights features by their edge over chance (validity - 0.5). For 0.99, the linear weight is 0.49, which is easily outvoted by combinations like 0.85 + 0.80 (0.35 + 0.30 = 0.65). By pitting a single extremely valid feature against a combination of moderately valid features, we create a strong quantitative dissociation where Log-Odds strongly predicts Option A and Linear WADD predicts Option B.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the overall proportion of trials where Option A is chosen. By design, Option A always contains the single highest-validity cue, which dominates under log-odds weighting but is outvoted by the sum of multiple moderate cues under linear weighting. Thus, this proportion will be near 1 for Log-Odds WADD and near 0 for Linear WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # The experimental design pits a single extremely valid feature (0.99) in Option A
    # against multiple moderately valid features in Option B.
    # Under Log-Odds WADD, the 0.99 feature gets a massive weight (~4.6) which dominates any combination of the rest.
    # Under Linear WADD, the 0.99 feature gets a weight of 0.49, which is easily outvoted by combinations (e.g., 0.85+0.80 -> 0.35+0.30=0.65).
    # Thus, Log-Odds strongly predicts choosing Option A (response == 0),
    # whereas Linear WADD strongly predicts choosing Option B (response == 1).
    return float((data['response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8290 (var=0.0106)
- Predicted under pi_4 (simulated): 0.3135 (var=0.0106)
- Observed on real data: 0.4092 (var=0.0156)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.92, 0.82, 0.72, 0.62, 0.52]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To dissociate Linear WADD from Log-Odds WADD, we exploit the difference between linear weights (validity - 0.5) and non-linear Bayesian weights (log-odds). Log-odds weighting assigns exponentially greater importance to highly valid cues, whereas linear weighting treats the differences more additively. By setting validities to [0.92, 0.82, 0.72, 0.62, 0.52], we create choice pairs where a single higher-validity cue is pitted against a combination of lower-validity cues. For example, comparing cue 1 (0.92) against cues 2 and 4 (0.82 and 0.62): Linear WADD assigns cue 1 a weight of 0.42, and cues 2+4 a combined weight of 0.32 + 0.12 = 0.44, thus favoring the combination. In contrast, Log-Odds WADD assigns cue 1 a weight of ~2.44, and cues 2+4 a combined weight of ~1.51 + 0.48 = 1.99, thus strongly favoring the single highly valid cue. These systematic reversals isolate the integration function.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By isolating specific trials where a single high-validity cue is pitted against two lower-validity cues, we can perfectly dissociate the two integration models. Linear WADD (which subtracts 0.5 from validities) evaluates the sum of the lower-validity cues as greater than the single high-validity cue, thus favoring the combination. Log-Odds WADD (Bayesian weighting) assigns exponentially more weight to the top validities, thus favoring the single high-validity cue. The metric computes the proportion of choices favoring the single high-validity cue on these critical trials.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.7254 (var=0.0179)
- Predicted under pi_4 (simulated): 0.4633 (var=0.0047)
- Observed on real data: 0.4533 (var=0.0077)

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
**Predicted under pi_3:** 0.3802 (var=0.0046)
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
**Predicted under pi_3:** 0.5902 (var=0.0031)
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
**Predicted under pi_3:** 0.8171 (var=0.0142)
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
**Predicted under pi_3:** 0.2057 (var=0.0076)
**Predicted under pi_4:** 0.4190 (var=0.0031)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across multiple experiments, Theory 1 (Log-Odds WADD) consistently and massively overpredicts the influence of the single highest-validity cue. For example, in Experiment 1 and 2, it predicts choices favoring the single high-validity cue at rates of ~83% and ~73%, while humans choose it only ~41% and ~45% of the time. Theory 2 (Linear WADD) is much closer, capturing the human tendency to integrate multiple lower-validity cues to outvote a single high-validity cue. Furthermore, Experiments 5 and 6 reveal that humans actually have a strong preference for the option with the higher sheer number of positive features (Tallying), choosing the Tallying-favored option over the WADD-favored option more than half the time (e.g., Exp 5 observed metric is 0.38, meaning ~62% of choices align with Tallying). Theory 1 is fundamentally misaligned with this flat-weighting behavior.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying (Equal Weights)' theory. In this theory, decision-makers largely ignore the explicit validities and simply count the number of positive features for each option, choosing the one with the most positive features (with a softmax/noise component). This would naturally explain why multiple lower-validity cues consistently outvote single high-validity cues (Exps 1 and 2) and why subjects often prefer the option with more positive features even when validity-weighted scores suggest otherwise (Exps 5 and 6)."
}
```

## Usage

```json
{
  "prompt_token_count": 6505,
  "candidates_token_count": 367,
  "total_token_count": 7968
}
```
