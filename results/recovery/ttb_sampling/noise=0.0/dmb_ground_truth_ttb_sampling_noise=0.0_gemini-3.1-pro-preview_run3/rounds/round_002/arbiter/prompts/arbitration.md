# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## THEORY 2 — pi_2_1
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson 1993).
    # Stimulus is the pair of option feature vectors for the current trial:
    # array-like of shape (2, n_features), row 0 = option A, row 1 = option B.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Subjective weights are a non-linear scaling of objective validities.
    # gamma > 1 increases the relative weight of highly valid features.
    subjective_weights = validities ** gamma
    
    # Weighted sum per option (dot product with subjective validities).
    scores = stim @ subjective_weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To strongly dissociate Take-The-Best (TTB) from the Weighted Additive (WADD) rule, we construct a 5-feature design with a clear validity hierarchy. TTB strictly relies on the most valid discriminating feature, completely ignoring the rest of the feature profile. WADD, on the other hand, computes a compensatory sum, where multiple lower-validity features can outweigh a single higher-validity feature. The critical trials pit the highest-validity feature against a coalition of lower-validity features (e.g., Option A wins on the single best feature, but Option B wins on the remaining four). In these compensatory trials, TTB deterministically predicts Option A, while WADD predicts Option B. We also include agreement trials and trials where the highest-validity feature is tied, forcing TTB to drop down to the second most valid feature, ensuring a robust test of the lexicographic stopping rule against compensatory integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Take-The-Best (TTB) heuristic. Since the features are presented in decreasing order of validity, TTB simply scans the features from first to last and chooses the option that wins on the first discriminating feature. The experimental design includes multiple 'compensatory' trials where the single best discriminating feature points to one option, but the sum of the remaining features points to the other. TTB will have a near-perfect match rate on these trials, whereas the Weighted Additive (WADD) model will frequently choose the opposite option due to its compensatory nature, resulting in a significantly lower TTB match rate.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred is not None:
            matches.append(1 if r == pred else 0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.8735 (var=0.0058)
- Predicted under pi_2_1 (simulated): 0.6110 (var=0.0358)
- Observed on real data: 0.8671 (var=0.0115)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 10: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To strongly dissociate the Weighted Additive (WADD) rule from Take-The-Best (TTB), we use a 5-feature design with a linearly spaced set of validities. TTB strictly relies on the most valid discriminating feature and ignores the rest. WADD computes a compensatory sum (weighted by validity^gamma), allowing multiple lower-validity features to outweigh a single higher-validity feature, depending on the gamma parameter. We construct critical trials where one option wins on the single most valid discriminating feature (favored by TTB), while the other option wins on all or most of the remaining lower-validity features (favored by WADD for low-to-moderate gamma values). We also include trials where the best feature is tied, forcing TTB to drop down to the second-best feature, while WADD integrates across the entire profile. Agreement trials are included to balance the design and ensure robust parameter recovery.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chose the option with the higher total number of positive features (the 'Tallying' winner), specifically restricting the analysis to trials where this option is opposed by the Take-The-Best (TTB) heuristic (which always favors the option winning on the single most valid discriminating feature). Under TTB, this metric will be very close to 0, as subjects will almost never choose the Tallying winner when it conflicts with the most valid cue. Under the Weighted Additive (WADD) rule, this proportion will be substantially higher, because compensatory decision-makers will often allow multiple lower-validity cues to jointly outweigh a single higher-validity cue.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB winner: based on the first feature that discriminates
    diff = a_ratings - b_ratings
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    has_diff = np.any(diff != 0, axis=1)
    
    # Tally winner: based on simple sum of features
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    tally_winner = np.where(sum_a > sum_b, 0, np.where(sum_b > sum_a, 1, -1))
    
    # Identify conflict trials where TTB and Tallying disagree
    conflict = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(conflict):
        return 0.0
        
    responses = data['response'].values
    tally_chosen = (responses[conflict] == tally_winner[conflict]).sum()
    
    return float(tally_chosen / np.sum(conflict))

### RESULTS
- Predicted under pi_3 (simulated): 0.1311 (var=0.0080)
- Predicted under pi_2_1 (simulated): 0.4178 (var=0.1099)
- Observed on real data: 0.1378 (var=0.0101)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    a_wadd = np.sum(a_ratings * validities, axis=1)
    b_wadd = np.sum(b_ratings * validities, axis=1)
    
    conflict_mask = (a_wins > b_wins) & (b_wadd > a_wadd)
    
    if np.sum(conflict_mask) == 0:
        return 0.5
        
    return float(np.mean(data['response'].values[conflict_mask]))
```

**Observed (real) value:** 0.8844 (var=0.0081)
**Predicted under pi_3:** 0.8728 (var=0.0081)
**Predicted under pi_2_1:** 0.9467 (var=0.0101)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    is_a_heavy = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_b_heavy = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    is_a_many = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_b_many = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    trial_1 = is_a_heavy & is_b_many
    trial_2 = is_a_many & is_b_heavy
    
    critical_trials = trial_1 | trial_2
    
    if not critical_trials.any():
        return 0.5
        
    heavy_chosen = (trial_1 & (data['response'] == 0)) | (trial_2 & (data['response'] == 1))
    
    return float(heavy_chosen[critical_trials].mean())
```

**Observed (real) value:** 0.8533 (var=0.0160)
**Predicted under pi_3:** 0.8750 (var=0.0083)
**Predicted under pi_2_1:** 0.9433 (var=0.0380)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    v = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_v = np.sum(a_ratings * v, axis=1)
    b_v = np.sum(b_ratings * v, axis=1)
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    wadd_prefers_a = a_v > b_v
    wadd_prefers_b = b_v > a_v
    
    conflict_a = tally_prefers_a & wadd_prefers_b
    conflict_b = tally_prefers_b & wadd_prefers_a
    
    conflict_mask = conflict_a | conflict_b
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    tally_aligned = np.zeros(len(data), dtype=bool)
    tally_aligned[conflict_a & (responses == 0)] = True
    tally_aligned[conflict_b & (responses == 1)] = True
    
    return float(np.mean(tally_aligned[conflict_mask]))
```

**Observed (real) value:** 0.1500 (var=0.0150)
**Predicted under pi_3:** 0.1625 (var=0.0258)
**Predicted under pi_2_1:** 0.0575 (var=0.0264)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def f0_chosen(row):
        a0 = row['option_a_ratings'][0]
        b0 = row['option_b_ratings'][0]
        if a0 == b0:
            return None
        return a0 if row['response'] == 0 else b0
        
    f0_vals = data.apply(f0_chosen, axis=1).dropna()
    if len(f0_vals) == 0:
        return 0.5
    return float(f0_vals.mean())
```

**Observed (real) value:** 0.9033 (var=0.0065)
**Predicted under pi_3:** 0.8964 (var=0.0042)
**Predicted under pi_2_1:** 0.7872 (var=0.0401)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Take-The-Best) consistently and accurately predicts the observed data, matching the non-compensatory behavior exhibited by subjects. Theory 2 (Weighted Additive) systematically fails because it predicts a higher degree of compensatory trade-offs than subjects actually make. Given that Take-The-Best is highly successful, the compensatory paradigm of WADD is fundamentally mismatched to this specific task and dataset. Thus, Theory 2 should be completely replaced with a stronger non-compensatory competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new non-compensatory heuristic to serve as a stronger competitor to Take-The-Best. A good candidate would be a Probabilistic Take-The-Best model (where the feature search order is stochastic but weighted by validity) or an Elimination by Aspects (EBA) model. This will test whether the strict deterministic ordering of TTB is strictly necessary, or if a more flexible non-compensatory process can explain the minor deviations in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 7204,
  "candidates_token_count": 238,
  "total_token_count": 8229
}
```
