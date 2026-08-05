# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[10, 10, 10]  B=[0, 0, 0]
  trial 2: A=[0, 0, 0]  B=[10, 10, 10]
  trial 3: A=[9, 8, 7]  B=[1, 2, 3]
  trial 4: A=[1, 2, 3]  B=[9, 8, 7]
  trial 5: A=[10, 0, 10]  B=[0, 10, 0]
  trial 6: A=[0, 10, 0]  B=[10, 0, 10]
  trial 7: A=[8, 2, 5]  B=[2, 8, 5]
  trial 8: A=[2, 8, 5]  B=[8, 2, 5]
  trial 9: A=[5, 5, 5]  B=[5, 5, 5]
  trial 10: A=[6, 4, 8]  B=[4, 6, 2]

**Rationale:** To strictly dissociate the Random Guessing theory from the Weighted Additive (WADD) model, we must construct a gradient of choice difficulty. WADD predicts that choice probability will scale monotonically with the weighted sum difference between options: it predicts near-deterministic choices when one option completely dominates the other by a large margin (e.g., 10s across all features vs 0s), and approaches chance only as the options become perfectly balanced. In contrast, the Random Guessing theory posits that subjects completely fail to integrate these cues and will guess uniformly at random (50/50) across all trials, regardless of whether the WADD difference is massive, moderate, or zero. By including transparently dominant options alongside complex compensatory tradeoffs, we can test whether subjects show any sensitivity to the magnitude of evidence (as WADD predicts) or if they remain at chance everywhere (as Random Guessing predicts).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option with the strictly greater unweighted sum of features. Under the Random Guessing theory, subjects choose options randomly, so this metric will tightly cluster around 0.5. Under the Weighted Additive (WADD) theory, the chosen experimental design ensures that the option with the higher unweighted sum also heavily dominates in the weighted sum (e.g., [10, 10, 10] vs [0, 0, 0]). Because the provided WADD policy deterministically selects the option with the highest probability (via argmax), WADD subjects will consistently choose the dominating option, yielding a metric value close to 1.0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    correct = np.where(sum_a > sum_b, 0, 1)
    
    return float(np.mean(data.loc[mask, 'response'] == correct[mask]))

### RESULTS
- Predicted under pi_4 (simulated): 0.5019 (var=0.0036)
- Predicted under pi_2 (simulated): 0.9657 (var=0.0111)
- Observed on real data: 0.4768 (var=0.0049)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=12):**
  trial 1: A=[10, 10, 10, 10]  B=[0, 0, 0, 0]
  trial 2: A=[8, 8, 8, 8]  B=[2, 2, 2, 2]
  trial 3: A=[6, 6, 6, 6]  B=[4, 4, 4, 4]
  trial 4: A=[5, 5, 5, 5]  B=[5, 5, 5, 5]
  trial 5: A=[4, 4, 4, 4]  B=[6, 6, 6, 6]
  trial 6: A=[2, 2, 2, 2]  B=[8, 8, 8, 8]
  trial 7: A=[0, 0, 0, 0]  B=[10, 10, 10, 10]
  trial 8: A=[10, 0, 0, 0]  B=[0, 10, 10, 0]
  trial 9: A=[0, 10, 10, 0]  B=[10, 0, 0, 0]
  trial 10: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 11: A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  trial 12: A=[9, 2, 2, 2]  B=[2, 9, 9, 9]

**Rationale:** To robustly dissociate the Weighted Additive (WADD) model from Random Guessing, we construct a parametric gradient of expected value differences. WADD predicts a smooth psychometric function where choice probability for option A scales with the validity-weighted difference between the options' features. In contrast, Random Guessing predicts a flat 50% choice rate across all trials regardless of the evidence. We include completely dominating options (where WADD predicts near 100% deterministic choice), completely balanced options (where WADD predicts 50%), and compensatory trials where high-validity features pit against multiple lower-validity features to ensure subjects are actually integrating validities rather than just summing features.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
WADD predicts that the probability of choosing option A will vary systematically across different trial types depending on the validity-weighted differences in feature values (ranging from near 0 to near 1). In contrast, Random Guessing predicts a flat 50% choice rate for option A across all trial types. By calculating the variance of the choice rate of option A across the unique trial types, we can clearly distinguish the two models: WADD will produce a high variance, while Random Guessing will produce a variance close to zero (driven only by binomial noise).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for each unique trial pair
    trial_keys = data['option_a_ratings'].apply(tuple).astype(str) + "_" + data['option_b_ratings'].apply(tuple).astype(str)
    
    # Calculate the probability of choosing option A (response == 0) for each unique trial
    p_choose_A = (data['response'] == 0).groupby(trial_keys).mean()
    
    # Return the variance of these probabilities across different trial types
    return float(np.var(p_choose_A))

### RESULTS
- Predicted under pi_4 (simulated): 0.0004 (var=0.0001)
- Predicted under pi_2 (simulated): 0.1872 (var=0.0000)
- Observed on real data: 0.0005 (var=0.0001)

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
**Predicted under pi_2:** 0.3067 (var=0.1115)

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
**Predicted under pi_2:** 0.8900 (var=0.0829)

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
**Predicted under pi_2:** 0.0200 (var=0.0046)

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
**Predicted under pi_2:** 0.1450 (var=0.0059)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed data consistently and tightly aligns with the predictions of THEORY 1 (Random Guessing). Metrics targeting unweighted sum (Exp 1), validity-weighted expected value variance (Exp 2), Tallying (Exp 3), WADD (Exp 4), and Take-The-Best (Exp 5 & 6) all yield observed values around 0.5 (or near 0 variance in Exp 2). THEORY 2 (WADD) consistently fails to predict the observed data, predicting strong deviations from 0.5 that do not materialize. The robust failure of WADD, Equal Weight, Tallying, and TTB implies that subjects are either genuinely guessing or using a highly impoverished decision rule that washes out any systematic preference in these aggregated metrics. THEORY 1 is extremely successful, rendering THEORY 2 degenerate in its current form.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since pure Random Guessing (THEORY 1) explains the aggregated data perfectly, propose a brand-new theory that posits an extremely impoverished but non-zero intelligence strategy to see if it captures any residual variance. For example, a 'Single Random Feature' theory: on each trial, a subject randomly samples exactly one feature (ignoring validities) and chooses the option with the higher value on that feature, guessing if they are tied. This provides a theoretical bridge between pure noise and compensatory integration, testing if subjects extract any attribute-level information at all."
}
```

## Usage

```json
{
  "prompt_token_count": 5890,
  "candidates_token_count": 333,
  "total_token_count": 7387
}
```
