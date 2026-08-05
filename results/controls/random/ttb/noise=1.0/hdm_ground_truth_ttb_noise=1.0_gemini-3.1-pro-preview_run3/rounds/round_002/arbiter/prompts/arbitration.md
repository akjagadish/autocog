# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) heuristic with Softmax Choice: People evaluate features sequentially in descending order of their subjective validities. They stop at the first feature that discriminates between the two options. Instead of a deterministic choice, they choose the option probabilistically using a softmax function over the values of that single discriminating feature, scaled by a sensitivity parameter beta. If no features discriminate, they guess. Response noise is additionally modeled via a global lapse rate epsilon.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    order = np.argsort(-validities)
    
    beta = float(parameters['beta'])
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] != b[idx]:
            scores = np.array([a[idx], b[idx]])
            z = beta * (scores - scores.max())
            e = np.exp(z)
            p_core = e / e.sum()
            break
            
    epsilon = float(parameters['epsilon'])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_2_1
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
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

    # Weighted sum per option (dot product with per-feature subjective weights).
    scores = stim @ (validities * w)

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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[5, 10, 10, 10]  B=[6, 0, 0, 0]
  trial 2: A=[10, 0, 0, 0]  B=[9, 10, 10, 10]
  trial 3: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 4: A=[5, 5, 5, 5]  B=[5, 6, 0, 0]
  trial 5: A=[7, 7, 0, 0]  B=[7, 6, 10, 10]
  trial 6: A=[8, 8, 8, 0]  B=[8, 8, 7, 10]

**Rationale:** To dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use cardinal ratings and exploit TTB's non-compensatory stopping rule. We construct trials where one option has a small advantage on the highest-validity cue (causing TTB to stop and choose it), while the other option has massive advantages on all lower-validity cues (causing WADD to choose the latter due to compensatory weighted sums). We also include trials where the highest-validity cues are tied, forcing TTB to evaluate the next cue, where again a small advantage dictates TTB's choice but is overridden by remaining features in WADD.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the proportion of trials where the subject's choice matches the deterministic prediction of the Take The Best (TTB) heuristic. Because the experimental design explicitly pits TTB against WADD by creating trials where the first discriminating cue favors one option while the remaining cues strongly favor the other, TTB and WADD will systematically predict opposite choices on 5 out of the 6 trial types. Consequently, data simulated under TTB will yield a match rate > 0.5 (albeit noisy due to parameter ranges), whereas data simulated under WADD will yield a match rate significantly below 0.5. This produces a stark and reliable contrast between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_diff_idx = nonzero[0]
            ttb_choices.append(0 if d[first_diff_idx] > 0 else 1)
        else:
            ttb_choices.append(-1)
            
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))

### RESULTS
- Predicted under pi_3 (simulated): 0.5273 (var=0.0021)
- Predicted under pi_2_1 (simulated): 0.3977 (var=0.0132)
- Observed on real data: 0.4979 (var=0.0021)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[6, 1, 1, 1]  B=[5, 10, 10, 10]
  trial 2: A=[8, 8, 8, 8]  B=[9, 0, 0, 0]
  trial 3: A=[5, 6, 1, 1]  B=[5, 5, 10, 10]
  trial 4: A=[5, 5, 10, 0]  B=[5, 5, 9, 10]
  trial 5: A=[10, 0, 0, 0]  B=[0, 10, 10, 10]
  trial 6: A=[7, 7, 7, 7]  B=[8, 6, 6, 6]

**Rationale:** This design systematically dissociates the Weighted Additive (WADD) strategy from Take The Best (TTB). TTB relies on a non-compensatory stopping rule, making its decision based entirely on the first feature (ordered by validity) that discriminates between the two options. WADD, on the other hand, integrates all features in a compensatory manner, weighting them by their validities. By setting a high rating_max (10), we can create scenarios where one option holds a minor advantage on the highest-validity feature (causing TTB to choose it), while the other option possesses massive advantages on all remaining lower-validity features (overwhelming the small deficit and causing WADD to choose it). We also include trials where the highest-validity features are tied, forcing TTB to stop at the second or third feature, where again its single-feature focus leads to an opposite prediction compared to WADD's holistic integration.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Take The Best (TTB) against Weighted Additive (WADD) by creating trials where the highest-validity discriminating feature points to one option, but the sum of the remaining lower-validity features strongly favors the other option. This metric dynamically computes the TTB prediction for each trial (by finding the first feature in validity-sorted order that discriminates between the two options) and calculates the proportion of subject responses that align with this prediction. TTB simulated data will yield values significantly above 0.5, whereas WADD simulated data will yield values significantly below 0.5, maximizing the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def ttb_choice(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            elif val_a < val_b:
                return 1
        return -1
        
    ttb_preds = [ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    agreements = [1 if p == r else 0 for p, r in zip(ttb_preds, data['response']) if p != -1]
    
    return float(np.mean(agreements)) if len(agreements) > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.5290 (var=0.0025)
- Predicted under pi_2_1 (simulated): 0.3810 (var=0.0140)
- Observed on real data: 0.5000 (var=0.0030)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5042 (var=0.0050)
**Predicted under pi_3:** 0.5746 (var=0.0109)
**Predicted under pi_2_1:** 0.5188 (var=0.0101)

### Experiment 4
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5156 (var=0.0035)
**Predicted under pi_3:** 0.5225 (var=0.0045)
**Predicted under pi_2_1:** 0.5500 (var=0.0079)

### Experiment 5
**Design**
  A=[2, 7, 7, 7]  B=[10, 6, 6, 6]
  A=[10, 2, 2, 2]  B=[2, 3, 3, 3]
  A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[0, 5, 5, 5]  B=[10, 0, 0, 1]
  A=[10, 0, 0, 1]  B=[0, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Count how many features option A strictly wins and option B strictly wins
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Only consider trials where one option wins on strictly more features than the other
    mask = a_wins != b_wins
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with more feature wins
    # response == 0 means A was chosen, response == 1 means B was chosen
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    actual = data.loc[mask, 'response']
    
    # Return the proportion of trials where the subject's choice aligns with the Tallying prediction
    return float((tally_pred == actual).mean())
```

**Observed (real) value:** 0.5088 (var=0.0034)
**Predicted under pi_3:** 0.4456 (var=0.0094)
**Predicted under pi_2_1:** 0.5081 (var=0.0067)

### Experiment 6
**Design**
  A=[1, 1, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  A=[10, 4, 4, 4, 4]  B=[0, 5, 5, 5, 5]
  A=[10, 10, 0, 0, 5]  B=[0, 0, 10, 10, 5]
  A=[0, 7, 7, 7, 7]  B=[10, 6, 6, 0, 0]
  A=[10, 0, 0, 0, 0]  B=[0, 4, 4, 5, 0]
  A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].tolist()
    # Trial 1: A=[1, 1, 6, 6, 6]
    is_t1 = np.array([a[0] == 1 for a in a_ratings])
    # Trial 2: A=[10, 4, 4, 4, 4]
    is_t2 = np.array([a[0] == 10 and a[1] == 4 for a in a_ratings])
    
    resp_A = (data['response'] == 0).values
    
    p_A_t1 = resp_A[is_t1].mean() if is_t1.sum() > 0 else 0.5
    p_A_t2 = resp_A[is_t2].mean() if is_t2.sum() > 0 else 0.5
    
    return float(p_A_t1 - p_A_t2)
```

**Observed (real) value:** -0.0100 (var=0.0271)
**Predicted under pi_3:** -0.1850 (var=0.0433)
**Predicted under pi_2_1:** -0.2162 (var=0.0574)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric values are strikingly close to chance (0.5 for proportion metrics, 0.0 for difference metrics). Both Take The Best (pi_3) and WADD (pi_2_1) predict systematic deviations from chance on these carefully designed conflict trials. For instance, in Experiments 1 and 2, WADD predicts match rates around 0.38-0.39, while TTB predicts ~0.53, but the observed data is exactly 0.50. In Experiment 6, both models predict a strong negative difference (-0.18 to -0.21), but the observed difference is -0.01. The consistent finding of chance-level performance across different trial types and metrics strongly suggests that subjects are not systematically employing either TTB or WADD in the way these models assume. Instead, behavior is dominated by noise, random guessing, or a strategy that perfectly cancels out across the population (like an equal mix of TTB and WADD, or a completely different approach like Satisficing with very low thresholds).",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a completely new theory that accounts for the near-chance performance across all these conflict trials. A strong candidate would be a 'Random Choice / High Noise' baseline, or a 'Similarity Heuristic' where subjects treat options as practically equivalent if they do not see a massive, overwhelmingly obvious dominance, leading to random guessing on complex conflict trials. Alternatively, an 'Equal Weight' or 'Tallying' model with extremely high temperature (beta near 0) could be formalized to explain why choices collapse to 50/50 when features conflict in magnitude and direction."
}
```

## Usage

```json
{
  "prompt_token_count": 6021,
  "candidates_token_count": 383,
  "total_token_count": 7416
}
```
