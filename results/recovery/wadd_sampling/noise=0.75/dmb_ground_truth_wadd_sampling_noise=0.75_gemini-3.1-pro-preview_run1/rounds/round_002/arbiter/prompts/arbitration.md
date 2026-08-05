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
Generalized Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a compensatory weighted sum of their features. However, instead of using raw cue validities as weights, individuals scale the validities non-linearly using a parameter gamma. This scaling allows the decision process to smoothly interpolate between Tallying (gamma=0, where all cues are weighted equally), standard WADD (gamma=1), and Take The Best (gamma -> infinity, where the most valid cue dominates). The final choice is made probabilistically via a softmax function over the computed option values, mixed with a random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Random Guessing (Zero-Intelligence) Theory: Without trial-by-trial feedback, subjects find the multi-attribute binary feature task too cognitively demanding or unengaging. As a result, they do not attempt to integrate the cue validities or compare the feature values. Instead, they simply guess uniformly at random on every trial.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    # The model assumes pure random guessing, ignoring stimulus and history entirely.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Generalized Weighted Additive (WADD) Theory from the Random Guessing Theory, we need to present trials where WADD predicts a systematic, varying preference across trials while Random Guessing predicts a flat 50% choice probability everywhere. We use four features with evenly spaced validities. The trials range from complete dominance (where WADD predicts near 100% preference) to partial conflicts (where WADD predicts moderate preferences depending on the gamma and beta parameters). If subjects are merely guessing randomly, all trials will yield ~50% choice rates. If they are evaluating features as WADD predicts, their choice probabilities will systematically track the difference in the non-linearly weighted feature sums.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the higher validity-weighted sum of features. For the Random Guessing theory, subjects choose A or B randomly, so this metric will be approximately 0.5. For the WADD theory, subjects systematically favor the option with the higher weighted sum (scaled by gamma and mixed with lapse, but still directionally consistent), so this metric will be significantly higher than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    
    mask = score_a != score_b
    if not np.any(mask):
        return 0.5
        
    expected = (score_b > score_a).astype(int)
    actual = data['response'].values
    
    return float(np.mean(expected[mask] == actual[mask]))

### RESULTS
- Predicted under pi_3 (simulated): 0.7469 (var=0.0099)
- Predicted under pi_4 (simulated): 0.4940 (var=0.0020)
- Observed on real data: 0.5743 (var=0.0037)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Generalized Weighted Additive (WADD) Theory from the Random Guessing Theory, we use a 5-feature design with a wide spread of validities. Random Guessing predicts a flat 50% choice probability across all trials, regardless of the features presented. WADD predicts systematic deviations from 50% based on the weighted sum of the features. We include trials ranging from complete dominance (where WADD predicts near 100% preference for one option) to complex trade-offs (e.g., the highest validity cue versus multiple lower validity cues). If subjects are merely guessing, their choice rates will remain at 50% across all these diverse trial types. If they follow a compensatory or heuristic-like strategy captured by WADD, their choice probabilities will systematically track the differences in feature values and validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Random Guessing predicts that subjects choose completely at random, meaning the proportion of choosing option A (or B) for any specific trial type will tightly hover around 0.5. The WADD model, however, predicts systematic preferences driven by feature values and validities, meaning choice proportions will reliably deviate from 0.5 (often approaching 0 or 1 on easy trials like A dominating B). By calculating the mean absolute deviation of choice proportions from 0.5 across all unique trial types and subjects, we directly measure this systematicity. Random guessing will yield a low value (driven only by binomial noise, ~0.14 for 12 reps), while WADD will yield a significantly higher value.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    keys = [tuple(a) + tuple(b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_key': keys,
        'response': data['response']
    })
    grouped = df.groupby(['subject_id', 'trial_key'])['response'].mean()
    return float((grouped - 0.5).abs().mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.3000 (var=0.0064)
- Predicted under pi_4 (simulated): 0.1119 (var=0.0012)
- Observed on real data: 0.1371 (var=0.0006)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    valid_trials = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB compares cues in order 0 to n_features-1
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_matches += 1
            valid_trials += 1
            
    return float(ttb_matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4737 (var=0.0046)
**Predicted under pi_3:** 0.4486 (var=0.0430)
**Predicted under pi_4:** 0.4964 (var=0.0023)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    tally_a = np.sum(a_ratings > b_ratings, axis=1)
    tally_b = np.sum(b_ratings > a_ratings, axis=1)
    valid = tally_a != tally_b
    if not np.any(valid):
        return 0.5
    pred = np.where(tally_a > tally_b, 0, 1)
    return float((pred[valid] == data['response'].values[valid]).mean())
```

**Observed (real) value:** 0.5479 (var=0.0020)
**Predicted under pi_3:** 0.5290 (var=0.0541)
**Predicted under pi_4:** 0.5008 (var=0.0022)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    return wadd_choices / total if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0233)
**Predicted under pi_3:** 0.8067 (var=0.0433)
**Predicted under pi_4:** 0.4992 (var=0.0084)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify Trial 3: A=[0, 0, 1, 1, 1], B=[1, 1, 0, 0, 0]
    # Tallying favors A (3 vs 2), meaning response should be closer to 0.
    # WADD with gamma > 1 favors B (due to the 0.95 and 0.85 validities), meaning response closer to 1.
    is_trial_3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Identify Trial 4: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 1]
    # Tallying favors B (2 vs 3), meaning response should be closer to 1.
    # WADD with gamma > 1 favors A (due to the 0.95 validity), meaning response closer to 0.
    is_trial_4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    
    resp_3 = data.loc[is_trial_3, 'response'].mean()
    resp_4 = data.loc[is_trial_4, 'response'].mean()
    
    if pd.isna(resp_3):
        resp_3 = 0.5
    if pd.isna(resp_4):
        resp_4 = 0.5
        
    # Tallying: resp_4 (high) - resp_3 (low) -> Positive
    # WADD: resp_4 (low) - resp_3 (high) -> Negative
    return float(resp_4 - resp_3)
```

**Observed (real) value:** -0.0025 (var=0.0589)
**Predicted under pi_3:** -0.2850 (var=0.2349)
**Predicted under pi_4:** -0.0212 (var=0.0279)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data shows very weak signal, hovering close to 0.5. Theory 2 (Random Guessing) predicts exactly 0.5 everywhere, failing to capture the slight but consistent directional preferences observed in Experiments 1 (0.57) and 4 (0.54). Theory 1 (WADD) captures the direction of these preferences but consistently overestimates their magnitude by a wide margin (e.g., predicting 0.74 in Exp 1 and 0.80 in Exp 5). Furthermore, in conflict trials (Exp 5 and Exp 6) where WADD strongly favors one option and equal-weighting (Tallying) favors another, subjects do not show the strong WADD-predicted preference, often landing near 0.5 or slightly favoring the Tallying option. Pure random guessing is a degenerate null model, while WADD assumes too much compensatory integration.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Noisy Tallying' or 'Random Cue Sampling' theory. Instead of purely guessing or performing complex weighted additions, subjects might simply count the number of positive features for each option (Tallying) or randomly sample a single feature to base their decision on, combined with a high lapse rate. This would explain the very weak but non-zero signal that aligns more with unweighted feature counts than with validity-weighted sums."
}
```

## Usage

```json
{
  "prompt_token_count": 5850,
  "candidates_token_count": 323,
  "total_token_count": 7425
}
```
