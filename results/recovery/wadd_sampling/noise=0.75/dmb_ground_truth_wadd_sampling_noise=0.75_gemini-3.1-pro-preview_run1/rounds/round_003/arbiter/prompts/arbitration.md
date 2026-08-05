# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_5
High-Lapse Tallying Theory: Subjects find the multi-attribute binary feature task cognitively demanding, leading to a very high rate of random guessing. However, when they do engage with the stimulus, they rely on a simple unweighted tally of positive features (Tallying) rather than integrating complex cue validities. This produces a very weak but non-zero behavioral signal that slightly favors options with a higher simple count of positive features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies for the engaged decision process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Mix with a high lapse rate (random guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 5: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Generalized WADD from High-Lapse Tallying, we exploit two main differences: the integration of cue validities and the base lapse rate. High-Lapse Tallying strictly counts positive features (ignoring validities) and assumes a high random guessing rate (epsilon >= 0.7), capping its maximum predicted choice probability around 65%. Generalized WADD accounts for validities (scaled non-linearly) and allows for much lower lapse rates, thus predicting extreme choice probabilities on dominant trials. We include trials where Tallying predicts a tie but WADD predicts a strong preference (e.g., A=[1,0,0,0] vs B=[0,1,0,0]), trials where Tallying prefers one option but WADD prefers the other due to high-validity cues (e.g., A=[1,0,0,0] vs B=[0,1,1,1]), and dominance trials to expose the difference in maximum choice probabilities (e.g., A=[1,1,1,0] vs B=[0,0,0,1]).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
We isolate trials where Option A has at least as many positive features as Option B, AND Option A possesses the most valid feature (cue 1). These correspond to trials 2, 3, and 6. Under High-Lapse Tallying, these trials yield a weak preference for A or a tie, heavily diluted by random guessing, resulting in a choice probability for A near 50-55%. Under Generalized WADD, Option A is strongly preferred due to the integration of the high-validity cue and lower lapse rates, yielding a much higher choice probability for A.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    
    mask = (a_sums >= b_sums) & (a_first == 1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    return float((subset['response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8029 (var=0.0117)
- Predicted under pi_5 (simulated): 0.5246 (var=0.0066)
- Observed on real data: 0.5300 (var=0.0073)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate High-Lapse Tallying from Generalized WADD, we exploit their differences in both feature weighting and base lapse rates. High-Lapse Tallying strictly counts positive features (ignoring validities) and assumes a high random guessing rate (capping its maximum choice probability around 65%). Generalized WADD accounts for validities (scaled non-linearly) and allows for much lower lapse rates, thus predicting extreme choice probabilities on dominant trials. We include: (1) trials where Tallying predicts a tie but WADD predicts a strong preference based on validity differences, (2) conflict trials where Tallying prefers one option because it has more positive features but WADD prefers the other due to high-validity cues, and (3) a pure dominance trial to expose the difference in maximum predicted choice probabilities due to the distinct epsilon bounds.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric isolates the pure dominance trial (Trial 5). High-Lapse Tallying enforces a very high lapse rate (epsilon in [0.7, 1.0]), which strictly bounds its maximum choice probability for the dominant option to ~0.65. In contrast, Generalized WADD allows for much lower lapse rates (epsilon in [0.0, 0.5]) and will predict a choice probability for the dominant option near 1.0. By directly computing the empirical probability of choosing the dominant option on this specific trial, we obtain a stark, structural dissociation between the two theories' bounds on decision determinism.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 5, where Option A is strictly dominant (all 1s) and Option B has all 0s.
    is_trial_5 = data['option_a_ratings'].apply(lambda x: sum(x) == 5)
    
    if not is_trial_5.any():
        return 0.5
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float((data.loc[is_trial_5, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8769 (var=0.0133)
- Predicted under pi_5 (simulated): 0.5831 (var=0.0206)
- Observed on real data: 0.5877 (var=0.0189)

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
**Predicted under pi_5:** 0.4398 (var=0.0056)

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
**Predicted under pi_5:** 0.5650 (var=0.0035)

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
**Predicted under pi_5:** 0.4483 (var=0.0116)

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
**Predicted under pi_5:** 0.0650 (var=0.0244)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.5743 (var=0.0037)
**Predicted under pi_3:** 0.7469 (var=0.0099)
**Predicted under pi_5:** 0.5471 (var=0.0035)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.1371 (var=0.0006)
**Predicted under pi_3:** 0.3000 (var=0.0064)
**Predicted under pi_5:** 0.1269 (var=0.0010)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 8 experiments, the observed data consistently shows extremely weak choice preferences, often hovering just above chance (0.50 - 0.58). Theory 2 (High-Lapse Tallying) accurately captures this macroscopic pattern because its high lapse rate strongly limits the maximum choice probability, matching the observed data closely on almost all metrics. Theory 1 (Generalized WADD) consistently overpredicts the strength of subjects' preferences (e.g., predicting 0.80-0.87 when the true values are ~0.53-0.58) because it allows for low lapse rates and integrates validities too deterministically. While Theory 2 fits well, it assumes pure tallying. The data clearly rejects strong validity-weighted integration, but a high-lapse heuristic that integrates some validity information (like a high-lapse Take-The-Best) might capture the subtle deviations where Theory 2 is slightly off.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'High-Lapse Take-The-Best (TTB)' or 'Noisy Lexicographic' theory. This theory should posit that subjects find the task tedious and guess randomly on a large majority of trials (high epsilon, e.g., > 0.6). However, when they do engage, instead of tallying all features (as in Theory 2) or integrating all validities (as in the old Theory 1), they simply look at the single most valid cue that discriminates the options and choose based on that. This maintains the high noise floor required to fit the data while testing a lexicographic rather than compensatory mechanism for the engaged trials."
}
```

## Usage

```json
{
  "prompt_token_count": 7074,
  "candidates_token_count": 374,
  "total_token_count": 8345
}
```
