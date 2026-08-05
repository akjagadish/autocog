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
Decision-makers use a Non-linear Weighted Additive (WADD) strategy, computing an overall value for each option by summing the exponentiated validities of its positive features. The non-linear scaling of validities allows the strategy to smoothly interpolate between Tallying (treating features equally) and Take The Best (heavily weighting the most valid features).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = np.power(val, gamma)
    
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Strategy Mixture (Take-The-Best and Tallying with Tallying Bias): Decision-makers are heterogeneous in their strategy use, probabilistically switching between strategies. On any given choice, a subject has a probability 'p_ttb' of applying a one-reason lexicographic heuristic (Take-The-Best) and a probability '1 - p_ttb' of applying an equal-weight compensatory heuristic (Tallying). Empirical data suggests that people generally favor Tallying over Take-The-Best in these environments, so the probability of using Take-The-Best is bounded between 10% and 50%, ensuring the mixture slightly favors Tallying to better match observed aggregate choice probabilities.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Tallying Prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.8, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 8: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Non-linear Weighted Additive (WADD) strategy from a Strategy Mixture of Take-The-Best (TTB) and Tallying. By using four features with a specific validity profile (one very high validity, two moderately high validities, and one low validity), we can create trials where the mixture model and WADD make opposite predictions. For example, when Option A has the two moderately high validities and Option B has the highest and lowest validities, Tallying predicts a tie, so the TTB+Tallying mixture model defaults to TTB's prediction (Option B). However, WADD calculates that the sum of the two moderately high validities exceeds the sum of the highest and lowest validities, thus firmly predicting Option A. Other trials vary the number of winning features to test WADD's non-linear scaling against the fixed linear combination of the mixture model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In trials where Tallying produces a tie (e.g., A=[0,1,1,0] vs B=[1,0,0,1]), the Strategy Mixture defaults to Take-The-Best, which selects the option with the highest validity cue (feature 0). In contrast, the Non-linear Weighted Additive (WADD) strategy typically favors the option with the two moderately high validity cues (features 1 and 2), as their combined exponentiated sum usually outweighs the sum of the highest and lowest validity cues. By measuring the proportion of choices for the option with features 1 and 2 in these specific tie trials, we can cleanly separate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_trial_1 = (a_tuples == (0, 1, 1, 0)) & (b_tuples == (1, 0, 0, 1))
    is_trial_7 = (a_tuples == (1, 0, 0, 1)) & (b_tuples == (0, 1, 1, 0))
    
    wadd_choices = 0
    total_trials = 0
    
    if is_trial_1.sum() > 0:
        wadd_choices += (data.loc[is_trial_1, 'response'] == 0).sum()
        total_trials += is_trial_1.sum()
        
    if is_trial_7.sum() > 0:
        wadd_choices += (data.loc[is_trial_7, 'response'] == 1).sum()
        total_trials += is_trial_7.sum()
        
    if total_trials == 0:
        return 0.5
        
    return float(wadd_choices / total_trials)

### RESULTS
- Predicted under pi_3 (simulated): 0.5608 (var=0.0263)
- Predicted under pi_4 (simulated): 0.3725 (var=0.0117)
- Observed on real data: 0.4600 (var=0.0252)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.92, 0.88, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design isolates the Strategy Mixture (Take-The-Best + Tallying) from the Non-linear Weighted Additive (WADD) model by exploiting how they resolve trade-offs between cue rank, cue validity magnitude, and the number of winning cues. The Mixture model heavily relies on Tallying (which simply counts winning features without weighting them by validity) and breaks ties using TTB (which strictly looks at the highest-ranked cue). By contrast, WADD computes a compensatory sum of non-linearly scaled validities. We use a validity profile with one very high validity (0.95), two closely following high validities (0.92, 0.88), and two low validities (0.60, 0.55). In crucial trials (e.g., A wins on cues 1 and 4; B wins on cues 2 and 3), Tallying sees a tie, so the Mixture model defaults to TTB's choice (Option A). However, WADD calculates that the sum of the two moderately high validities for B strictly exceeds the sum of the highest and a low validity for A, consistently predicting Option B regardless of its non-linearity parameter (gamma). Additional trials pit a Tallying advantage against a WADD magnitude advantage to further decouple the predictions across the parameter space.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on trials 1 and 2 where the Strategy Mixture and WADD make diametrically opposite predictions. In these trials, Tallying results in a tie, causing the Mixture model to default to Take-The-Best. Conversely, WADD combines the two moderately high validities of the alternative option to strictly outweigh the highest validity cue. The metric calculates the proportion of choices aligning with the Mixture model's predictions on these critical trials.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 0]
    # Mixture predicts A (0), WADD predicts B (1)
    t1_match = (a_str == '10010') & (data['response'] == 0)
    
    # Trial 2: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 0, 1]
    # Mixture predicts B (1), WADD predicts A (0)
    t2_match = (a_str == '01100') & (data['response'] == 1)
    
    valid_trials = (a_str == '10010') | (a_str == '01100')
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float((t1_match.sum() + t2_match.sum()) / valid_trials.sum())

### RESULTS
- Predicted under pi_3 (simulated): 0.2075 (var=0.0236)
- Predicted under pi_4 (simulated): 0.6181 (var=0.0121)
- Observed on real data: 0.4475 (var=0.0246)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    preds = []
    for i in range(len(data)):
        p = -1
        for j in range(4):
            if a[i, j] > b[i, j]:
                p = 0
                break
            elif b[i, j] > a[i, j]:
                p = 1
                break
        preds.append(p)
        
    preds = np.array(preds)
    return float(np.mean(data['response'] == preds))
```

**Observed (real) value:** 0.4850 (var=0.0051)
**Predicted under pi_3:** 0.4531 (var=0.0128)
**Predicted under pi_4:** 0.4435 (var=0.0063)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    
    diff = a - b
    # The first index where features differ (since validities are strictly descending)
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_winner = np.where(ttb_val == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    
    valid_mask = (a_wins != b_wins)
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    
    # Focus only on trials where Tallying and Take The Best make strictly opposite predictions
    conflict_mask = valid_mask & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_matches = np.sum(responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(ttb_matches / np.sum(conflict_mask))

```

**Observed (real) value:** 0.3844 (var=0.0082)
**Predicted under pi_3:** 0.3056 (var=0.0217)
**Predicted under pi_4:** 0.3558 (var=0.0123)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ties = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        a_w = sum(1 for x, y in zip(a, b) if x > y)
        b_w = sum(1 for x, y in zip(a, b) if y > x)
        ties.append(a_w == b_w)
        
    tie_data = data[ties]
    if len(tie_data) == 0:
        return 0.5
        
    # In tie trials, A is designed to have higher-validity features than B.
    # Tallying predicts 50% A (response == 0) because the tallies are tied.
    # WADD predicts > 50% A because A's WADD score is higher.
    return float(np.mean(tie_data['response'] == 0))
```

**Observed (real) value:** 0.5667 (var=0.0123)
**Predicted under pi_3:** 0.8056 (var=0.0141)
**Predicted under pi_4:** 0.6022 (var=0.0106)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Project option_a_ratings to string for hashability and comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    
    # Trial 1 is uniquely identified by Option A having exactly these ratings
    t1_mask = a_str == '00111'
    
    if t1_mask.sum() == 0:
        return 0.0
        
    # Return the proportion of times Option B was chosen on this trial
    return float(data.loc[t1_mask, 'response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0450)
**Predicted under pi_3:** 0.7300 (var=0.0663)
**Predicted under pi_4:** 0.3567 (var=0.0383)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, both theories frequently bound the observed data, with the true metric falling somewhere in between. Theory 2 (Strategy Mixture of TTB and Tallying) is generally closer to the observed data in Experiments 4 and 5, though it struggles with Experiment 6. Theory 1 (Non-linear WADD) frequently overestimates or underestimates the effects (e.g., Experiments 2, 5, and 6) because it assumes a rigidly deterministic integration of exponentiated validities that does not align with the more bounded or heuristic-like human behavior. Because neither theory perfectly captures the robust intermediate values found in experiments like 1, 2, and 6, and since Theory 1's non-linear WADD is systematically mispredicting tie-breaking behavior, Theory 1 should be replaced with a novel framework.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Sequential Evidence Accumulation' (or Threshold-based) theory. Instead of computing a static non-linear sum (WADD) or probabilistically switching between two discrete heuristics (Mixture), this theory posits that decision-makers inspect cues sequentially in order of validity. Each discriminating cue adds evidence to an accumulator. If the evidence difference between options reaches a certain threshold, the search stops and a choice is made; if all cues are exhausted, the decision relies on the accumulated evidence with some noise. This provides a natural interpolation between TTB (low threshold) and Tallying/WADD (high threshold) based on a single unified cognitive mechanism rather than a probabilistic mixture."
}
```

## Usage

```json
{
  "prompt_token_count": 6659,
  "candidates_token_count": 354,
  "total_token_count": 7808
}
```
