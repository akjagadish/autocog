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
Tallying with Recency/Right-Most Tie-Breaker: Decision makers primarily evaluate options by tallying the number of features where one option strictly dominates the other, ignoring feature validities (equal weighting). If one option has more winning features, it is chosen. However, if the feature wins are tied, individuals do not simply guess. Instead, they rely on a 'recency' or 'right-most' bias, breaking the tie based solely on the last feature in the array. Responses are subject to softmax noise over the resulting scores and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Tie-breaking mechanism: Recency/Right-most feature bias
    if a_wins == b_wins:
        if a[-1] > b[-1]:
            a_wins += 1.0
        elif b[-1] > a[-1]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Recency-Weighted Tallying: Decision makers evaluate options by computing a weighted tally of feature wins. Due to recency effects in working memory, features presented later (further to the right) are more salient and receive exponentially increasing weights. Because the growth rate of these weights is bounded, the heuristic strictly preserves standard tallying when one option has more feature wins than the other. However, when the number of feature wins is tied, the exponentially increasing weights naturally break the tie in favor of the option that wins on the most recent (right-most) features, providing a continuous and elegant mechanism for tie-breaking without invoking discontinuous rules.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    recency_base = float(parameters['recency_base'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Weights exponentially increasing from left to right to model recency in working memory.
    # For recency_base in [1.01, 1.3], the sum of any k weights is strictly less 
    # than the sum of any k+1 weights, preserving the strict tallying property for unequal wins.
    w = recency_base ** np.arange(n_features)
    
    # Weighted tallying of strict feature-wise wins
    a_wins = float(np.sum(w * (a > b)))
    b_wins = float(np.sum(w * (b > a)))
    
    scores = np.array([a_wins, b_wins])
    
    # Softmax choice with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.55, 0.65, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 1]  B=[1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1]  B=[0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on trials where both the overall number of strict feature wins AND the right-most feature are tied between the two options. The Advocated theory predicts a 50-50 guess on these trials because its tie-breaker fails. The Competing theory resolves the tie using the exponentially weighted sum of the remaining features, systematically predicting a specific option. This metric calculates the proportion of choices that align with the Competing theory's secondary tie-breaking predictions.

Source:
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total_trials = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 1, 1, 1) and b == (1, 1, 0, 1):
            match_count += (1 if resp == 0 else 0)
            total_trials += 1
        elif a == (0, 1, 0, 1) and b == (1, 0, 0, 1):
            match_count += (1 if resp == 0 else 0)
            total_trials += 1
        elif a == (0, 1, 0, 1) and b == (0, 0, 1, 1):
            match_count += (1 if resp == 1 else 0)
            total_trials += 1
            
    if total_trials == 0:
        return 0.5
    return match_count / total_trials

### RESULTS
- Predicted under pi_3 (simulated): 0.4644 (var=0.0102)
- Predicted under pi_4 (simulated): 0.7467 (var=0.0261)
- Observed on real data: 0.6044 (var=0.0055)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.87, 0.89, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1]  B=[0, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0]  B=[0, 1, 0, 0]
  trial 12: A=[0, 1, 0, 0]  B=[1, 0, 1, 0]
  trial 13: A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 15: A=[1, 1, 1, 1]  B=[0, 1, 0, 0]
  trial 16: A=[0, 1, 0, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses exclusively on Trials 3, 6, and 16. In these trials, the strict feature tally results in a tie, and the last (right-most) feature is identical for both options. The Competing Theory's tie-breaker thus fails and it predicts random guessing (50% choice rate). The Advocated Theory, however, applies exponentially increasing weights to more recent features, leading to a strict preference for one option over the other based on the non-tied features. The metric computes the proportion of choices that align with the Advocated Theory's preference on these three specific trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_t3 = (a_tuples == (1, 0, 1, 0)) & (b_tuples == (0, 1, 1, 0))
    is_t6 = (a_tuples == (0, 0, 1, 0)) & (b_tuples == (0, 1, 0, 0))
    is_t16 = (a_tuples == (0, 1, 0, 1)) & (b_tuples == (0, 0, 1, 1))
    
    t3_score = (data[is_t3]['response'] == 1).sum()
    t6_score = (data[is_t6]['response'] == 0).sum()
    t16_score = (data[is_t16]['response'] == 1).sum()
    
    total_count = is_t3.sum() + is_t6.sum() + is_t16.sum()
    
    if total_count == 0:
        return 0.5
    
    return float(t3_score + t6_score + t16_score) / total_count


### RESULTS
- Predicted under pi_3 (simulated): 0.4856 (var=0.0170)
- Predicted under pi_4 (simulated): 0.7089 (var=0.0245)
- Observed on real data: 0.4000 (var=0.0072)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    target_trials_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tally wins
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # TTB winner (validities: 0.95, 0.72, 0.73, 0.55 -> cue order: 0, 2, 1, 3)
        ttb_winner = None
        for j in [0, 2, 1, 3]:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
        
        # Target trials: TTB has a strict preference, but Tallying ties
        if ttb_winner is not None and a_wins == b_wins:
            target_trials_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if target_trials_count == 0:
        return 0.5
        
    return ttb_match_count / target_trials_count
```

**Observed (real) value:** 0.2033 (var=0.0206)
**Predicted under pi_3:** 0.1642 (var=0.0128)
**Predicted under pi_4:** 0.2117 (var=0.0211)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 9 where Tallying and TTB make opposite strict predictions
    # Trial 9: A=[0, 1, 0, 1], B=[0, 0, 1, 0]
    is_trial_9 = (
        data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1)) &
        data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0))
    )
    trial_data = data[is_trial_9]
    if len(trial_data) == 0:
        return 0.5
    
    # Return the proportion of times the subject chose A (response == 0)
    return float(np.mean(trial_data['response'] == 0))
```

**Observed (real) value:** 0.8200 (var=0.0354)
**Predicted under pi_3:** 0.8833 (var=0.0258)
**Predicted under pi_4:** 0.8767 (var=0.0187)

### Experiment 5
**Design**
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def check_agreement(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins == b_wins and a[-1] != b[-1]:
            expected_resp = 1 if b[-1] > a[-1] else 0
            return 1.0 if row['response'] == expected_resp else 0.0
        return np.nan
        
    agreements = data.apply(check_agreement, axis=1)
    agreements = agreements.dropna()
    if len(agreements) == 0:
        return 0.5
    return float(agreements.mean())
```

**Observed (real) value:** 0.8200 (var=0.0420)
**Predicted under pi_3:** 0.8667 (var=0.0244)
**Predicted under pi_4:** 0.7933 (var=0.0528)

### Experiment 6
**Design**
  A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_target_trial(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        # We only care about trials where tallying results in a tie,
        # AND the last feature is not tied.
        return (a_wins == b_wins) and (a[-1] != b[-1])
    
    def choice_matches_last_feature(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        if a[-1] > b[-1]:
            return row['response'] == 0
        else:
            return row['response'] == 1
            
    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.5
        
    matches = target_data.apply(choice_matches_last_feature, axis=1)
    return float(matches.mean())
```

**Observed (real) value:** 0.8178 (var=0.0285)
**Predicted under pi_3:** 0.8656 (var=0.0111)
**Predicted under pi_4:** 0.7600 (var=0.0231)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (pi_3 - Tallying with Right-Most Tie-Breaker) consistently outperforms Theory 2 (pi_4 - Recency-Weighted Tallying). In Experiments 1 and 2, which specifically isolate trials where tallying results in a tie and the right-most feature is also tied, Theory 2 predicts a strong preference (e.g., > 70%) driven by exponentially increasing weights on the remaining right-side features. However, the observed data in Experiment 2 shows a choice fraction of 0.40, completely contradicting Theory 2's prediction of 0.71. Meanwhile, Theory 1 predicts near 0.50 for these double-tie trials, which is much closer to the observed ~0.40-0.60 range. This indicates that the strict recency-weighted tallying mechanism of Theory 2 is degenerate and fails to capture how humans actually break ties when the right-most feature provides no signal. Furthermore, looking at the validities in Experiment 2 (0.95, 0.87, 0.89, 0.55), the option that wins on the highest validity feature (index 0) was chosen 60% of the time (metric = 0.40 for the alternative), suggesting that validities play a role in tie-breaking.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory: 'Tallying with Validity-Based Tie-Breaker'. The primary mechanism remains equal-weight tallying of strict feature wins (ignoring validities). However, instead of a right-most spatial bias or recency weights, ties in the tally are broken by relying on the single feature with the highest objective validity. If the options also tie on the most valid feature, it moves to the second most valid, and so on (essentially falling back to Take-The-Best only for tie-breaking). This differs from both pi_3 (which uses spatial recency to break ties) and pi_4 (which uses continuous exponential recency weights), and can better explain systematic deviations from 0.50 in trials where right-most tie-breakers fail but validity asymmetries exist."
}
```

## Usage

```json
{
  "prompt_token_count": 7559,
  "candidates_token_count": 500,
  "total_token_count": 9582
}
```
