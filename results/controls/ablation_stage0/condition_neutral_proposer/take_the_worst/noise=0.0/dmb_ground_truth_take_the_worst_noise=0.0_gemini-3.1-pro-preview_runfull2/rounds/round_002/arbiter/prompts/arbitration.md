# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Top-N Tallying (Truncated Tallying): People simplify complex decisions by restricting their attention to only the most valid cues and ignoring the rest. Within this considered subset of top cues, they abandon complex compensatory weighting and instead simply tally the number of features favoring each option. If the tallies are tied, they guess. This bridges the gap between fully compensatory WADD and unweighted Tallying by selectively ignoring low-validity features to save cognitive effort, while still avoiding the cognitive cost of cardinal weighting among the considered features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Top-N Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # Determine k: the number of top features to consider
    k_prop = float(parameters["k_prop"])
    k = max(1, int(round(k_prop * n_features)))
    
    # Get indices of top k validities (descending order, stable tie-breaking)
    cue_order = np.argsort(-val, kind="stable").tolist()
    top_k_indices = cue_order[:k]
    
    a, b = stim[0], stim[1]
    
    # Tally only on the top k features
    a_top = a[top_k_indices]
    b_top = b[top_k_indices]
    
    a_wins = float(np.sum(a_top > b_top))
    b_wins = float(np.sum(b_top > a_top))
    
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To dissociate Top-N Tallying (Theory 1) from standard Tallying (Theory 2), we exploit the fact that Theory 1 ignores the lowest validity features (with k_prop in [0.8, 1.0], a 5-feature design means Theory 1 often ignores the 5th feature). By constructing trials where the top 4 features result in a tie (or a 1-point difference) but the 5th feature breaks the tie (or creates one), we generate sharp qualitative differences. Theory 2 will reliably use the 5th feature to break ties or form ties, whereas Theory 1 will guess when the top 4 tie, or stick with its top-4 preference despite the 5th feature.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the top 4 features favor one option, but the 5th feature brings the total sum to a tie. For Standard Tallying, these trials are perfect ties across all 5 features, so the model predicts guessing (choice probability exactly 0.5). For Top-N Tallying, subjects who restrict their attention to the top 4 features will strictly prefer the option favored by the top 4, ignoring the 5th feature that creates the tie. Thus, Top-N Tallying predicts a significantly higher match rate with the top-4 favored option.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    top4_a = a_ratings[:, :4].sum(axis=1)
    top4_b = b_ratings[:, :4].sum(axis=1)
    
    all5_a = a_ratings.sum(axis=1)
    all5_b = b_ratings.sum(axis=1)
    
    mask = (top4_a != top4_b) & (all5_a == all5_b)
    if not np.any(mask):
        return 0.5
        
    a_top4_better = top4_a[mask] > top4_b[mask]
    chose_a = (data['response'].values[mask] == 0)
    
    match = (a_top4_better == chose_a)
    return float(np.mean(match))

### RESULTS
- Predicted under pi_4 (simulated): 0.6822 (var=0.0494)
- Predicted under pi_2 (simulated): 0.4911 (var=0.0087)
- Observed on real data: 0.1289 (var=0.0168)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** To effectively dissociate Tallying (Theory 1) from Top-N Tallying (Theory 2), we use a 6-feature design. Theory 2 uses a parameter k_prop in [0.8, 1.0], which means it will typically truncate its consideration to the top 5 features (since round(0.8 * 6) = 5). By manipulating the 6th (lowest validity) feature, we can create sharp divergences. We include trials where the top 5 features result in a tie, but the 6th feature breaks it (Theory 1 will have a strict preference, while Theory 2 will guess). We also include trials where the top 5 features slightly favor one option, but the 6th feature favors the other, resulting in a tie across all 6 features (Theory 1 will guess, while Theory 2 will have a strict preference).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in choice rates between two specific sets of trials. In the first set (Trials 1, 4, 6, 8), the top 5 features result in a tie, but the 6th feature breaks it. Theory 1 (Tallying across all features) will show a strict preference, while Theory 2 (Top-N Tallying) will guess. In the second set (Trials 2, 3, 5, 7), the top 5 features favor one option slightly, but the 6th feature favors the other, resulting in a tie across all 6 features. Theory 1 will guess, while Theory 2 will have a strict preference. By subtracting the adherence rate of the second set from the first, Theory 1 produces a positive value, whereas Theory 2 produces a negative value, cleanly distinguishing the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    t1_score = 0
    t1_count = 0
    t2_score = 0
    t2_count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0, 1) and b == (0, 0, 1, 1, 0, 0):
            t1_score += (1 if resp == 0 else 0)
            t1_count += 1
        elif a == (1, 0, 0, 0, 0, 0) and b == (0, 1, 0, 0, 0, 1):
            t1_score += (1 if resp == 1 else 0)
            t1_count += 1
        elif a == (0, 0, 1, 1, 0, 1) and b == (1, 1, 0, 0, 0, 0):
            t1_score += (1 if resp == 0 else 0)
            t1_count += 1
        elif a == (0, 1, 0, 0, 0, 1) and b == (1, 0, 0, 0, 0, 0):
            t1_score += (1 if resp == 0 else 0)
            t1_count += 1
            
        elif a == (1, 1, 1, 0, 0, 0) and b == (0, 0, 0, 1, 1, 1):
            t2_score += (1 if resp == 0 else 0)
            t2_count += 1
        elif a == (1, 0, 1, 0, 0, 0) and b == (0, 1, 0, 0, 0, 1):
            t2_score += (1 if resp == 0 else 0)
            t2_count += 1
        elif a == (0, 1, 0, 0, 0, 1) and b == (1, 0, 1, 0, 0, 0):
            t2_score += (1 if resp == 1 else 0)
            t2_count += 1
        elif a == (0, 0, 0, 1, 1, 1) and b == (1, 1, 1, 0, 0, 0):
            t2_score += (1 if resp == 1 else 0)
            t2_count += 1

    t1_rate = t1_score / t1_count if t1_count > 0 else 0.5
    t2_rate = t2_score / t2_count if t2_count > 0 else 0.5
    
    return float(t1_rate - t2_rate)

### RESULTS
- Predicted under pi_4 (simulated): -0.1250 (var=0.1835)
- Predicted under pi_2 (simulated): 0.3325 (var=0.0131)
- Observed on real data: 0.7117 (var=0.0409)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        a = a_ratings[i]
        b = b_ratings[i]
        pred = 0.5
        for j in range(4):
            if a[j] > b[j]:
                pred = 0
                break
            elif b[j] > a[j]:
                pred = 1
                break
        ttb_preds[i] = pred
        
    matches = (responses == ttb_preds).mean()
    return float(matches)
```

**Observed (real) value:** 0.2213 (var=0.0039)
**Predicted under pi_4:** 0.2729 (var=0.0101)
**Predicted under pi_2:** 0.2956 (var=0.0061)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    count = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        if resp == tally_pref:
            matches += 1
        count += 1
        
    return float(matches / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.7294 (var=0.0080)
**Predicted under pi_4:** 0.8772 (var=0.0081)
**Predicted under pi_2:** 0.8592 (var=0.0113)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    wadd_aligned = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_A = np.sum(a * v)
        wadd_B = np.sum(b * v)
        
        tally_A = np.sum(a > b)
        tally_B = np.sum(b > a)
        
        # Identify trials where WADD and Tallying make opposite predictions
        if wadd_A > wadd_B and tally_A < tally_B:
            wadd_aligned.append(1 if row['response'] == 0 else 0)
        elif wadd_A < wadd_B and tally_A > tally_B:
            wadd_aligned.append(1 if row['response'] == 1 else 0)
            
    if len(wadd_aligned) == 0:
        return 0.5
    return float(np.mean(wadd_aligned))
```

**Observed (real) value:** 0.2067 (var=0.0216)
**Predicted under pi_4:** 0.2289 (var=0.0607)
**Predicted under pi_2:** 0.1156 (var=0.0152)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    # Isolate the critical trials where one option has 2 positive features and the other has 3
    mask = ((a_sums == 2) & (b_sums == 3)) | ((a_sums == 3) & (b_sums == 2))
    if not mask.any():
        return 0.5
    subset = data[mask]
    
    # Tallying prefers the option with 3 features; WADD prefers the option with 2 features 
    # (because the 2 features have higher validities: 0.95 + 0.85 = 1.8 vs 0.6 + 0.55 + 0.5 = 1.65)
    a_is_3 = subset['option_a_ratings'].apply(sum) == 3
    
    # response == 0 means choice A, response == 1 means choice B
    # We check if the subject chose the option with 3 features
    chose_tallying = (a_is_3.astype(int) == (1 - subset['response']))
    
    return float(chose_tallying.mean())
```

**Observed (real) value:** 0.8433 (var=0.0173)
**Predicted under pi_4:** 0.7200 (var=0.0643)
**Predicted under pi_2:** 0.8558 (var=0.0184)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither Top-N Tallying nor Standard Tallying adequately explains the observed data, particularly in Experiment 1 and 2 where Tallying produces ties. In Experiment 1, Standard Tallying predicts guessing (0.5) and Top-N Tallying predicts a preference for the option with more top-4 features (~0.68), but the observed metric is 0.1289. This indicates subjects strongly preferred the other option. In these trials, the total feature counts are tied, but the options differ in feature validities. The strong deviation from 0.5 suggests that when Tallying results in a tie, subjects do not guess; instead, they use a compensatory or lexicographic fallback (like Take-The-Best) to break the tie. Experiments 4, 5, and 6 show that subjects predominantly use Tallying when it discriminates. Therefore, a hybrid theory is needed.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory: 'Tallying with Take-The-Best Tiebreaker'. People primarily use a simple Tallying heuristic, counting the total number of positive features for each option. If one option has a higher tally, it is chosen. However, if the tallies are tied, subjects do not guess; they fall back to the Take-The-Best (TTB) heuristic, breaking the tie by choosing the option that is favored by the single most valid feature. This explains the strong preference on tied-tally trials (Exp 1 and Exp 2) while maintaining the general Tallying behavior seen in Exp 4, 5, and 6."
}
```

## Usage

```json
{
  "prompt_token_count": 7552,
  "candidates_token_count": 377,
  "total_token_count": 9797
}
```
