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
Take The Best (Lexicographic Heuristic)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected state to have shape (2, n_features)")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    sorted_indices = np.argsort(-validities, kind="mergesort")
    
    scores = np.zeros(2)
    discriminated = False
    
    # Lexicographic search: stop at the first discriminating feature
    for f_idx in sorted_indices:
        if a[f_idx] != b[f_idx]:
            if a[f_idx] > b[f_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
            discriminated = True
            break
            
    if not discriminated:
        return np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    logits = beta * scores
    logits -= np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # Mixture with uniform lapse
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))

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
**Validities (n_features=5):** [0.6, 0.9, 0.5, 0.8, 0.7]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** To maximally dissociate Take The Best (TTB) from Tallying, this experiment uses a 5-feature design with randomized validity ordering to prevent simple left-to-right reading. The trials are constructed to create direct conflicts between the two heuristics. In 'compensatory' trials, one option wins on the single highest-validity discriminating feature (triggering a TTB choice) while the other option wins on a greater number of lower-validity features (triggering a Tallying choice). Additionally, the design includes 'tie' trials where both options win on an equal number of features (causing Tallying to guess uniformly) but one option's wins are on higher-validity features (causing TTB to strongly prefer it).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the compensatory trials where Take The Best (TTB) and Tallying make strictly opposing predictions. It computes the proportion of choices on these critical trials that align with the TTB prediction. Data generated by TTB will yield scores near 1 (modulo response noise), whereas data generated by Tallying will yield scores near 0, making it a highly discriminative and stable metric.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.6, 0.9, 0.5, 0.8, 0.7])
    sorted_indices = np.argsort(-validities)
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    ttb_preds = np.full(len(data), 0.5)
    for f_idx in sorted_indices:
        mask = (a_mat[:, f_idx] != b_mat[:, f_idx]) & (ttb_preds == 0.5)
        ttb_preds[mask & (a_mat[:, f_idx] > b_mat[:, f_idx])] = 0
        ttb_preds[mask & (a_mat[:, f_idx] < b_mat[:, f_idx])] = 1
        
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_preds = np.full(len(data), 0.5)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    disagree = (ttb_preds != 0.5) & (tally_preds != 0.5) & (ttb_preds != tally_preds)
    
    if not np.any(disagree):
        return 0.5
        
    match = (responses[disagree] == ttb_preds[disagree])
    return float(np.mean(match))

### RESULTS
- Predicted under pi_4 (simulated): 0.8400 (var=0.0099)
- Predicted under pi_2 (simulated): 0.1392 (var=0.0078)
- Observed on real data: 0.1183 (var=0.0078)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.65, 0.95, 0.55, 0.75, 0.51, 0.85]

**Trial pairs (n=10):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 2: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 9: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** To effectively dissociate Tallying from Take The Best (TTB), we utilize a 6-feature design with randomized validity order to prevent simple left-to-right reading. TTB makes decisions based solely on the highest-validity discriminating feature, while Tallying counts the total number of features favoring each option, ignoring validities. We design 'compensatory' trials where one option excels on the most valid feature (triggering TTB) but the other option wins on a greater number of less valid features (triggering Tallying). We also include 'tie' trials where both options win on an equal number of features (causing Tallying to guess uniformly) but one option's wins include the highest-validity discriminating feature (causing TTB to strongly prefer it).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying and Take The Best (TTB) make strictly opposing predictions. It computes the proportion of choices on these 'conflict' trials that are consistent with Tallying. For data generated by Tallying, this value will be close to 1 (modulo noise); for data generated by TTB, it will be close to 0. This creates a massive mean gap with very low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validity_order = [1, 5, 3, 0, 2, 4]
    
    consistent_count = 0
    conflict_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            continue
        tally_pref = 0 if a_wins > b_wins else 1
        
        ttb_pref = -1
        for idx in validity_order:
            if a[idx] != b[idx]:
                ttb_pref = 0 if a[idx] > b[idx] else 1
                break
                
        if ttb_pref != -1 and tally_pref != ttb_pref:
            conflict_count += 1
            if row['response'] == tally_pref:
                consistent_count += 1
                
    if conflict_count == 0:
        return 0.5
    return float(consistent_count / conflict_count)

### RESULTS
- Predicted under pi_4 (simulated): 0.1396 (var=0.0082)
- Predicted under pi_2 (simulated): 0.8680 (var=0.0081)
- Observed on real data: 0.8818 (var=0.0077)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a = np.array(data['option_a_ratings'].tolist())
    b = np.array(data['option_b_ratings'].tolist())
    
    diff = a - b
    mask = diff != 0
    first_diff_idx = np.argmax(mask, axis=1)
    has_diff = np.any(mask, axis=1)
    
    row_indices = np.arange(len(data))
    first_diff_val = diff[row_indices, first_diff_idx]
    
    ttb_winner = np.where(first_diff_val == 1, 0, 1)
    
    a_wins = np.sum(diff == 1, axis=1)
    b_wins = np.sum(diff == -1, axis=1)
    tally_winner = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    disagree = has_diff & (tally_winner != -1) & (ttb_winner != tally_winner)
    
    if not np.any(disagree):
        return 0.5
        
    responses = data['response'].values
    ttb_matches = (responses[disagree] == ttb_winner[disagree])
    
    return float(np.mean(ttb_matches))
```

**Observed (real) value:** 0.1425 (var=0.0068)
**Predicted under pi_4:** 0.8333 (var=0.0124)
**Predicted under pi_2:** 0.1579 (var=0.0134)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = sum_a != sum_b
    if not np.any(mask):
        return 0.5
    tally_choices = (sum_b > sum_a).astype(int)
    responses = np.array(data['response'].values)
    agreements = (responses[mask] == tally_choices[mask]).astype(float)
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.8556 (var=0.0092)
**Predicted under pi_4:** 0.1378 (var=0.0120)
**Predicted under pi_2:** 0.8500 (var=0.0101)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0)
    t2 = (1, 0, 1, 0, 0)
    t5 = (0, 1, 1, 0, 0)
    t6 = (0, 0, 1, 1, 1)
    t8 = (0, 1, 0, 1, 0)
    
    score = 0.0
    count = 0
    
    m1 = data['A_tuple'] == t1
    if m1.any():
        score += (data.loc[m1, 'response'] == 0).mean()
        count += 1
        
    m2 = data['A_tuple'] == t2
    if m2.any():
        score += (data.loc[m2, 'response'] == 0).mean()
        count += 1
        
    m5 = data['A_tuple'] == t5
    if m5.any():
        score += (data.loc[m5, 'response'] == 1).mean()
        count += 1
        
    m6 = data['A_tuple'] == t6
    if m6.any():
        score += (data.loc[m6, 'response'] == 1).mean()
        count += 1
        
    m8 = data['A_tuple'] == t8
    if m8.any():
        score += (data.loc[m8, 'response'] == 1).mean()
        count += 1
        
    return score / max(1, count)
```

**Observed (real) value:** 0.4260 (var=0.0028)
**Predicted under pi_4:** 0.8503 (var=0.0088)
**Predicted under pi_2:** 0.4440 (var=0.0050)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.85, 0.5, 0.95, 0.6, 0.9, 0.55])
    wadd_match = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Calculate Tallying score difference
        tally_diff = np.sum(a > b) - np.sum(b > a)
        
        # Focus strictly on Tallying-tie trials
        if tally_diff == 0:
            # Calculate WADD score difference (assuming gamma=1 for direction)
            wadd_diff = np.sum(val * a) - np.sum(val * b)
            
            if wadd_diff > 0:
                # WADD prefers option A (response 0)
                wadd_match.append(1 if row['response'] == 0 else 0)
            elif wadd_diff < 0:
                # WADD prefers option B (response 1)
                wadd_match.append(1 if row['response'] == 1 else 0)
                
    if len(wadd_match) == 0:
        return 0.5
        
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.1433 (var=0.0085)
**Predicted under pi_4:** 0.8483 (var=0.0153)
**Predicted under pi_2:** 0.4956 (var=0.0070)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 2 (Tallying) provides an excellent fit for the observed data across Experiments 1 through 5, whereas Theory 1 (Take The Best) consistently predicts the exact opposite of subject behavior on conflict trials. However, Experiment 6 reveals a critical failure in Theory 2: on trials where Tallying produces a tie, Theory 2 predicts uniform guessing (0.50), but the observed data shows a highly systematic preference (0.14) against the Weighted Additive (WADD) / TTB direction. This indicates that while subjects primarily use a Tallying (equal-weight) heuristic, their tie-breaking mechanism is not random, nor is it based on the stated cue validities (which would yield a metric > 0.5). Instead, subjects likely rely on a different secondary attribute, such as a left-to-right reading order (primacy bias) or a specific subset of features, to break ties. Theory 1 is fundamentally misaligned with the primary decision policy and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory: 'Tallying with Position-Based Tie-Breaking'. The model should primarily compute equal-weighted tallies for each option (identical to Theory 2). However, instead of guessing randomly on ties, the model should break ties using a lexicographic search based strictly on the left-to-right presentation order of the features (ignoring the numerical validities provided). This maintains the strong Tallying performance on standard conflict trials while providing a deterministic, non-validity-based mechanism to explain the systematic deviations observed in Experiment 6."
}
```

## Usage

```json
{
  "prompt_token_count": 7278,
  "candidates_token_count": 367,
  "total_token_count": 9455
}
```
