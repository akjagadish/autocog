# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People compare two options by computing a weighted sum of all features for each option, where the weights are directly equal to the feature validities. Because all validities are bounded between 0.5 and 1.0, using them directly restricts the maximum weight ratio between cues, allowing multiple weaker cues to easily outvote a single strong cue. This Weighted Additive (WADD) rule is a fully compensatory strategy that captures the human tendency to integrate all available information while still giving a slight edge to more valid cues. Response noise enters through a softmax over the two options' weighted scores, along with an independent lapse probability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Retrieve validities and use them directly as weights
    val = np.asarray(parameters["validities"], dtype=float)
    weights = val
    
    # Compute weighted sum for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(len(probabilities), p=probabilities))


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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design systematically pits the Weighted Additive (WADD) theory against the Tallying heuristic. WADD integrates all features by weighting them directly by their validities, allowing strong features to outweigh a larger number of weak features. Tallying, on the other hand, ignores validities and simply counts the number of features favoring each option. By setting validities to a spread of [1.0, 0.9, 0.6, 0.5, 0.5], we create trials where WADD strongly prefers an option with fewer but highly valid features, while Tallying prefers the option with a greater number of less valid features. We also include trials where WADD predicts a tie (e.g., one 1.0 validity feature vs. two 0.5 validity features) but Tallying predicts a clear winner, as well as trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a clear winner due to the validity differences.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the WADD prediction on the two trial types where WADD and Tally make strictly opposite predictions. Specifically, in these trials, one option has two highly valid features (WADD preference) while the other has three less valid features (Tally preference). A higher value indicates WADD-like behavior, while a lower value indicates Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A='11000', B='00111' -> WADD prefers A (1.9 vs 1.6), Tally prefers B (2 vs 3)
    # Trial 2: A='00111', B='11000' -> WADD prefers B (1.6 vs 1.9), Tally prefers A (3 vs 2)
    
    t1 = df[(df['A_str'] == '11000') & (df['B_str'] == '00111')]
    t2 = df[(df['A_str'] == '00111') & (df['B_str'] == '11000')]
    
    score = 0.0
    n = 0
    
    if len(t1) > 0:
        score += (t1['response'] == 0).sum()
        n += len(t1)
    if len(t2) > 0:
        score += (t2['response'] == 1).sum()
        n += len(t2)
        
    if n == 0:
        return 0.5
    return float(score / n)


### RESULTS
- Predicted under pi_3 (simulated): 0.7842 (var=0.0284)
- Predicted under pi_2 (simulated): 0.1667 (var=0.0219)
- Observed on real data: 0.1333 (var=0.0128)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.5, 0.5, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 9: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** This design systematically pits Tallying against the Weighted Additive (WADD) theory. Tallying simply counts the number of features favoring each option, ignoring validities, while WADD computes a weighted sum using the validities as weights. By setting validities to [1.0, 0.9, 0.5, 0.5, 0.5], we create strong dissociations. For example, in some trials, one option wins on the two most valid cues (WADD favors it) while the other option wins on the three least valid cues (Tallying favors it). We also include trials where WADD predicts a tie (e.g., one 1.0 validity feature vs. two 0.5 validity features) but Tallying predicts a clear winner, as well as trials where Tallying predicts a tie (equal number of winning features) but WADD predicts a clear winner due to the validity differences.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
On Trial 1, one option has the two most valid cues (1.0 and 0.9) and the other has the three least valid cues (0.5, 0.5, 0.5). WADD computes a higher score for the first option (1.9 vs 1.5) and thus predicts it will be chosen. Conversely, Tallying simply counts the number of winning features, favoring the second option (3 wins vs 2 wins). This metric isolates this specific trial type and calculates the proportion of choices that align with the WADD prediction, cleanly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    wadd_chosen = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_chosen += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_chosen += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_chosen / total)

### RESULTS
- Predicted under pi_3 (simulated): 0.7733 (var=0.0291)
- Predicted under pi_2 (simulated): 0.1511 (var=0.0177)
- Observed on real data: 0.1956 (var=0.0240)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))
```

**Observed (real) value:** 0.1383 (var=0.0087)
**Predicted under pi_3:** 0.1503 (var=0.0082)
**Predicted under pi_2:** 0.1297 (var=0.0093)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8286 (var=0.0105)
**Predicted under pi_3:** 0.8264 (var=0.0085)
**Predicted under pi_2:** 0.8479 (var=0.0103)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data strongly supports Tallying (THEORY 2, pi_2) over Weighted Additive (THEORY 1, pi_3). In Experiments 1 and 2, which were specifically designed to pit WADD against Tallying by having high-validity cues contradict the sheer number of winning cues, participants overwhelmingly chose the option favored by Tallying (observed WADD-choice proportions of 0.13 and 0.20, matching Tallying's predictions of 0.17 and 0.15). THEORY 1 completely fails to capture this, predicting WADD-choice proportions of ~0.78. Experiments 3 and 4 further confirm that people rely heavily on counting feature wins rather than weighting by validity or using a strict Take-The-Best strategy. Thus, THEORY 1 is fundamentally misaligned with human behavior in this paradigm.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Since Tallying (pi_2) is already highly successful, the new theory should explore a nuanced boundary condition of Tallying rather than reverting to a fully compensatory WADD model. Propose a 'Take-The-Best with Tallying Fallback' (or Lexicographic Semi-Tallying) theory: people first check if the most valid cue strongly discriminates the options (perhaps only if its validity is exceptionally high or above a threshold). If it doesn't, they fall back to a simple Tallying strategy. Alternatively, propose a 'Top-K Tallying' model where individuals tally feature wins but only among the top K most valid features, ignoring the least valid ones to save cognitive effort."
}
```

## Usage

```json
{
  "prompt_token_count": 5719,
  "candidates_token_count": 379,
  "total_token_count": 6713
}
```
