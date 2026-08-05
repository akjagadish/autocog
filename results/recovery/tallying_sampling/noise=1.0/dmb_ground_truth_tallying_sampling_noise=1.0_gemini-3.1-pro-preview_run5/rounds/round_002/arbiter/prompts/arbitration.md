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
When faced with multi-attribute choices without correctness feedback and where no single option overwhelmingly dominates, subjects may lack the motivation or cognitive resources to systematically apply complex heuristics like Take-The-Best or Tallying. Instead, their behavior is best described by a Random Guessing baseline. In this model, subjects simply guess between the two options on every trial, potentially exhibiting a slight idiosyncratic bias toward one option over the other, but otherwise ignoring the feature validities and values entirely.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Pure random guessing baseline with an idiosyncratic side bias.
    # The stimulus and history are ignored.
    bias = float(parameters["bias"])
    
    # Return the probabilities for choosing Option A (index 0) and Option B (index 1)
    return np.array([bias, 1.0 - bias])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))

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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To dissociate Random Guessing from the Weighted Additive (WADD) rule, we must vary the difference in weighted feature sums across trials. Random Guessing predicts a flat choice probability (modulo an idiosyncratic side bias) regardless of the stimulus. In contrast, WADD predicts that choice probabilities will track the difference in weighted sums, approaching deterministic choice for large differences and dropping to 50% only when the weighted sums are equal. The design includes trials where Option A strongly dominates, strongly loses, wins or loses by a small margin, and ties with Option B. This range of WADD score differences will yield an S-shaped response curve under WADD, but a flat horizontal line under Random Guessing.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the subject's accuracy on 'extreme' trials where one option completely dominates the other (i.e., has all positive features while the other has none). Under the Random Guessing theory, subjects ignore the stimulus, so their choice on these trials will only reflect their baseline bias, averaging to 0.5 across both types of extreme trials. Under the WADD theory, the large difference in weighted sums will lead to a very high probability of choosing the dominating option (bounded only by the lapse rate epsilon). Thus, WADD predicts a score significantly higher than 0.5, while Random Guessing predicts exactly 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of features for options A and B
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Identify extreme trials where one option completely dominates the other
    diff = sum_a - sum_b
    extreme_mask = diff.abs() == 5
    
    if extreme_mask.sum() == 0:
        return 0.5
        
    # Determine the 'correct' choice (0 for A, 1 for B)
    correct_responses = (diff[extreme_mask] < 0).astype(int)
    actual_responses = data.loc[extreme_mask, 'response']
    
    # Calculate the proportion of choices that align with the dominating option
    return float((actual_responses == correct_responses).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.4708 (var=0.0099)
- Predicted under pi_2 (simulated): 0.8708 (var=0.0124)
- Observed on real data: 0.5050 (var=0.0055)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**Trial pairs (n=11):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 9: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 10: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 11: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) rule from Random Guessing, we vary the difference in weighted feature sums systematically across trials. Under Random Guessing, choice probabilities remain flat (ignoring the stimulus entirely, modulo a side bias). Under WADD, the probability of choosing Option A should track the difference in weighted feature sums, forming an S-shaped psychometric curve. We use a 4-feature design and construct pairs that span a wide range of WADD score differences: from Option A strongly dominating, to moderate advantages, slight advantages, exact ties, and corresponding advantages for Option B. This gradient of score differences will yield a monotonic response curve under WADD but a flat horizontal line under Random Guessing.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



### METRIC
Rationale:
Under the Weighted Additive (WADD) theory, subjects compute a weighted sum of features using the validities and choose the option with the higher score. Therefore, their choices will align with the higher WADD score significantly more often than chance. Under the Random Guessing theory, subjects ignore the features entirely, so the proportion of choices consistent with the WADD score will hover around 0.5. This metric directly calculates the accuracy of the deterministic WADD rule on trials with unequal scores, yielding a large and highly significant contrast between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.8, 0.6, 0.5])
    
    # Vectorized computation of WADD scores for Option A and B
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    score_A = A.dot(validities)
    score_B = B.dot(validities)
    diff = score_A - score_B
    
    # Filter out trials where options have identical WADD scores
    mask = np.abs(diff) > 1e-5
    if not np.any(mask):
        return 0.5
        
    # Predicted response: 0 (Option A) if score_A > score_B, else 1 (Option B)
    predicted = (diff[mask] < 0).astype(int)
    actual = data['response'].values[mask]
    
    # Return the proportion of choices consistent with the WADD rule
    return float(np.mean(predicted == actual))

### RESULTS
- Predicted under pi_4 (simulated): 0.5078 (var=0.0034)
- Predicted under pi_2 (simulated): 0.6715 (var=0.0171)
- Observed on real data: 0.5015 (var=0.0036)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Calculate the difference between features
    diff = a_mat - b_mat
    
    # Find the index of the first discriminating cue (first non-zero difference)
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    
    ttb_preds = np.zeros(len(diff))
    for i in range(len(diff)):
        idx = first_diff_idx[i]
        if diff[i, idx] > 0:
            ttb_preds[i] = 0
        elif diff[i, idx] < 0:
            ttb_preds[i] = 1
        else:
            ttb_preds[i] = 0.5
            
    # Return the proportion of choices that align with the Take-The-Best prediction
    return float(np.mean(resp == ttb_preds))
```

**Observed (real) value:** 0.4883 (var=0.0032)
**Predicted under pi_4:** 0.4985 (var=0.0023)
**Predicted under pi_2:** 0.4183 (var=0.0236)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    valid_trials = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        if ttb_pred != -1:
            if row['response'] == ttb_pred:
                matches += 1
            valid_trials += 1
    return matches / valid_trials if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.4963 (var=0.0022)
**Predicted under pi_4:** 0.4925 (var=0.0018)
**Predicted under pi_2:** 0.4096 (var=0.0238)

### Experiment 5
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A has 3 low-validity cues, B has 2 high-validity cues
    is_trial_1 = (a_str == '00111') & (b_str == '11000')
    # Trial 2: A has 2 high-validity cues, B has 3 low-validity cues
    is_trial_2 = (a_str == '11000') & (b_str == '00111')
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    # Tallying prefers the option with 3 positive features
    tally_correct_1 = is_trial_1 & (data['response'] == 0)
    tally_correct_2 = is_trial_2 & (data['response'] == 1)
    
    return float((tally_correct_1 | tally_correct_2).sum() / mask.sum())
```

**Observed (real) value:** 0.4833 (var=0.0088)
**Predicted under pi_4:** 0.4767 (var=0.0081)
**Predicted under pi_2:** 0.4708 (var=0.1009)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 1 directly pits the two most valid features (Option A) 
    # against the three least valid features (Option B).
    # Tallying strictly prefers Option B (tally 3 vs 2).
    # WADD, on average, prefers Option A because the combined validity 
    # of the top 2 features outweighs the bottom 3.
    t1_mask = data['option_a_ratings'].apply(tuple) == (1, 1, 0, 0, 0)
    if not t1_mask.any():
        return 0.5
    
    # Return the proportion of times Option A was chosen
    return float((data.loc[t1_mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.4500 (var=0.0206)
**Predicted under pi_4:** 0.4750 (var=0.0231)
**Predicted under pi_2:** 0.4667 (var=0.0733)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed metric values consistently hover around 0.5, strongly aligning with the predictions of THEORY 1 (Random Guessing). THEORY 2 (WADD) consistently overpredicts the extent to which subjects utilize feature validities to make choices, as seen in its expected accuracies of ~0.87 in Exp 1 and ~0.67 in Exp 2, which fail to materialize in the real data. The lack of correctness feedback and the complexity of the multi-attribute arrays seem to drive subjects away from compensatory integration. Since Random Guessing captures the aggregate data so well, WADD is not a viable competitor. We need a new theory that can mimic near-chance performance on these metrics but posits a specific, boundedly rational mechanism rather than pure randomness.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'First-Feature Heuristic' (or 'Left-to-Right Reading Bias') theory. Instead of integrating all features (like WADD) or guessing entirely at random (like THEORY 1), subjects might minimize cognitive effort by only inspecting the very first feature in the array (the leftmost one). If one option has a 1 and the other a 0 on this first feature, they choose the option with the 1. If they tie on the first feature, the subject guesses. Because the first feature's validity and values vary across trials and don't necessarily align with the overall WADD score or TTB predictions, this heuristic would produce aggregate metric values near 0.5 for the current metrics, making it a strong and mechanistically distinct competitor to pure Random Guessing."
}
```

## Usage

```json
{
  "prompt_token_count": 6449,
  "candidates_token_count": 365,
  "total_token_count": 7790
}
```
