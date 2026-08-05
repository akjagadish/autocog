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
People use a Tallying (Equal Weights) strategy to make decisions when faced with multiple cues. Instead of weighting cues by their validities or relying solely on the single most valid cue, individuals simply sum the number of positive features for each option and choose the one with the highest total count. This non-weighted compensatory approach favors coalitions of numerous lower-validity cues over a single high-validity cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weight to all features, simply sum them up
    scores = stim.sum(axis=1)
    
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
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.6, 0.55, 0.5, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 8: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 10: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tallying (Equal Weights) from the Weighted Additive (WADD) rule, this design uses a 6-feature environment where the validities are distinctly skewed: two high-validity cues and four low-validity cues. The critical trials pit an option with fewer, but higher-validity, positive features against an option with more, but lower-validity, positive features. Tallying simply counts the number of positive features and will reliably prefer the option with the greater sheer quantity of endorsements (the 'coalition' of lower-validity cues). WADD, conversely, weights each feature by its validity, leading it to prefer the option endorsed by the fewer but more predictive cues. Additional trials include equal-count feature comparisons to demonstrate that WADD can differentiate where Tallying must guess.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
To robustly discriminate Tallying from WADD, we can evaluate the proportion of choices that align with the Tallying model's core prediction: choosing the option with the strictly greater number of positive features. This design pits 'coalitions' of lower-validity cues against fewer, high-validity cues on several critical trials (e.g., trials 1, 2, 9, 10). On these trials, Tallying will consistently prefer the option with more features, while WADD will prefer the option with fewer but highly-weighted features. By measuring the overall agreement with the simple feature-counting strategy across all unequal-sum trials, Tallying will yield a very high score (close to 1.0, minus noise), whereas WADD will yield a much lower score because it systematically contradicts this strategy on the critical trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate the unweighted sum of positive features for options A and B
    sum_a = data['option_a_ratings'].apply(lambda x: sum(x))
    sum_b = data['option_b_ratings'].apply(lambda x: sum(x))
    
    # Filter to trials where one option has strictly more positive features than the other
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_data = data[mask]
    sum_a_f = sum_a[mask]
    sum_b_f = sum_b[mask]
    
    # Tallying predicts choosing the option with the higher count of positive features
    # Response is 0 for A, 1 for B
    tallying_pred = (sum_b_f > sum_a_f).astype(int)
    
    # Calculate the proportion of choices that align with the Tallying prediction
    match_rate = (filtered_data['response'] == tallying_pred).mean()
    
    return float(match_rate)

### RESULTS
- Predicted under pi_3 (simulated): 0.8486 (var=0.0132)
- Predicted under pi_2 (simulated): 0.4825 (var=0.0538)
- Observed on real data: 0.8419 (var=0.0109)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.7, 0.65, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) rule from Tallying (Equal Weights), we use a 6-feature design with a mix of high, medium, and low validities. Tallying ignores feature validities and simply sums the number of positive features, preferring the option with the higher count. WADD, on the other hand, weights each feature by its validity. By creating trials where one option has fewer positive features but higher validities (e.g., 2 positive features with high validities vs. 3 positive features with lower validities), Tallying will strictly prefer the option with more features, whereas WADD will prefer the option with the higher weighted sum. We also include trials where Tallying produces a tie (equal number of positive features) but WADD has a clear preference, demonstrating WADD's sensitivity to validity differences when feature counts are matched.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on the critical trial where one option has only 2 positive features but they are the most valid (0.95, 0.85), while the other option has 3 positive features but they are the least valid (0.65, 0.55, 0.50). The Tallying model strictly prefers the option with 3 features, leading to a choice probability for the 2-feature option near 0. The WADD model, weighting by validities, assigns a higher average score to the 2-feature option, leading to a much higher choice probability. This metric directly isolates the compensatory nature of WADD against the simple counting of Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_2 = (1, 1, 0, 0, 0, 0)
    target_3 = (0, 0, 0, 1, 1, 1)
    
    choices = []
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ta, tb = tuple(a), tuple(b)
        if ta == target_2 and tb == target_3:
            choices.append(1 if resp == 0 else 0)
        elif ta == target_3 and tb == target_2:
            choices.append(1 if resp == 1 else 0)
            
    return float(np.mean(choices)) if choices else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.1375 (var=0.0119)
- Predicted under pi_2 (simulated): 0.5212 (var=0.0848)
- Observed on real data: 0.1450 (var=0.0105)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    preds = np.zeros(len(data))
    for i in range(len(data)):
        d = diff[i]
        nonzero = np.where(d != 0)[0]
        if len(nonzero) > 0:
            first_cue = nonzero[0]
            preds[i] = 0 if d[first_cue] > 0 else 1
        else:
            preds[i] = -1
            
    valid = preds != -1
    if not np.any(valid):
        return 0.5
        
    match = (preds[valid] == data['response'].values[valid])
    return float(np.mean(match))

```

**Observed (real) value:** 0.2308 (var=0.0062)
**Predicted under pi_3:** 0.2081 (var=0.0055)
**Predicted under pi_2:** 0.4152 (var=0.0282)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    responses = data['response'].values
    diffs = a_ratings - b_ratings
    is_diff = diffs != 0
    first_diff_idx = np.argmax(is_diff, axis=1)
    row_indices = np.arange(len(diffs))
    first_diffs = diffs[row_indices, first_diff_idx]
    ttb_preds = np.where(first_diffs > 0, 0, 1)
    has_diff = np.any(is_diff, axis=1)
    if not np.any(has_diff):
        return 0.5
    matches = (responses[has_diff] == ttb_preds[has_diff])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2096 (var=0.0041)
**Predicted under pi_3:** 0.2233 (var=0.0036)
**Predicted under pi_2:** 0.3287 (var=0.0164)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Tallying / Equal Weights) predicts the observed metric values almost perfectly, while Theory 2 (Weighted Additive / WADD) consistently fails to capture human behavior. The data strongly suggests that subjects are ignoring the cue validities and simply counting the number of positive features for each option. Furthermore, Experiments 3 and 4 evaluate the Take-The-Best (TTB) match rate, which is observed to be around 0.21-0.23. Tallying correctly predicts this low TTB match rate, whereas WADD overestimates it. Since Tallying is highly successful and WADD is fundamentally mismatched to the subjects' counting heuristic, WADD should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the WADD theory with a new theory that builds upon the success of feature-counting. Since Tallying (Equal Weights) is already very strong, a good competing theory would be 'Majority of Confirming Dimensions' or a 'Thresholded Counting' rule, where individuals only count features that pass a certain validity threshold, or perhaps a 'Take-The-Many' heuristic where subjects only look for a strict majority and guess otherwise."
}
```

## Usage

```json
{
  "prompt_token_count": 5507,
  "candidates_token_count": 279,
  "total_token_count": 6435
}
```
