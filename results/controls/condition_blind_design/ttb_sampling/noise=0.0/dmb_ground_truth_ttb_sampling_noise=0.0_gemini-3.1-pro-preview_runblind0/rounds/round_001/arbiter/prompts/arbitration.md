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
Decision-makers use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. Features are ranked by their validity, and options are compared on features one by one in descending order of validity. The choice is determined entirely by the first feature that discriminates between the options, ignoring all lower-validity cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Lexicographic evaluation
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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
**Validities (n_features=4):** [0.95, 0.8, 0.78, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0]  B=[1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 1]  B=[0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0]  B=[0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 16: A=[0, 1, 1, 0]  B=[0, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific 'compensatory' trials in the design where the most valid cue (F1) favors one option, but a combination of lower-validity cues favors the other option. On these trials, Take The Best strictly predicts choosing the option with the highest valid cue, while Weighted Additive (WADD) can compensate and choose the other option. By calculating the proportion of choices that align with the TTB prediction on these specific trials, we expect TTB to score close to 1.0 (minus noise), whereas WADD will score significantly lower as it frequently chooses the compensatory option.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 3: A=[0, 1, 1, 0], B=[1, 0, 0, 0]
        # TTB chooses B (due to F1), WADD might choose A (due to F2+F3)
        if a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
        elif a == (1, 0, 0, 0) and b == (0, 1, 1, 0):
            if resp == 0:
                ttb_match += 1
            total += 1
            
        # Trial 11: A=[1, 0, 1, 0], B=[0, 1, 1, 1]
        # TTB chooses A (due to F1), WADD might choose B (due to F2+F4)
        elif a == (1, 0, 1, 0) and b == (0, 1, 1, 1):
            if resp == 0:
                ttb_match += 1
            total += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 1, 0):
            if resp == 1:
                ttb_match += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_match / total)


### RESULTS
- Predicted under pi_3 (simulated): 0.8117 (var=0.0236)
- Predicted under pi_2 (simulated): 0.3750 (var=0.0726)
- Observed on real data: 0.8433 (var=0.0296)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.94, 0.66, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 2: A=[1, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 12: A=[0, 1, 1, 1]  B=[0, 0, 1, 0]
  trial 13: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 15: A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials (10 and 14) where the two theories make opposing predictions. On these trials, one option has a singular advantage on the second-most valid feature (validity 0.94), while the other option has advantages on both the third and fourth features (validities 0.66 and 0.55). TTB, being lexicographic, will always choose the option with the advantage on the higher-validity single feature. WADD, being compensatory, will frequently evaluate the combined weight of the two lesser features as exceeding the single higher-validity feature, leading to the opposite choice. The metric returns the proportion of choices that align with the compensatory (WADD) prediction.

Source:
def metric(data: pd.DataFrame) -> float:
    # Convert feature lists to tuples to allow element-wise comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the two trials where WADD's compensatory nature opposes TTB's lexicographic rule
    # Trial 10: TTB chooses B (due to feature 2), WADD leans A (features 3 + 4 compensate for feature 2)
    is_trial_10 = (a_tuples == (0, 0, 1, 1)) & (b_tuples == (0, 1, 0, 0))
    # Trial 14: TTB chooses A (due to feature 2), WADD leans B (features 3 + 4 compensate for feature 2)
    is_trial_14 = (a_tuples == (1, 1, 0, 0)) & (b_tuples == (1, 0, 1, 1))
    
    # Calculate the proportion of choices that align with the WADD compensatory prediction
    wadd_choice_10 = (data.loc[is_trial_10, 'response'] == 0).mean()
    wadd_choice_14 = (data.loc[is_trial_14, 'response'] == 1).mean()
    
    # Handle edge cases where a subject might have missing data for these specific trials
    if pd.isna(wadd_choice_10): wadd_choice_10 = 0.5
    if pd.isna(wadd_choice_14): wadd_choice_14 = 0.5
    
    return float((wadd_choice_10 + wadd_choice_14) / 2.0)

### RESULTS
- Predicted under pi_3 (simulated): 0.1550 (var=0.0122)
- Predicted under pi_2 (simulated): 0.5783 (var=0.0843)
- Observed on real data: 0.1333 (var=0.0156)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.76, 0.92, 0.55])
    
    # Extract matrices of A and B features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying differences
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_diff = a_wins - b_wins
    
    # WADD differences (expected without subject-specific weights)
    wadd_diff = np.sum((a_mat - b_mat) * validities, axis=1)
    
    # 1 if chose A, 0 if chose B
    chose_a = 1.0 - data['response'].values
    
    # We compare choice probabilities within trials where Tallying predicts the exact same difference.
    # For tally_diff == 1, WADD predicts some trials favor A more strongly than others.
    mask1_high = (tally_diff == 1) & (wadd_diff > 0.85)
    mask1_low = (tally_diff == 1) & (wadd_diff < 0.85)
    
    diff1 = 0.0
    if np.any(mask1_high) and np.any(mask1_low):
        diff1 = np.mean(chose_a[mask1_high]) - np.mean(chose_a[mask1_low])
        
    # For tally_diff == -1, WADD predicts some trials favor B more strongly than others.
    mask_m1_high = (tally_diff == -1) & (wadd_diff > -0.85)
    mask_m1_low = (tally_diff == -1) & (wadd_diff < -0.85)
    
    diff_m1 = 0.0
    if np.any(mask_m1_high) and np.any(mask_m1_low):
        diff_m1 = np.mean(chose_a[mask_m1_high]) - np.mean(chose_a[mask_m1_low])
        
    # Under Tallying, both diff1 and diff_m1 should be 0.
    # Under WADD, both diff1 and diff_m1 should be positive.
    return float(diff1 + diff_m1)
```

**Observed (real) value:** 0.1467 (var=0.0250)
**Predicted under pi_3:** 0.1900 (var=0.0422)
**Predicted under pi_2:** 0.0792 (var=0.0977)

### Experiment 4
**Design**
  A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Tallying predicts a tie but WADD predicts a strong preference.
    # Trial 8: A=[0, 1, 0, 1], B=[1, 1, 0, 0]
    # A wins on feature 4 (validity 0.55). B wins on feature 1 (validity 0.95).
    # Tallying sees 1 win for A and 1 win for B, predicting exactly 50% choice for B.
    # WADD sees B's win on the most important feature as outweighing A's win on the least important, predicting >50% choice for B.
    is_target = data['option_a_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1]) & \
                data['option_b_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0])
    
    if is_target.sum() == 0:
        return 0.5
        
    return float(data.loc[is_target, 'response'].mean())
```

**Observed (real) value:** 0.8200 (var=0.0532)
**Predicted under pi_3:** 0.8733 (var=0.0284)
**Predicted under pi_2:** 0.6533 (var=0.0887)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Take The Best, pi_3) consistently outperforms Theory 2 (Weighted Additive, pi_2). In Experiment 1, subjects overwhelmingly chose the option favored by the single most valid cue, ignoring combinations of lower-validity cues (observed 0.84, TTB predicted 0.81, WADD predicted 0.38). In Experiment 2, when WADD opposed TTB, subjects again sided with the TTB prediction (observed 0.13 matching WADD, TTB predicted 0.15). Experiments 3 and 4 further confirmed that choices align closely with lexicographic evaluation rather than compensatory validity-weighted sums. Theory 2 is systematically failing to capture the heavily non-compensatory nature of the subjects' choices.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Rank-Weighted Additive' theory. Instead of using the raw validities as weights (which allows too much compensation), decision-makers rank the features by validity and assign exponentially decaying weights based on rank (e.g., w_k = 2^{-k}). This acts as a 'soft' lexicographic model: it largely mimics Take The Best by making the top cue dominant, but allows for compensation only when a large number of lower-ranked cues unanimously oppose the top cue. This will provide a much stronger, competitive alternative to the strict TTB heuristic."
}
```

## Usage

```json
{
  "prompt_token_count": 6772,
  "candidates_token_count": 325,
  "total_token_count": 8097
}
```
