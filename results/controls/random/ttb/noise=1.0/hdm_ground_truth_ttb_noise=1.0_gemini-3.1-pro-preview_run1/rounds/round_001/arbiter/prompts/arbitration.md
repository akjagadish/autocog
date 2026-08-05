# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take The Best (TTB) heuristic: People make decisions between two options by sequentially inspecting features in descending order of their subjective or objective validity. The process stops at the first feature that discriminates between the options (i.e., one option has a higher value than the other). The option with the higher value on this discriminating feature is chosen, and all remaining features are ignored, regardless of their magnitudes. If no features discriminate, the decision maker guesses. Response noise is modeled via a softmax function on the binary outcome and an epsilon lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(validities)[::-1]
    
    # Default scores if no cue discriminates (leads to 50/50 after softmax)
    scores = np.array([0.0, 0.0])
    
    # Lexicographic search
    for cue in cue_order:
        if stim[0, cue] > stim[1, cue]:
            scores = np.array([1.0, 0.0])
            break
        elif stim[1, cue] > stim[0, cue]:
            scores = np.array([0.0, 1.0])
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[6, 0, 0]  B=[5, 10, 10]
  trial 2: A=[5, 10, 10]  B=[6, 0, 0]
  trial 3: A=[8, 2, 10]  B=[8, 3, 0]
  trial 4: A=[0, 8, 8]  B=[1, 0, 0]
  trial 5: A=[10, 0, 0]  B=[9, 9, 9]
  trial 6: A=[5, 5, 5]  B=[6, 0, 0]
  trial 7: A=[2, 10, 10]  B=[3, 2, 2]
  trial 8: A=[7, 8, 0]  B=[7, 7, 10]

**Rationale:** To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we exploit their fundamental difference in processing: TTB relies exclusively on the highest-validity discriminating feature and ignores both the magnitudes of differences and all lower-validity features. WADD, conversely, computes a compensatory weighted sum that integrates all features, sensitive to both their validities and their cardinal magnitudes. By setting a high rating maximum (rating_max = 10) and using three features with clear validity differences (0.9, 0.7, 0.6), we can construct trials where one option has a minimal advantage on the highest-validity cue (thus chosen by TTB), while the other option has massive advantages on the remaining cues (thus chosen by WADD). We also include trials where the highest validity cue is tied, pushing TTB to the second cue, which is then outweighed by the third cue in WADD's calculation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, the chosen experimental trials perfectly dissociate Take The Best (TTB) and Weighted Additive (WADD) models: on every single trial, the two models predict opposite choices. This metric directly computes the proportion of choices that are consistent with the lexicographic TTB rule. For data generated by TTB, this metric will be close to 1.0 (subject to lapse noise). For data generated by WADD, it will be close to 0.0, as WADD reliably chooses the compensatory option. This ensures a massive mean separation with very tight per-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = np.zeros(len(data), dtype=int) - 1
    for i in range(len(data)):
        for j in range(a_ratings.shape[1]):
            if diff[i, j] > 0:
                ttb_choices[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_choices[i] = 1
                break
                
    valid = ttb_choices != -1
    if not np.any(valid):
        return 0.5
        
    match = (responses[valid] == ttb_choices[valid])
    return float(np.mean(match))

### RESULTS
- Predicted under pi_3 (simulated): 0.8508 (var=0.0143)
- Predicted under pi_2 (simulated): 0.0200 (var=0.0046)
- Observed on real data: 0.4829 (var=0.0017)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=3):** [0.95, 0.75, 0.6]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[6, 0, 0]  B=[5, 10, 10]
  trial 2: A=[10, 2, 0]  B=[10, 1, 10]
  trial 3: A=[1, 10, 10]  B=[2, 0, 0]
  trial 4: A=[5, 5, 5]  B=[6, 0, 0]
  trial 5: A=[0, 8, 8]  B=[1, 1, 1]
  trial 6: A=[7, 7, 7]  B=[7, 8, 0]
  trial 7: A=[4, 9, 0]  B=[4, 10, 0]
  trial 8: A=[0, 0, 10]  B=[0, 1, 0]

**Rationale:** This design quantitatively dissociates WADD from TTB by exploiting TTB's non-compensatory, lexicographic nature and its insensitivity to cardinal magnitudes. By using a large rating scale (rating_max = 10), we create trials where one option has a negligible advantage on the highest-validity cue (thus deterministically chosen by TTB) but massive disadvantages on lower-validity cues (thus rejected by WADD). We also include trials featuring ties on the primary cue, pushing TTB to the secondary cue, where again a small advantage dictates TTB's choice while WADD integrates the massive advantage of the alternative option on the tertiary cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject makes the choice predicted by the non-compensatory Take The Best (TTB) heuristic. The experimental design specifically pits TTB against the compensatory Weighted Additive (WADD) rule on 7 out of 8 trial types, meaning TTB will consistently score very high on this metric (close to 1.0, minus lapse rate) while WADD will score very low, providing strong discriminability.

Source:
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (6, 0, 0) and b == (5, 10, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (10, 2, 0) and b == (10, 1, 10):
            ttb_match += (resp == 0)
            count += 1
        elif a == (1, 10, 10) and b == (2, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (5, 5, 5) and b == (6, 0, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 8, 8) and b == (1, 1, 1):
            ttb_match += (resp == 1)
            count += 1
        elif a == (7, 7, 7) and b == (7, 8, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (4, 9, 0) and b == (4, 10, 0):
            ttb_match += (resp == 1)
            count += 1
        elif a == (0, 0, 10) and b == (0, 1, 0):
            ttb_match += (resp == 1)
            count += 1
    if count == 0:
        return 0.5
    return ttb_match / count


### RESULTS
- Predicted under pi_3 (simulated): 0.8462 (var=0.0150)
- Predicted under pi_2 (simulated): 0.1450 (var=0.0059)
- Observed on real data: 0.5017 (var=0.0022)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 4, 4]  B=[5, 3, 3]
  A=[5, 0, 0]  B=[2, 1, 1]
  A=[2, 5, 0]  B=[3, 1, 1]
  A=[0, 4, 4]  B=[2, 3, 3]
  A=[0, 5, 5]  B=[3, 4, 4]
  A=[4, 1, 0]  B=[1, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_pred = (b_wins > a_wins).astype(int)
    
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.4917 (var=0.0021)
**Predicted under pi_3:** 0.2800 (var=0.0057)
**Predicted under pi_2:** 0.3067 (var=0.1115)

### Experiment 4
**Design**
  A=[2, 6, 6]  B=[10, 5, 5]
  A=[10, 2, 2]  B=[8, 3, 3]
  A=[9, 5, 4]  B=[1, 5, 8]
  A=[5, 5, 5]  B=[0, 8, 8]
  A=[1, 9, 9]  B=[8, 8, 8]
  A=[9, 10, 1]  B=[10, 1, 9]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        resp = row['response']
        # Trial 1: A=[2, 6, 6], B=[10, 5, 5]
        if a[0] == 2:
            wadd_match.append(1 if resp == 1 else 0)
        # Trial 5: A=[1, 9, 9], B=[8, 8, 8]
        elif a[0] == 1:
            wadd_match.append(1 if resp == 1 else 0)
    if not wadd_match:
        return 0.5
    return float(np.mean(wadd_match))
```

**Observed (real) value:** 0.5000 (var=0.0073)
**Predicted under pi_3:** 0.8675 (var=0.0085)
**Predicted under pi_2:** 0.8900 (var=0.0829)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed metric values hover remarkably close to 0.50 (e.g., 0.4829, 0.5017, 0.4917, 0.5000) with extremely low between-subject variance. This indicates that subjects are not consistently using either the non-compensatory Take The Best (TTB) heuristic or the compensatory Weighted Additive (WADD) rule. In fact, because the experimental designs specifically pitted these strategies against each other to create maximal divergence (where TTB and WADD would predict close to 1.0 or 0.0 depending on the metric), an observed score of 0.50 with near-zero variance strongly implies that subjects are behaving at chance. They are likely either guessing randomly, overwhelmed by the cardinal ratings, or ignoring the cue validities altogether. Neither TTB nor WADD captures this pervasive baseline randomness without degenerate parameterizations.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Random Guessing' or 'Zero-Intelligence' baseline theory. The theory should state that in complex multi-attribute choice tasks without trial-by-trial feedback, subjects may fail to integrate cue validities or magnitudes and instead simply guess with equal probability between the available options. The model should predict a 50/50 probability distribution across all trials, ignoring the stimulus features and validities entirely. This will serve as a necessary baseline to capture the observed ~0.50 metrics and low variance."
}
```

## Usage

```json
{
  "prompt_token_count": 5146,
  "candidates_token_count": 346,
  "total_token_count": 7601
}
```
