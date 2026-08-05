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
Take The Best (TTB) heuristic: People make decisions by searching through features in descending order of their validities. They stop at the first feature that discriminates between the two options and choose the option with the positive value on that feature. If no feature discriminates, they guess randomly. Response noise is incorporated via a lapse rate (epsilon) that can range up to 1.0, allowing for fully random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(-validities)
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive (WADD) model by exploiting the non-compensatory nature of TTB versus the compensatory nature of WADD. We use 5 features with linearly decreasing validities. In the critical trials, one option is favored by the single most valid discriminating feature, while the other option is favored by a larger number of less valid features whose combined weight exceeds that of the single best feature. TTB will strictly follow the single highest-validity cue that discriminates between the options, ignoring all others. Conversely, WADD integrates all available information and will choose the option with the higher weighted sum of features, consistently favoring the option with multiple lower-validity cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice matches the prediction of the Take The Best (TTB) heuristic. The experimental design specifically includes 'critical trials' where the single most valid discriminating feature points to one option, while the sum of all other less valid features points to the other option. On these trials, TTB and WADD will make completely opposite predictions. Therefore, data generated by the TTB model will yield a high score on this metric (close to 1.0, modulo noise), whereas data generated by the WADD model will yield a very low score (close to 0.0, or random chance depending on noise), creating a massive, robust mean gap and low per-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None:
            matches.append(1.0 if ttb_pred == resp else 0.0)
            
    return float(np.mean(matches)) if len(matches) > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.7612 (var=0.0230)
- Predicted under pi_2 (simulated): 0.3573 (var=0.0093)
- Observed on real data: 0.4954 (var=0.0018)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take The Best (TTB) heuristic. We use 5 features with linearly decreasing validities. In the critical trials, one option is favored by the single most valid discriminating feature (which TTB relies on exclusively), while the other option is favored by a larger number of less valid features whose combined weight exceeds that of the single best feature. Because WADD integrates all available information, it will consistently favor the option with multiple lower-validity cues, directly opposing TTB's choices. We also include trials where the best discriminating feature is tied, forcing TTB to look at the second or third best feature, while WADD still integrates the entire profile.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
In this specific experimental design, the Compensatory WADD model will consistently favor the option with the higher total number of positive features (equivalent to the Tallying choice) because the sum of the less valid features outweighs the single most valid feature in the critical trials. Conversely, the non-compensatory Take The Best (TTB) model relies entirely on the first discriminating feature, which in 6 out of the 7 trials favors the option with the *fewer* total positive features. By calculating the proportion of trials where the subject chooses the option with the greater sum of positive features, we perfectly capture the opposing predictions of the two models: WADD will score very high (close to 1.0, minus noise), whereas TTB will score very low (around 1/7, plus noise).

Source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(np.sum)
    b_sums = data['option_b_ratings'].apply(np.sum)
    wadd_choice = (b_sums > a_sums).astype(int)
    return float((data['response'] == wadd_choice).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.3204 (var=0.0117)
- Predicted under pi_2 (simulated): 0.7226 (var=0.0183)
- Observed on real data: 0.5046 (var=0.0028)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask1 = (a_ratings == (0, 0, 0, 1, 1, 1)) & (b_ratings == (1, 1, 0, 0, 0, 0))
    mask2 = (a_ratings == (1, 1, 0, 0, 0, 0)) & (b_ratings == (0, 0, 0, 1, 1, 1))
    
    resp = []
    if mask1.any():
        resp.extend(data.loc[mask1, 'response'].tolist())
    if mask2.any():
        resp.extend((1 - data.loc[mask2, 'response']).tolist())
        
    if len(resp) == 0:
        return 0.5
    return float(np.mean(resp))
```

**Observed (real) value:** 0.4844 (var=0.0167)
**Predicted under pi_3:** 0.7689 (var=0.0409)
**Predicted under pi_2:** 0.5311 (var=0.0885)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    is_t1 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    is_t2 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_consistent = (is_t1 & (data['response'] == 0)) | (is_t2 & (data['response'] == 1))
    mask = is_t1 | is_t2
    
    if mask.sum() == 0:
        return 0.5
    return float(wadd_consistent.sum() / mask.sum())
```

**Observed (real) value:** 0.4800 (var=0.0103)
**Predicted under pi_3:** 0.7592 (var=0.0265)
**Predicted under pi_2:** 0.5550 (var=0.0973)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed metric values are consistently hovering around 0.5, with extremely low between-subject variance (e.g., var=0.0018 in Exp 1, var=0.0028 in Exp 2). If the population consisted of a mix of TTB users and WADD users, the mean might be 0.5, but the between-subject variance would be much higher (closer to 0.25). The extremely low variance indicates that nearly every individual subject is choosing at chance (0.5) on these critical trials. Neither the pure Take The Best (pi_3) nor the Weighted Additive (pi_2) models capture this, as they both predict strong directional preferences (deviations from 0.5). This suggests that subjects are likely overwhelmed by the task, ignoring the validities, and simply guessing randomly.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the TTB theory with a 'Random Guessing / High Noise' theory. The new theory should posit that due to task complexity, lack of correctness feedback, or low motivation, subjects do not systematically evaluate the options using the provided validities or features. Instead, they make uniformly random choices on every trial. The model should predict a flat 0.5 probability for each option, irrespective of the stimulus."
}
```

## Usage

```json
{
  "prompt_token_count": 5327,
  "candidates_token_count": 308,
  "total_token_count": 6992
}
```
