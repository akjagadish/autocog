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
Rank-Weighted Additive Theory: People evaluate options by computing a weighted sum of features, but instead of using cardinal validities, they assign weights based on the ordinal rank of the cues' validities. Specifically, weights decay linearly with rank (e.g., n_features - rank), providing a compensatory mechanism where the top cue has higher importance but not so much that it completely dominates lower-ranked cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    
    # Rank-Weighted Additive: weight cues by linear rank decay (n_features - rank)
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    cue_order = np.argsort(-val, kind="stable")
    weights = np.zeros(n_features)
    for rank, idx in enumerate(cue_order):
        weights[idx] = float(n_features - rank)
        
    scores = stim @ weights
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
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
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To dissociate Rank-Weighted Additive (Rank-WADD) from cardinal Weighted Additive (WADD) models, we exploit the difference between linear rank-based weights and actual validities. We use 5 features where the highest validity is much larger than the others, while the remaining validities decline gradually (0.95, 0.65, 0.60, 0.55, 0.50). In Rank-WADD, weights decay linearly (5, 4, 3, 2, 1). Thus, combinations of lower-ranked features can easily outweigh the top feature in rank-weight sums, but not in cardinal-validity sums. For instance, comparing [1, 0, 0, 0, 1] to [0, 1, 1, 0, 0]: Rank-WADD prefers Option B (rank sum 7 vs 6), while WADD prefers Option A (cardinal sum 1.45 vs 1.25). We include multiple such dissociative pairs in both directions alongside baseline pairs where models agree.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the three dissociative trial pairs where Rank-WADD and WADD make opposite predictions. It computes the proportion of choices that align with the Rank-WADD prediction on these specific trials. Data simulated under Rank-WADD will yield a high proportion (close to 1, modulo noise), whereas data simulated under WADD will yield a low proportion (close to 0).

Source:
def metric(data: pd.DataFrame) -> float:
    match_count = 0
    total_count = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        if a == (1, 0, 0, 0, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 0, 1, 1):
            match_count += (resp == 1)
            total_count += 1
        elif a == (1, 0, 1, 0, 0):
            match_count += (resp == 0)
            total_count += 1
    if total_count == 0:
        return 0.5
    return match_count / total_count

### RESULTS
- Predicted under pi_3 (simulated): 0.8458 (var=0.0130)
- Predicted under pi_2 (simulated): 0.4662 (var=0.0377)
- Observed on real data: 0.4467 (var=0.0059)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.98, 0.7, 0.68, 0.66, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the cardinal Weighted Additive (WADD) rule from the Rank-Weighted Additive theory, we exploit the non-linear gaps between feature validities. We use five features with validities [0.98, 0.70, 0.68, 0.66, 0.50]. In Rank-WADD, the weights are strictly linear based on rank (5, 4, 3, 2, 1). Therefore, the second and third features combined (rank sum 4 + 3 = 7) will outweigh the first and fifth features combined (rank sum 5 + 1 = 6). However, in the cardinal WADD model, the first feature is exceptionally strong (0.98), so the sum of the first and fifth features (0.98 + 0.50 = 1.48) outweighs the sum of the second and third (0.70 + 0.68 = 1.38). By including pairs where these combinations are pitted against each other, we create strong opposite predictions between the two models. Additional pairs are included where both models agree, serving as baseline controls.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The critical trial pair pits [1, 0, 0, 0, 1] against [0, 1, 1, 0, 0]. The WADD model, using cardinal validities [0.98, 0.70, 0.68, 0.66, 0.50], assigns a higher value to the first option (0.98 + 0.50 = 1.48) over the second (0.70 + 0.68 = 1.38). In contrast, the Rank-WADD model assigns linear rank weights (5, 4, 3, 2, 1) and thus prefers the second option (4 + 3 = 7) over the first (5 + 1 = 6). By calculating the proportion of times the second option [0, 1, 1, 0, 0] is chosen over the first, we obtain a metric where WADD predicts a value < 0.5 and Rank-WADD predicts a value > 0.5. Averaging across 16 repetitions per subject stabilizes the metric and minimizes between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # We want to identify trials with the critical pair:
    # [1, 0, 0, 0, 1] vs [0, 1, 1, 0, 0]
    
    def is_target(a, b):
        return (tuple(a) == (1, 0, 0, 0, 1) and tuple(b) == (0, 1, 1, 0, 0))
    
    def is_target_swapped(a, b):
        return (tuple(a) == (0, 1, 1, 0, 0) and tuple(b) == (1, 0, 0, 0, 1))

    target_chosen = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if is_target(a, b):
            # Option B is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 1 else 0)
        elif is_target_swapped(a, b):
            # Option A is [0, 1, 1, 0, 0]
            target_chosen.append(1 if resp == 0 else 0)
            
    if not target_chosen:
        return 0.5
    
    return float(np.mean(target_chosen))

### RESULTS
- Predicted under pi_3 (simulated): 0.8600 (var=0.0162)
- Predicted under pi_2 (simulated): 0.4875 (var=0.0942)
- Observed on real data: 0.5000 (var=0.0194)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
                
        if ttb_choice is not None and row['response'] == ttb_choice:
            ttb_matches += 1
            
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4071 (var=0.0029)
**Predicted under pi_3:** 0.4281 (var=0.0019)
**Predicted under pi_2:** 0.3210 (var=0.0381)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    filtered_resp = data.loc[mask, 'response']
    a_higher = sum_a[mask] > sum_b[mask]
    
    chose_higher = (a_higher & (filtered_resp == 0)) | (~a_higher & (filtered_resp == 1))
    return float(chose_higher.mean())
```

**Observed (real) value:** 0.6008 (var=0.0065)
**Predicted under pi_3:** 0.6779 (var=0.0068)
**Predicted under pi_2:** 0.6663 (var=0.0281)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, Theory 1 (Rank-WADD) strongly predicts a preference for options that win on rank-based weights (predicting ~0.85), but the observed data shows subjects are indifferent or slightly favor the cardinal-WADD prediction (observed ~0.45-0.50). This indicates that people do not simply assign linear weights based on cue rank. Theory 2 (WADD) captures Experiments 1 and 2 much better. However, in Experiments 3 and 4, subjects show behaviors that partially align with simpler heuristics like Tallying or Take-The-Best. Theory 1 is fundamentally degenerate because its core prediction (linear rank weighting) is robustly contradicted by the data in its own targeted experiments.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Take-The-Best' (TTB) or 'Lexicographic' theory. Instead of compensatory weighting (either cardinal or rank-based), subjects may sequentially compare features in order of validity and make a decision based solely on the first feature that discriminates between the options. To account for the soft probabilities observed, the theory should include a lapse rate or probability of feature-evaluation error. This provides a non-compensatory contrast to the cardinal WADD model."
}
```

## Usage

```json
{
  "prompt_token_count": 5373,
  "candidates_token_count": 296,
  "total_token_count": 6521
}
```
