# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
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


## THEORY 2 — pi_2_1
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Use the log-odds of objective validities as the subjective weights
    validities = np.asarray(parameters["validities"], dtype=float)
    validities = np.clip(validities, 1e-5, 1.0 - 1e-5)  # Prevent division by zero or log(0)
    weights = np.log(validities / (1.0 - validities))
    
    # Weighted sum per option (dot product with subjective weights)
    scores = stim @ weights
    
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
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=6):** [0.88, 0.73, 0.73, 0.62, 0.62, 0.62]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** This design carefully selects validities to produce specific log-odds weights (~2.0 for the first feature, ~1.0 for the next two, and ~0.5 for the last three). This allows us to construct trials that precisely dissociate Tallying (which simply counts feature wins) from WADD (which weights by log-odds of validities). The trials include pure reversals (where one heuristic strongly prefers A and the other B), trials where Tallying predicts a perfect tie while WADD has a strong preference, and trials where WADD predicts a perfect tie while Tallying has a clear preference.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific trials where Tallying and WADD make opposite strict predictions (i.e., pure reversals). By computing the proportion of choices on these critical trials that align with Tallying's prediction, we get a measure that should be significantly greater than 0.5 for subjects using Tallying, and significantly less than 0.5 for subjects using WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Approximate log-odds weights for WADD based on validities [0.88, 0.73, 0.73, 0.62, 0.62, 0.62]
    w = np.array([1.9924, 0.9946, 0.9946, 0.4895, 0.4895, 0.4895])
    
    tally_matches = []
    
    for _, row in data.iterrows():
        A = np.array(row['option_a_ratings'])
        B = np.array(row['option_b_ratings'])
        
        # Tallying counts strict wins
        A_tally = np.sum(A > B)
        B_tally = np.sum(B > A)
        
        # WADD uses weighted sums
        A_wadd = np.sum(A * w)
        B_wadd = np.sum(B * w)
        
        # 0 for A, 1 for B, -1 for tie
        tally_pref = 0 if A_tally > B_tally else (1 if B_tally > A_tally else -1)
        wadd_pref = 0 if A_wadd > B_wadd else (1 if B_wadd > A_wadd else -1)
        
        # We only care about trials where the two models make opposite strict predictions
        if tally_pref != -1 and wadd_pref != -1 and tally_pref != wadd_pref:
            if row['response'] == tally_pref:
                tally_matches.append(1.0)
            else:
                tally_matches.append(0.0)
                
    if len(tally_matches) == 0:
        return 0.5
        
    return float(np.mean(tally_matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.8460 (var=0.0095)
- Predicted under pi_2_1 (simulated): 0.1823 (var=0.0070)
- Observed on real data: 0.1467 (var=0.0156)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5333]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]

**Rationale:** By carefully selecting validities, we engineer log-odds weights that allow for precise dissociations between WADD (which sums log-odds weights) and Tallying (which counts strict feature wins). We use validities of [0.90, 0.80, 0.70, 0.60, 0.5333], which yield approximate weights of 2.20, 1.39, 0.85, 0.41, and 0.13. Crucially, the sum of the last three weights (1.39) perfectly equals the second weight. This allows us to create trials where WADD is indifferent but Tallying has a strong preference (3 wins vs 1 win). The design also includes pure reversals where Tallying prefers options with more wins on weak features, but WADD prefers options with fewer wins on strong features, as well as trials where Tallying is indifferent (equal wins) but WADD has a strong preference.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where WADD and Tallying have strictly opposite preferences. It calculates the proportion of choices that align with Tallying on these conflict trials. Data generated by Tallying will yield values significantly greater than 0.5 (approaching 1.0 depending on noise), whereas data generated by WADD will yield values significantly lower than 0.5 (approaching 0.0).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5333])
    weights = np.log(validities / (1.0 - validities))
    
    match_tally = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wadd = np.sum(a * weights)
        b_wadd = np.sum(b * weights)
        
        a_tally = np.sum(a > b)
        b_tally = np.sum(b > a)
        
        wadd_prefers_a = a_wadd > b_wadd + 1e-5
        wadd_prefers_b = b_wadd > a_wadd + 1e-5
        tally_prefers_a = a_tally > b_tally
        tally_prefers_b = b_tally > a_tally
        
        disagree = (wadd_prefers_a and tally_prefers_b) or (wadd_prefers_b and tally_prefers_a)
        
        if disagree:
            chose_a = (row['response'] == 0)
            chose_b = (row['response'] == 1)
            
            if tally_prefers_a and chose_a:
                match_tally.append(1.0)
            elif tally_prefers_b and chose_b:
                match_tally.append(1.0)
            else:
                match_tally.append(0.0)
                
    if len(match_tally) == 0:
        return 0.5
        
    return float(np.mean(match_tally))

### RESULTS
- Predicted under pi_1 (simulated): 0.8400 (var=0.0116)
- Predicted under pi_2_1 (simulated): 0.2258 (var=0.0073)
- Observed on real data: 0.1133 (var=0.0069)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    a_wadd = a_mat @ validities
    b_wadd = b_mat @ validities
    
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    wadd_prefers_a = a_wadd > b_wadd
    wadd_prefers_b = b_wadd > a_wadd
    
    disagree = (tally_prefers_a & wadd_prefers_b) | (tally_prefers_b & wadd_prefers_a)
    
    if not np.any(disagree):
        return 0.5
        
    tally_choice = np.where(tally_prefers_a, 0, 1)
    
    match = (data['response'].values[disagree] == tally_choice[disagree])
    return float(np.mean(match))
```

**Observed (real) value:** 0.1067 (var=0.0120)
**Predicted under pi_1:** 0.8575 (var=0.0138)
**Predicted under pi_2_1:** 0.0633 (var=0.0069)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_vals = np.stack(data['option_a_ratings'].values)
    b_vals = np.stack(data['option_b_ratings'].values)
    
    # Identify trials where Tallying and WADD make strictly opposing predictions.
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0] -> Tallying prefers A, WADD prefers B
    is_trial_1 = (a_vals[:, 0] == 0) & (a_vals[:, 2] == 1) & (b_vals[:, 0] == 1) & (b_vals[:, 2] == 0)
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1] -> Tallying prefers B, WADD prefers A
    is_trial_2 = (a_vals[:, 0] == 1) & (a_vals[:, 2] == 0) & (b_vals[:, 0] == 0) & (b_vals[:, 2] == 1)
    
    mask = is_trial_1 | is_trial_2
    if not mask.any():
        return 0.5
        
    responses = data['response'].values
    wadd_aligned = (is_trial_1 & (responses == 1)) | (is_trial_2 & (responses == 0))
    
    return float(wadd_aligned[mask].mean())
```

**Observed (real) value:** 0.8649 (var=0.0063)
**Predicted under pi_1:** 0.1520 (var=0.0057)
**Predicted under pi_2_1:** 0.9360 (var=0.0056)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed data strongly and consistently favors WADD (THEORY 2) over Tallying (THEORY 1). The metrics specifically isolate conflict trials where the two theories make opposite predictions. The observed values show that subjects overwhelmingly choose the option favored by WADD (e.g., in Experiment 4, 86.5% of choices align with WADD, and in Experiments 1-3, choices matching Tallying are around 10-15%). Tallying fails completely, predicting the exact opposite of the empirical data. Because Tallying is fundamentally contradicted by the data, it should be replaced entirely with a stronger competitor to WADD.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'Take The Best' (TTB) as the new theory. TTB posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue. This provides a strong non-compensatory heuristic alternative to WADD (which integrates all features compensatorily) and differs from Tallying (which integrates all features but ignores validities). TTB will test whether subjects are truly integrating multiple cues or simply relying on the most valid discriminating cue."
}
```

## Usage

```json
{
  "prompt_token_count": 6438,
  "candidates_token_count": 302,
  "total_token_count": 7535
}
```
