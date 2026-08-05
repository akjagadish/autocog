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
Mixture of Heuristics (Tallying and WADD)

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- WADD ---
    scores_wadd = stim @ val
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.52, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Mixture of Heuristics (Tallying + WADD) from a pure WADD model with subjective weights, we use a 5-feature design with a clear split between high-validity and low-validity features. The trials are constructed to include 'conflict' cases where WADD and Tallying strongly disagree (e.g., fewer high-validity features vs. more low-validity features), 'tie' cases where Tallying predicts guessing but WADD has a strict preference, and 'agreement' cases. Because the Mixture model computes a convex combination of two distinct choice probability distributions (one based on unweighted sums, one on weighted sums), it generates a non-linear profile of choice probabilities across these trial types. A pure WADD model, which uses a single softmax over a single set of linearly weighted sums, cannot simultaneously capture the flat 'tie' behavior of Tallying and the sharp 'conflict' preferences without its subjective weights being pulled in contradictory directions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts two structurally critical trials. In Trial 5, both WADD and Tallying strongly prefer Option A. In Trial 1, WADD still prefers Option A (due to the high validity of the first two features), but Tallying prefers Option B (since it has 3 positive features compared to A's 2). For a pure WADD model, the choice rate for A will be relatively high in both trials, making their difference small to moderate. For the Mixture of Heuristics model, Tallying pulls the choice probability in opposite directions across the two trials (towards A in Trial 5 and towards B in Trial 1), leading to a substantially larger difference in the choice rate for Option A.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Project lists to strings for hashability and comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 1: WADD prefers A (validity diff +0.28), Tallying prefers B (2 vs 3)
    t1_mask = (a_str == '11000') & (b_str == '00111')
    # Trial 5: WADD prefers A (validity diff +0.62), Tallying prefers A (3 vs 1)
    t5_mask = (a_str == '00111') & (b_str == '10000')
    
    # Calculate proportion of times Option A was chosen (response == 0)
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    # Return the difference in choice rates
    return float(p_a_t5 - p_a_t1)

### RESULTS
- Predicted under pi_4 (simulated): 0.4783 (var=0.0874)
- Predicted under pi_2 (simulated): 0.1267 (var=0.2403)
- Observed on real data: 0.3767 (var=0.0362)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 9: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate a pure Weighted Additive (WADD) model with subjective weights from a Mixture of Heuristics (Tallying + objective WADD), this design uses 5 features with a linear spread of validities. The trials comprehensively pit individual higher-validity features against combinations of lower-validity features. The Mixture model is restricted to a convex combination of equal weights (Tallying) and strictly objective validities (WADD). In contrast, the advocated pure WADD model allows for subjective weighting, meaning it can capture non-linear transformations of the validities (e.g., extreme overweighting of the most valid cue, resembling Take-The-Best, or idiosyncratic trade-offs between middle-tier cues) that cannot be expressed as any mixture of objective WADD and Tallying. By spanning diverse trade-offs across the validity hierarchy, we can identify choice probability profiles that fall outside the rigid manifold of the Mixture model but are perfectly captured by WADD with free subjective weights.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
In the experimental design, Trials 1 through 5 are characterized by Option B having more positive features than Option A. Consequently, the Competing Theory (a mixture of Tallying and Objective WADD) will heavily favor Option B on all these trials, as both the equal-weight heuristic and the objective validities point in the same direction. The probability of choosing Option A under the Competing Theory will be near zero, bounded only by the lapse rate (epsilon). In contrast, the Advocated Theory uses subjective, unconstrained weights. A participant might idiosyncratically assign a very high weight to the single feature present in Option A, overcoming the multiple features in Option B. Thus, the Advocated Theory predicts a significantly higher proportion of Option A choices on these trials. Measuring the mean rate of choosing Option A on this specific subset of trials will cleanly discriminate the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials 1 to 5 where option B has more positive features than option A.
    # In the design, these are exactly the trials where sum(A) < sum(B).
    # For these trials, both Tallying and Objective WADD strongly favor option B.
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum < b_sum
    
    subset = data[mask]
    if len(subset) == 0:
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    # We compute it per subject first to ensure stable between-subject variance.
    subj_prop_a = subset.groupby('subject_id')['response'].apply(lambda x: (x == 0).mean())
    
    return float(subj_prop_a.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.1422 (var=0.0144)
- Predicted under pi_2 (simulated): 0.3871 (var=0.0209)
- Observed on real data: 0.3307 (var=0.0056)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = a_sum != b_sum
    if not mask.any():
        return 0.5
    filtered_data = data[mask]
    a_sum_f = a_sum[mask]
    b_sum_f = b_sum[mask]
    chose_fewer = ((filtered_data['response'] == 0) & (a_sum_f < b_sum_f)) | ((filtered_data['response'] == 1) & (b_sum_f < a_sum_f))
    return float(chose_fewer.mean())
```

**Observed (real) value:** 0.3172 (var=0.0051)
**Predicted under pi_4:** 0.0847 (var=0.0078)
**Predicted under pi_2:** 0.3389 (var=0.0293)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    diff = a - b
    # Find the index of the first discriminating cue (highest validity first)
    idx = (diff != 0).argmax(axis=1)
    # TTB chooses option A (0) if A > B on this cue, else option B (1)
    ttb_winner = np.where(diff[np.arange(len(diff)), idx] > 0, 0, 1)
    return float((data['response'].values == ttb_winner).mean())
```

**Observed (real) value:** 0.3383 (var=0.0035)
**Predicted under pi_4:** 0.1483 (var=0.0062)
**Predicted under pi_2:** 0.3296 (var=0.0225)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    
    conflict_mask = ((sum_a > sum_b) & (wadd_a < wadd_b)) | ((sum_a < sum_b) & (wadd_a > wadd_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    sum_a_conf = sum_a[conflict_mask]
    sum_b_conf = sum_b[conflict_mask]
    responses = data['response'].values[conflict_mask]
    
    tallying_predictions = (sum_a_conf < sum_b_conf).astype(int)
    
    return float(np.mean(responses == tallying_predictions))
```

**Observed (real) value:** 0.6700 (var=0.0082)
**Predicted under pi_4:** 0.5317 (var=0.0512)
**Predicted under pi_2:** 0.3683 (var=0.0688)

### Experiment 6
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to strings for easy comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Identify trials where one option has fewer but more valid features (11000)
    # and the other has more but less valid features (00111)
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    total_critical_trials = mask1.sum() + mask2.sum()
    if total_critical_trials == 0:
        return 0.5
        
    # Count how often the subject chose the WADD-favored option (11000)
    wadd_choices = (data.loc[mask1, 'response'] == 0).sum() + (data.loc[mask2, 'response'] == 1).sum()
    
    return float(wadd_choices / total_critical_trials)
```

**Observed (real) value:** 0.3233 (var=0.0099)
**Predicted under pi_4:** 0.3425 (var=0.0408)
**Predicted under pi_2:** 0.5450 (var=0.0989)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, neither theory is universally successful. THEORY 1 (Mixture of Tallying and Objective WADD) accurately captures behavior in Experiments 1, 5, and 6, where the conflict between Tallying and objective validities drives choice rates. However, it fails drastically on Experiments 2, 3, and 4. Conversely, THEORY 2 (Subjective WADD) performs very well on Experiments 2, 3, and 4, indicating that participants likely employ subjective cue weighting (potentially approximating Take-The-Best through extreme weights), but it fails to capture the strong Tallying effects observed in Experiments 1, 5, and 6. Because each theory captures roughly half of the experimental phenomena while failing on the rest, we need a new theory that integrates both mechanisms.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory: 'Mixture of Subjective WADD and Tallying'. This theory posits that decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule with *subjective* (free) weights, rather than objective validities. By allowing the WADD component to use subjective weights, the model can capture the Take-The-Best-like behavior seen in Experiments 2, 3, and 4. By retaining the Tallying component, it can simultaneously explain the strong equal-weighting pull observed in Experiments 1, 5, and 6. This unifies the successful elements of both prior theories."
}
```

## Usage

```json
{
  "prompt_token_count": 6911,
  "candidates_token_count": 350,
  "total_token_count": 8224
}
```
