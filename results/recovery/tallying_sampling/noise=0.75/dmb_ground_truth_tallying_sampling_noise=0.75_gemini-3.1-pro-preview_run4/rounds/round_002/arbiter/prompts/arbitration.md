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
Non-linear Weighting Model: Instead of probabilistically mixing distinct heuristics like Take The Best and Weighted Additive, decision-makers rely on a single compensatory integration process where cue validities are transformed non-linearly. By exponentiating the objective validities by a subjective parameter gamma, the model dynamically scales between equal-weighting (gamma near 0), objective WADD (gamma = 1), and effectively lexicographic/TTB-like behavior (gamma > 1) where the highest validity cue dominates the sum of all lower cues. This allows the model to capture the intermediate TTB-match rates on conflict trials while still producing compensatory ties (and thus guessing) when the non-linearly weighted sums are close.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear transformation of validities
    weights = val ** gamma
    
    # Compensatory integration using subjective weights
    scores = stim @ weights
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To dissociate the Advocated Non-linear Weighting Model from the Competing Weighted Additive (WADD) Model, we exploit the difference in their parameterizations. The Advocated model uses a single parameter, gamma, to non-linearly transform all objective validities simultaneously. This means the trade-offs between any single high-validity cue and multiple lower-validity cues are strictly coupled across the entire validity hierarchy. In contrast, the Competing model possesses an independent subjective weight parameter (w) for every feature, allowing it to decouple these trade-offs (e.g., treating the most valid cue non-compensatorily while treating the rest compensatorily). By systematically pitting single higher-validity cues against different combinations of lower-validity cues across the hierarchy, we test if the transitions between compensatory and non-compensatory choices follow the strict 1-dimensional manifold predicted by a single gamma parameter, or if they require the independent flexibility of the Competing model.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the within-subject variance in the proportion of times Option A is chosen across three structurally similar trials: Trial 1, Trial 3, and Trial 5. In each of these trials, a single higher-validity cue is pitted against the sum of the next two lower-validity cues. Under the Advocated (Single Gamma) model, the relative preferences across these three trials are tightly coupled by the single non-linear exponent, resulting in similar choice probabilities and therefore low within-subject variance. Under the Competing (WADD) model, each feature has an independent subjective weight, allowing the decision-maker to strongly prefer Option A on one trial type while strongly preferring Option B on another. This independent weighting leads to significantly higher within-subject variance across these three trial types.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    def get_trial_type(row):
        if row['A_str'] == '10000' and row['B_str'] == '01100': return 'T1'
        if row['A_str'] == '01000' and row['B_str'] == '00110': return 'T3'
        if row['A_str'] == '00100' and row['B_str'] == '00011': return 'T5'
        return None
        
    df['trial_type'] = df.apply(get_trial_type, axis=1)
    df_trials = df[df['trial_type'].notnull()].copy()
    
    df_trials['chose_A'] = (df_trials['response'] == 0).astype(float)
    
    rates = df_trials.groupby(['subject_id', 'trial_type'])['chose_A'].mean().unstack()
    
    # Calculate the within-subject variance of the choice rates for these 3 trial types
    variances = rates.var(axis=1, ddof=0)
    
    return float(variances.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0124 (var=0.0003)
- Predicted under pi_2 (simulated): 0.0646 (var=0.0042)
- Observed on real data: 0.0226 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 12: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Weighted Additive with free per-feature weights) from the Competing Theory (Non-linear Weighting with a single gamma parameter), we exploit the strict structural constraints imposed by the single gamma parameter. In the Competing model, the subjective weights must strictly preserve the objective validity ordering (since x^gamma > y^gamma for any x > y and gamma > 0), and the trade-offs between cues are tightly coupled across the entire validity hierarchy. In contrast, the Advocated model possesses an independent weighting parameter for each feature, allowing it to easily decouple trade-offs (e.g., heavily discounting one specific high-validity cue while preserving the weights of others) and even invert the subjective importance of cues. We present direct 1-on-1 cue comparisons alongside complex 1-vs-many and 2-vs-many tradeoffs to test if subjects' choices adhere to the rigid 1-dimensional manifold predicted by a single gamma parameter, or if they exhibit the multidimensional flexibility captured by the Advocated model's free weights.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with the higher objective validity cue in direct 1-on-1 comparisons (Trials 1-4). The Competing Theory (Non-linear Weighting) uses a single gamma parameter to scale validities, which strictly preserves the objective validity ordering (since x^gamma > y^gamma for x > y and gamma > 0). Thus, it predicts a very high rate of choosing the objectively better cue. In contrast, the Advocated Theory (Weighted Additive) uses free per-feature weights that are multiplied by the objective validities. Because these weights vary independently and uniformly across features, they can easily invert the subjective importance of the cues, leading to a significantly lower choice rate for the objectively higher validity cue.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings as 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Identify 1-on-1 trials (where each option has exactly one feature = 1)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    mask = (sum_a == 1) & (sum_b == 1)
    
    if not np.any(mask):
        return 0.5
        
    a_subset = a_ratings[mask]
    b_subset = b_ratings[mask]
    resp_subset = data['response'].values[mask]
    
    # The index of the '1' indicates the cue (lower index = higher objective validity)
    idx_a = np.argmax(a_subset, axis=1)
    idx_b = np.argmax(b_subset, axis=1)
    
    # Check if option A has the higher objective validity cue
    a_is_better = idx_a < idx_b
    
    # Check if subject chose option A
    chose_a = (resp_subset == 0)
    
    # Calculate proportion of trials where subject chose the objectively higher validity cue
    correct = (chose_a == a_is_better)
    
    return float(np.mean(correct))

### RESULTS
- Predicted under pi_4 (simulated): 0.6881 (var=0.0196)
- Predicted under pi_2 (simulated): 0.5175 (var=0.0097)
- Observed on real data: 0.4963 (var=0.0088)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_response(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            elif b[j] > a[j]:
                return 1
        return 0.5
    
    ttb_preds = data.apply(get_ttb_response, axis=1)
    return float(np.mean(data['response'] == ttb_preds))
```

**Observed (real) value:** 0.4392 (var=0.0021)
**Predicted under pi_4:** 0.2758 (var=0.0145)
**Predicted under pi_2:** 0.3152 (var=0.0188)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    wadd_a = a_ratings @ validities
    wadd_b = b_ratings @ validities
    wadd_prefers_a = wadd_a > wadd_b
    wadd_prefers_b = wadd_b > wadd_a
    
    diff = a_ratings - b_ratings
    ttb_prefers_a = np.zeros(len(data), dtype=bool)
    ttb_prefers_b = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        for j in range(5):
            if diff[i, j] > 0:
                ttb_prefers_a[i] = True
                break
            elif diff[i, j] < 0:
                ttb_prefers_b[i] = True
                break
                
    conflict = (ttb_prefers_a & wadd_prefers_b) | (ttb_prefers_b & wadd_prefers_a)
    
    if not np.any(conflict):
        return 0.5
        
    response = data['response'].values
    ttb_match = (ttb_prefers_a & (response == 0)) | (ttb_prefers_b & (response == 1))
    
    return float(np.mean(ttb_match[conflict]))
```

**Observed (real) value:** 0.4100 (var=0.0050)
**Predicted under pi_4:** 0.3088 (var=0.0329)
**Predicted under pi_2:** 0.2984 (var=0.0285)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    abs_diff = np.abs(diff)
    
    # TTB winner: first cue where options differ
    first_diff_idx = np.argmax(abs_diff, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # WADD winner
    wadd_a = a_mat @ val
    wadd_b = b_mat @ val
    
    wadd_winner = np.where(wadd_a > wadd_b, 0, 1)
    valid = (wadd_a != wadd_b) & (np.max(abs_diff, axis=1) > 0)
    
    conflict = valid & (ttb_winner != wadd_winner)
    
    if not np.any(conflict):
        return 0.5
        
    responses = data['response'].values
    ttb_chosen = (responses[conflict] == ttb_winner[conflict]).sum()
    
    return float(ttb_chosen / np.sum(conflict))
```

**Observed (real) value:** 0.4044 (var=0.0030)
**Predicted under pi_4:** 0.2916 (var=0.0195)
**Predicted under pi_2:** 0.2669 (var=0.0255)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    wadd_a = a_ratings @ val
    wadd_b = b_ratings @ val
    
    # Identify trials where the objective WADD model predicts a tie
    tied_trials = np.abs(wadd_a - wadd_b) < 1e-5
    if not np.any(tied_trials):
        return 0.5
    
    a_tied = a_ratings[tied_trials]
    b_tied = b_ratings[tied_trials]
    responses = data['response'].values[tied_trials]
    
    # In all tied trials, cue 1 (the highest validity cue) discriminates the options.
    # TTB predicts A (0) if A has cue 1, else B (1).
    ttb_preds = (b_tied[:, 0] > a_tied[:, 0]).astype(int)
    
    return float(np.mean(responses == ttb_preds))
```

**Observed (real) value:** 0.4867 (var=0.0072)
**Predicted under pi_4:** 0.6211 (var=0.0151)
**Predicted under pi_2:** 0.5050 (var=0.0527)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The observed data heavily contradicts both current theories and strongly points toward a Tallying (Equal Weighting) heuristic. In Experiment 2, subjects chose the option with the higher objective validity cue in direct 1-on-1 comparisons at a rate of 0.4963 (essentially chance). This means subjects completely ignored the cue validities when the number of positive features was equal. In Experiment 1, the within-subject variance across trials pitting one higher-validity cue against two lower-validity cues was extremely low (0.0226). Under a Tallying model, Option B always has 2 positive features and Option A has 1, so the subject consistently chooses B, resulting in near-zero variance. Neither the Non-linear Weighting model (which strictly preserves validity order, failing Exp 2) nor the free-weight WADD model (which predicts too much variance in Exp 1) can explain this pattern. Tallying elegantly captures both phenomena.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying' (or Equal Weighting) theory. The theory should posit that decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen; if the tallies are equal, the decision-maker guesses randomly. This perfectly explains the chance-level performance in 1-on-1 single-cue comparisons (Exp 2) and the highly consistent choices (low variance) when one option always has more positive features than the other (Exp 1)."
}
```

## Usage

```json
{
  "prompt_token_count": 7798,
  "candidates_token_count": 359,
  "total_token_count": 9808
}
```
