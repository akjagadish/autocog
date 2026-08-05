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
People make decisions by computing a weighted sum of the features for each option, using the cue validities as the weights. This Weighted Additive Model (WADD) is fully compensatory. Instead of a translation-invariant Softmax mapping, choice probabilities are generated using a ratio-based Luce choice rule (power rule) over the options' weighted sums, along with a baseline rate of random guessing. This allows choices to be sensitive to the ratio of evidence rather than just the absolute difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Ratio-based Luce choice rule (power rule)
    scores = np.maximum(scores, 1e-9)
    v = scores ** theta
    p_core = v / np.sum(v)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
Under the Tallying heuristic, the number of strictly winning features for the favored option is exactly 1 (and 0 for the other) across all four of these trials. Thus, Tallying predicts identical choice probabilities for the better option in all four trials, making the difference metric zero in expectation. Under the Advocated WADD theory (using a ratio-based Luce choice rule), the magnitude of the weighted sum heavily impacts choice probabilities. Trials 4 and 8 have a much larger ratio of evidence (~2.4) compared to Trials 1 and 12 (~1.3), so WADD predicts a significantly higher accuracy on the former pair, making the metric strongly positive.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify specific trials by their feature string representations
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # High WADD ratio trials (Tallying difference is exactly 1)
    # Trial 4: A=[0,0,0,1], B=[0,1,0,1] -> WADD ratio B/A = 1.35 / 0.55 = 2.45
    t4 = data[(data['a_str'] == '0001') & (data['b_str'] == '0101')]
    # Trial 8: A=[0,0,1,1], B=[0,0,0,1] -> WADD ratio A/B = 1.33 / 0.55 = 2.41
    t8 = data[(data['a_str'] == '0011') & (data['b_str'] == '0001')]
    
    # Low WADD ratio trials (Tallying difference is exactly 1)
    # Trial 1: A=[1,0,1,0], B=[1,0,1,1] -> WADD ratio B/A = 2.28 / 1.73 = 1.31
    t1 = data[(data['a_str'] == '1010') & (data['b_str'] == '1011')]
    # Trial 12: A=[1,1,0,0], B=[1,1,0,1] -> WADD ratio B/A = 2.30 / 1.75 = 1.31
    t12 = data[(data['a_str'] == '1100') & (data['b_str'] == '1101')]
    
    p_b_t4 = t4['response'].mean() if len(t4) > 0 else 0.5
    p_a_t8 = 1.0 - t8['response'].mean() if len(t8) > 0 else 0.5
    
    p_b_t1 = t1['response'].mean() if len(t1) > 0 else 0.5
    p_b_t12 = t12['response'].mean() if len(t12) > 0 else 0.5
    
    high_ratio_acc = (p_b_t4 + p_a_t8) / 2.0
    low_ratio_acc = (p_b_t1 + p_b_t12) / 2.0
    
    return float(high_ratio_acc - low_ratio_acc)

### RESULTS
- Predicted under pi_3 (simulated): 0.1000 (var=0.0253)
- Predicted under pi_2 (simulated): 0.0050 (var=0.0301)
- Observed on real data: -0.0067 (var=0.0177)

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
Under Tallying, all features are weighted equally. In Trial 9 (A=[1, 1, 0, 0], B=[0, 0, 0, 1]) and Trial 10 (A=[0, 0, 1, 1], B=[0, 1, 0, 0]), Option A wins on exactly 2 features and Option B wins on exactly 1 feature. Thus, Tallying predicts the exact same probability of choosing A for both trials. Under WADD, the weighted sum for Trial 9 heavily favors A (1.89 vs 0.55, ratio=3.43) because A has the most valid features, whereas for Trial 10 the weighted sum favors A much less (1.21 vs 0.94, ratio=1.28). Thus, WADD predicts a much higher probability of choosing A on Trial 9 than on Trial 10. The metric measures the difference in the rate of choosing B (response=1) between Trial 10 and Trial 9, which equals P(Choose A)_9 - P(Choose A)_10. It will be near 0 for Tallying and strictly positive for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    t9_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0]) & \
              data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 0, 1])
              
    t10_mask = data['option_a_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1]) & \
               data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 0, 0])
               
    resp_9 = data[t9_mask]['response'].mean()
    resp_10 = data[t10_mask]['response'].mean()
    
    if pd.isna(resp_9) or pd.isna(resp_10):
        return 0.0
        
    return float(resp_10 - resp_9)

### RESULTS
- Predicted under pi_3 (simulated): 0.1300 (var=0.0459)
- Predicted under pi_2 (simulated): -0.0233 (var=0.0300)
- Observed on real data: -0.7133 (var=0.0434)

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
    def is_trial_3(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        t1 = (1, 0, 0, 1)
        t2 = (0, 1, 1, 1)
        return (a == t1 and b == t2) or (a == t2 and b == t1)
        
    mask = data.apply(is_trial_3, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    def chose_ttb_option(row):
        a = tuple(row['option_a_ratings'])
        # The TTB-favored option is the one with cue 0 == 1, i.e., (1, 0, 0, 1)
        if a == (1, 0, 0, 1):
            return 1.0 if row['response'] == 0 else 0.0
        else:
            return 1.0 if row['response'] == 1 else 0.0
            
    return float(subset.apply(chose_ttb_option, axis=1).mean())
```

**Observed (real) value:** 0.1933 (var=0.0304)
**Predicted under pi_3:** 0.2867 (var=0.0500)
**Predicted under pi_2:** 0.1367 (var=0.0174)

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
    a_ratings = data['option_a_ratings'].apply(tuple)
    b_ratings = data['option_b_ratings'].apply(tuple)
    
    mask8 = (a_ratings == (0, 1, 0, 1)) & (b_ratings == (1, 1, 0, 0))
    mask11 = (a_ratings == (1, 1, 0, 1)) & (b_ratings == (1, 0, 1, 1))
    
    resp8 = data.loc[mask8, 'response']
    resp11 = data.loc[mask11, 'response']
    
    score8 = (resp8 == 1).mean() if len(resp8) > 0 else 0.5
    score11 = (resp11 == 0).mean() if len(resp11) > 0 else 0.5
    
    return float(score8 + score11)
```

**Observed (real) value:** 0.2067 (var=0.0540)
**Predicted under pi_3:** 1.1433 (var=0.1045)
**Predicted under pi_2:** 0.9967 (var=0.0961)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data strongly contradicts both the standard Weighted Additive (WADD) model and the Tallying heuristic. For example, in Experiment 2, the metric (which measures the difference in choosing Option B between Trial 10 and Trial 9) was heavily negative (-0.7133), meaning subjects chose B much more often on Trial 9 than on Trial 10. However, Option B in Trial 9 only possessed the least valid cue (validity 0.55), while Option A had the two most valid cues (0.95, 0.94). A similar pattern appears in Experiment 4, where subjects overwhelmingly chose options that were favored by the last cue in the array rather than the first. This suggests that subjects are systematically treating the features at the end of the array (the right-most features) as the most important, either due to a right-to-left reading strategy or a strong recency effect in feature presentation, effectively reversing the normative cue validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Recency-Biased Lexicographic' or 'Right-to-Left Take-The-Best' theory. The theory should state that subjects process the feature arrays from right to left (or heavily weight the last features due to recency/layout effects). Instead of using the provided validities, the decision policy evaluates features starting from the last index (n-1) down to the first (0), choosing the option that wins on the first discriminating feature encountered in this right-to-left order. Alternatively, it could be a WADD model where the subjective weights strictly increase from left to right."
}
```

## Usage

```json
{
  "prompt_token_count": 6444,
  "candidates_token_count": 380,
  "total_token_count": 9583
}
```
