# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

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
    return int(np.argmax(probabilities))


## THEORY 2 — pi_4
Weighted Tallying heuristic: People evaluate options by comparing them feature-by-feature, but instead of just counting the number of winning features (as in regular Tallying), they weight each win by the feature's subjective validity or importance. This heuristic discards cardinal magnitudes (the size of the difference between feature values is ignored), making it robust to extreme outlier values that would skew an additive model. However, unlike unweighted Tallying, it incorporates the known validities of the cues, allowing more important features to break ties or even override a larger count of less important features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Weighted Tallying expects a (2, n_features) stimulus; got {stim.shape}.")
    
    a, b = stim[0], stim[1]
    v = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate strict wins for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Weight the wins by the validities
    score_a = np.sum(a_wins * v)
    score_b = np.sum(b_wins * v)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.5, 0.5, 0.5]

**rating_max:** 5

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 4, 4, 4]  B=[4, 4, 1, 1, 1]
  trial 2: A=[5, 5, 2, 2, 2]  B=[2, 2, 4, 4, 4]
  trial 3: A=[4, 2, 4, 2, 3]  B=[2, 4, 2, 4, 3]
  trial 4: A=[2, 3, 4, 4, 2]  B=[4, 3, 2, 2, 4]
  trial 5: A=[4, 2, 3, 3, 3]  B=[2, 4, 3, 3, 3]

**Rationale:** To dissociate unweighted Tallying from Weighted Tallying, we use 5 features with a steep validity gradient. Tallying simply counts the number of winning features, ignoring validities. Weighted Tallying weights each win by its validity. We include trials where one option wins on a majority of low-validity features (favored by Tallying) while the other option wins on fewer high-validity features (favored by Weighted Tallying). We also include trials that produce a perfect tie in Tallying (equal number of winning features) but a strong preference in Weighted Tallying due to validity differences.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with the Weighted Tallying heuristic on the two most diagnostic trials (Trial 1 and Trial 2). In Trial 1, Option A wins on 3 low-validity features while Option B wins on 2 high-validity features. Tallying picks A, but Weighted Tallying picks B. Trial 2 is the mirror image. By averaging the proportion of 'Weighted Tallying' choices across these two trial types, we expect a value near 0 for the unweighted Tallying theory and near 1 for the Weighted Tallying theory, providing maximal discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Trial 1: A=[1, 1, 4, 4, 4], B=[4, 4, 1, 1, 1]
    # Tallying prefers A (3 vs 2). Weighted Tallying prefers B (1.7 vs 1.5).
    t1_mask = data['option_a_ratings'].apply(lambda x: x[0] == 1 and x[1] == 1 and x[2] == 4)
    
    # Trial 2: A=[5, 5, 2, 2, 2], B=[2, 2, 4, 4, 4]
    # Tallying prefers B (3 vs 2). Weighted Tallying prefers A (1.7 vs 1.5).
    t2_mask = data['option_a_ratings'].apply(lambda x: x[0] == 5 and x[1] == 5 and x[2] == 2)
    
    # Proportion of times choosing B (response == 1) in Trial 1
    t1_data = data.loc[t1_mask, 'response']
    val1 = t1_data.mean() if len(t1_data) > 0 else 0.5
    
    # Proportion of times choosing A (response == 0) in Trial 2
    t2_data = data.loc[t2_mask, 'response']
    val2 = (1.0 - t2_data.mean()) if len(t2_data) > 0 else 0.5
    
    return float((val1 + val2) / 2.0)

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_4 (simulated): 0.7405 (var=0.0148)
- Observed on real data: 0.4200 (var=0.2436)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**rating_max:** 1

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To dissociate Weighted Tallying from unweighted Tallying, we use binary features (rating_max=1) with a set of validities that clearly distinguish feature counts from validity-weighted counts. Tallying simply counts the number of winning features, ignoring validities. Weighted Tallying weights each win by its validity. We include trials where one option wins on a larger number of low-validity features (favored by Tallying) while the other option wins on fewer, high-validity features (favored by Weighted Tallying). We also include trials that produce a perfect tie in Tallying (equal number of winning features) but a strong preference in Weighted Tallying due to validity differences.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
In Trial 1 (A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]), Tallying predicts a strong preference for B because it wins on 3 features compared to A's 2 features. Weighted Tallying, however, predicts a preference for A because the sum of validities for A's winning features (0.95 + 0.9 = 1.85) is greater than the sum of validities for B's winning features (0.6 + 0.55 + 0.5 = 1.65). By measuring the proportion of times Option A is chosen on this specific trial, we obtain a metric that will be near 1 for Weighted Tallying and near 0 for Tallying, providing maximum discrimination with minimal within-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify Trial 1 where A=[1, 1, 0, 0, 0] and B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.5
    trial_1_data = data[is_trial_1]
    # Calculate the proportion of times Option A was chosen (response == 0)
    prob_a = (trial_1_data['response'] == 0).mean()
    return float(prob_a)

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_4 (simulated): 0.7411 (var=0.0266)
- Observed on real data: 0.2800 (var=0.2016)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 4, 4, 4]  B=[5, 3, 3, 3]
  A=[5, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[2, 5, 2, 2]  B=[3, 1, 3, 3]
  A=[4, 4, 1, 5]  B=[5, 5, 2, 0]
  A=[0, 0, 5, 5]  B=[1, 1, 4, 4]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        r = row['response']
        if a[0] == 1:
            matches.append(r == 0)
        elif a[0] == 5:
            matches.append(r == 1)
        elif a[0] == 2:
            matches.append(r == 1)
        elif a[0] == 4:
            matches.append(r == 1)
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7250 (var=0.0281)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_4:** 0.8592 (var=0.0096)

### Experiment 4
**Design**
  A=[5, 5, 5, 5]  B=[0, 6, 6, 6]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  A=[10, 0, 0, 0]  B=[0, 5, 5, 4]
  A=[8, 4, 4, 4]  B=[5, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[0, 4, 4, 4], B=[10, 0, 0, 0]
    # WADD prefers B (response=1) because 10*0.9 = 9.0 > 4*(0.8+0.6+0.5) = 7.6
    # Tallying prefers A (response=0) because A wins on 3 features vs B's 1
    
    # Identify trial 2 by checking the first element of option A
    is_trial_2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 4, 4, 4))
    trial_data = data[is_trial_2]
    
    if len(trial_data) == 0:
        return 0.5
        
    return float(trial_data['response'].mean())
```

**Observed (real) value:** 0.2600 (var=0.1924)
**Predicted under pi_1:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.1537 (var=0.0208)

### Experiment 5
**Design**
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  A=[10, 10, 0, 0]  B=[0, 0, 1, 1]
  A=[5, 5, 5, 5]  B=[4, 4, 4, 8]
  A=[9, 9, 0, 0]  B=[10, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            matches.append(1 if row['response'] == tally_pref else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.7440 (var=0.0257)
**Predicted under pi_1:** 1.0000 (var=0.0000)
**Predicted under pi_4:** 0.8658 (var=0.0049)

### Experiment 6
**Design**
  A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  A=[8, 8, 8, 0]  B=[7, 7, 7, 10]
  A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  A=[10, 10, 0, 0]  B=[0, 0, 2, 2]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  A=[2, 2, 0, 0]  B=[0, 0, 10, 10]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    eqw_predicts_a = a_sums > b_sums
    eqw_predicts_b = b_sums > a_sums
    
    match_a = eqw_predicts_a & (data['response'] == 0)
    match_b = eqw_predicts_b & (data['response'] == 1)
    
    valid_trials = eqw_predicts_a | eqw_predicts_b
    if not valid_trials.any():
        return 0.5
        
    return float((match_a | match_b).sum() / valid_trials.sum())
```

**Observed (real) value:** 0.3680 (var=0.0278)
**Predicted under pi_1:** 0.2000 (var=0.0000)
**Predicted under pi_4:** 0.2812 (var=0.0030)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Theory 1 (unweighted Tallying) makes extreme deterministic predictions (0.0 or 1.0) that fail to capture the intermediate observed values. Theory 2 (Weighted Tallying) generally performs better, as its predictions are closer to the observed metrics in most experiments (e.g., Exp 3, 4, 5, 6). However, both models fail to fully capture the observed behavior, which often falls between the predictions of unweighted and weighted tallying, or shows sensitivity to cardinal magnitudes. The complete failure of Theory 1 to account for validity-driven choices suggests it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory that incorporates a different mechanism, such as Take-The-Best (TTB) or Weighted Additive (WADD). For example, TTB would search through cues in descending order of validity and stop at the first cue that discriminates between the options, ignoring all other cues. Alternatively, WADD would multiply the cardinal feature values by their validities and sum them, accounting for both magnitude and importance. Either would provide a stronger competitor to Weighted Tallying than the current unweighted Tallying model."
}
```

## Usage

```json
{
  "prompt_token_count": 5873,
  "candidates_token_count": 282,
  "total_token_count": 7222
}
```
