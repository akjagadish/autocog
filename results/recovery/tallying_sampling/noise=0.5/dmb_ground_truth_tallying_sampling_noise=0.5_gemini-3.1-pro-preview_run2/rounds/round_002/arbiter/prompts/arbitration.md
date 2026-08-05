# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Tallying with Validity Bias: People primarily make decisions by tallying the number of positive features for each option (treating all cues as roughly equal), but they also incorporate a secondary, subtle bias based on the explicit cue validities. This means that while the overall count of positive features dominates the choice, the validities of those features can break ties or slightly adjust the strength of preference, blending a compensatory equal-weight strategy with a sensitivity to information quality.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Tallying component: equal weight sum
    tally_scores = stim.sum(axis=1)
    
    # Validity component: weighted sum using explicit validities
    validities = np.asarray(parameters["validities"], dtype=float)
    val_scores = stim @ validities
    
    # Blend the two strategies
    w_val = float(parameters["w_val"])
    scores = (1.0 - w_val) * tally_scores + w_val * val_scores
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the blended scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 7: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate pure Tallying from Tallying with Validity Bias, this design orthogonally manipulates the difference in feature counts (Tally Difference) and the difference in validity-weighted sums (Validity Difference). Pure Tallying predicts that choice probabilities are determined entirely by the Tally Difference, resulting in flat, step-like choice functions where variations in Validity Difference have zero effect (e.g., all Tally Ties yield 50/50 choices). In contrast, Tallying with Validity Bias predicts a graded response: within any given Tally Difference (including ties), the choice probability will shift systematically according to the Validity Difference. By including trials where Tally and Validity conflict, align, or tie, we can isolate the precise contribution of the validity bias.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
To robustly isolate the validity bias from the dominant tallying heuristic, this metric computes the within-subject residual choice probability after controlling for the exact tally difference. For every subject and every tally difference level, we compute their empirical mean probability of choosing Option B. We then subtract this mean from their actual trial-by-trial choices to get a 'residual choice'. Under Pure Tallying, this residual is pure noise and is strictly uncorrelated with the difference in cue validities between the options. Under Tallying with Validity Bias, this residual will be positively correlated with the validity difference. By computing the covariance between the residual choice and the exact validity difference (using the true experimental validities) across all trials, we perfectly partial out the tallying effect, yielding an expected value of exactly 0 for Pure Tallying and a robustly positive value for the Competing Theory. This approach optimally weights all 96 trials, minimizing binomial noise and maximizing the signal-to-noise ratio.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Use the exact validities specified in the experimental design
    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])
    
    # Safely convert lists to numpy arrays for vectorized operations
    a_mat = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b_mat = np.array(data['option_b_ratings'].tolist(), dtype=float)
    
    # Compute the Tally Difference and Validity Difference (Option B - Option A)
    tally_diff = b_mat.sum(axis=1) - a_mat.sum(axis=1)
    val_diff = b_mat.dot(validities) - a_mat.dot(validities)
    
    df = pd.DataFrame({
        'td': tally_diff,
        'vd': val_diff,
        'resp': data['response'].values
    })
    
    # Compute the empirical mean response per subject per tally difference
    if 'subject_id' in data.columns:
        df['subject_id'] = data['subject_id'].values
        td_means = df.groupby(['subject_id', 'td'])['resp'].transform('mean')
    else:
        td_means = df.groupby('td')['resp'].transform('mean')
        
    # The residual choice perfectly partials out the main effect of the Tallying heuristic
    res = df['resp'] - td_means
    
    # The covariance between the residual choice and the validity difference 
    # isolates the unique contribution of the Validity Bias.
    return float(np.mean(res * df['vd']))

### RESULTS
- Predicted under pi_3 (simulated): -0.0017 (var=0.0004)
- Predicted under pi_4 (simulated): 0.0031 (var=0.0003)
- Observed on real data: 0.0049 (var=0.0003)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To cleanly dissociate pure Tallying from Tallying with Validity Bias, this design focuses on trials where the two models predict qualitatively different behavior. Specifically, we include 'Tally Tie' trials where both options have the same number of positive features, but one option's features have much higher validities. Pure Tallying predicts exactly 50/50 choice probabilities on these trials, whereas Tallying with Validity Bias predicts a systematic preference for the option with higher validities. Furthermore, we include 'Conflict' trials where one option has more positive features (favored by Tallying) but the other option has fewer, highly valid features (favored by the Validity component). Comparing these to 'Alignment' trials reveals the graded influence of validity weights that the pure Tallying model completely ignores.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the option with the higher validity score, strictly on trials where the simple tally of positive features is tied. The pure Tallying model predicts random guessing (0.5) on these trials because the tally counts are equal. In contrast, the Tallying with Validity Bias model predicts a systematic preference (>0.5) for the option that possesses the higher-validity features. Isolating tie trials provides a clean, highly discriminatory test between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    def get_tally(x):
        return sum(x)
        
    def get_val(x):
        return sum(v * r for v, r in zip(validities, x))
        
    tally_a = data['option_a_ratings'].apply(get_tally)
    tally_b = data['option_b_ratings'].apply(get_tally)
    
    # Only consider trials where the tally scores are equal (ties)
    tie_mask = tally_a == tally_b
    if tie_mask.sum() == 0:
        return 0.5
        
    tie_data = data[tie_mask]
    
    val_a = tie_data['option_a_ratings'].apply(get_val)
    val_b = tie_data['option_b_ratings'].apply(get_val)
    
    # Determine which option has the higher validity sum
    higher_val_is_b = (val_b > val_a).astype(int)
    
    # Calculate the proportion of choices that align with the higher validity option
    match = (tie_data['response'] == higher_val_is_b).mean()
    
    return float(match)

### RESULTS
- Predicted under pi_3 (simulated): 0.4974 (var=0.0052)
- Predicted under pi_4 (simulated): 0.5800 (var=0.0079)
- Observed on real data: 0.4964 (var=0.0074)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    weights = np.array([1000, 100, 10, 1])
    score_diff = diff @ weights
    ttb_preds = np.where(score_diff > 0, 0, 1)
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.3504 (var=0.0026)
**Predicted under pi_3:** 0.3846 (var=0.0045)
**Predicted under pi_4:** 0.3565 (var=0.0084)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    mask = sum_b > sum_a
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6741 (var=0.0053)
**Predicted under pi_3:** 0.6930 (var=0.0104)
**Predicted under pi_4:** 0.7648 (var=0.0137)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify Trial 3 where both options have exactly 1 positive feature.
    # Tallying predicts a perfect tie (score 1 vs 1) for all subjects.
    # WADD predicts a preference based on subjective weights for the specific features.
    def is_t3(row):
        return sum(row['option_a_ratings']) == 1 and sum(row['option_b_ratings']) == 1
    
    mask = data.apply(is_t3, axis=1)
    t3_data = data[mask]
    
    if len(t3_data) == 0:
        return 0.0
        
    # Calculate each subject's absolute deviation from 0.5 probability of choosing A
    subject_devs = []
    for subj, subj_df in t3_data.groupby('subject_id'):
        pA = (subj_df['response'] == 0).mean()
        subject_devs.append(abs(pA - 0.5))
        
    if not subject_devs:
        return 0.0
        
    return float(np.mean(subject_devs))
```

**Observed (real) value:** 0.1000 (var=0.0041)
**Predicted under pi_3:** 0.0988 (var=0.0038)
**Predicted under pi_4:** 0.0875 (var=0.0048)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_11000(x):
        return tuple(x) == (1, 1, 0, 0, 0)
    
    def is_00111(x):
        return tuple(x) == (0, 0, 1, 1, 1)

    a_11000 = data['option_a_ratings'].apply(is_11000)
    b_00111 = data['option_b_ratings'].apply(is_00111)
    
    a_00111 = data['option_a_ratings'].apply(is_00111)
    b_11000 = data['option_b_ratings'].apply(is_11000)
    
    trial_1 = a_11000 & b_00111
    trial_2 = a_00111 & b_11000
    
    mask = trial_1 | trial_2
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
        
    chose_00111 = (trial_1 & (data['response'] == 1)) | (trial_2 & (data['response'] == 0))
    chose_00111_subset = chose_00111[mask]
    
    return float(chose_00111_subset.mean())
```

**Observed (real) value:** 0.7017 (var=0.0062)
**Predicted under pi_3:** 0.6583 (var=0.0175)
**Predicted under pi_4:** 0.6900 (var=0.0185)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The two experiments present a fascinating contradiction. Experiment 1 shows a small but robust positive correlation between residual choices and validity differences, suggesting some sensitivity to cue validities (favoring Theory 2). However, Experiment 2 specifically isolates 'Tally Tie' trials and finds that subjects choose exactly at chance (0.4964), perfectly matching pure Tallying (Theory 1) and strongly rejecting the idea that validities are used as a tie-breaker (Theory 2 predicted 0.58). This means the validity bias observed in Experiment 1 cannot be a compensatory or tie-breaking mechanism. Instead, the data suggests that validities might only influence choice when they align with or amplify an existing tally difference, or perhaps subjects use a completely different mechanism like a stochastic feature-comparison heuristic where validities dictate the order of feature evaluation rather than acting as additive weights.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory such as 'Stochastic Feature Comparison' or 'Tallying with Attention Allocation'. Instead of a linear additive validity bias, suggest that subjects primarily tally, but their attention to features is probabilistically driven by validities. If a tie occurs, they do not default to validities to break it, but guess randomly because the tallying process itself has concluded. Alternatively, propose a 'Lexicographic Tallying' model where validities only matter if the tally difference is large enough to trigger confidence, or a 'Feature Cancellation' model where shared features are ignored and validities only weight the unique features."
}
```

## Usage

```json
{
  "prompt_token_count": 6524,
  "candidates_token_count": 346,
  "total_token_count": 7843
}
```
