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
Weighted Additive (WADD) Decision Making with Option Bias

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate the weighted sum of features for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    # Apply option bias to Option A's score
    option_bias = float(parameters.get("option_bias", 0.0))
    scores = np.array([score_a + option_bias, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Weighted Additive (WADD) from Tallying, this design uses five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. Option bias in WADD will further shift preferences asymmetrically, a pattern Tallying cannot capture.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of WADD-consistent choices specifically on 'opposition' trials (Trials 3 and 4). On these trials, one option wins on the two highest-validity features (WADD preference), while the other option wins on the three lowest-validity features (Tallying preference). By isolating these trials, the metric directly captures the theoretical divergence, returning values systematically >0.5 for WADD and <0.5 for Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    mask3 = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    mask4 = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    wadd_choices = 0
    total = 0
    
    if mask3.any():
        wadd_choices += (data.loc[mask3, 'response'] == 0).sum()
        total += mask3.sum()
        
    if mask4.any():
        wadd_choices += (data.loc[mask4, 'response'] == 1).sum()
        total += mask4.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_choices / total)

### RESULTS
- Predicted under pi_3 (simulated): 0.5433 (var=0.0089)
- Predicted under pi_2 (simulated): 0.1400 (var=0.0151)
- Observed on real data: 0.1500 (var=0.0147)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Tallying from Weighted Additive (WADD) with Option Bias, we use five features with a large spread in validities. Tallying ignores validities entirely, counting only the number of features on which one option strictly beats the other. WADD, by contrast, weights each feature by its validity. The trials include 'opposition' cases where one option wins on a few high-validity features while the other wins on a larger number of low-validity features (WADD prefers the former, Tallying prefers the latter). We also include 'tie' trials for Tallying where both options win on an equal number of features (leading to guessing under Tallying), but WADD strongly prefers the option that wins on the higher-validity features. This mixture isolates the feature-summing behavior of Tallying from the validity-weighted integration of WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses on the two 'opposition' trials where Tallying and WADD make diametrically opposed predictions. Tallying chooses the option with more positive features (ignoring validity), while WADD chooses the option with fewer but higher-validity features. The metric returns the proportion of choices consistent with Tallying on these trials. It should be close to 1 for Tallying and close to 0 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A=[0,0,1,1,1], B=[1,1,0,0,0]
    # Tallying prefers A (3 features vs 2), WADD prefers B (val 1.65 vs 1.85)
    t1_mask = (a_tuples == (0, 0, 1, 1, 1))
    
    # Trial 2: A=[1,1,0,0,0], B=[0,0,1,1,1]
    # Tallying prefers B, WADD prefers A
    t2_mask = (a_tuples == (1, 1, 0, 0, 0))
    
    t1_responses = data.loc[t1_mask, 'response']
    t2_responses = data.loc[t2_mask, 'response']
    
    val = 0.0
    count = 0
    if len(t1_responses) > 0:
        val += (t1_responses == 0).mean()
        count += 1
    if len(t2_responses) > 0:
        val += (t2_responses == 1).mean()
        count += 1
        
    return float(val / count) if count > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.4942 (var=0.0075)
- Predicted under pi_2 (simulated): 0.8671 (var=0.0080)
- Observed on real data: 0.8583 (var=0.0155)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    # Find the highest validity feature where A and B differ.
    # Since features are ordered by descending validity, this is the first non-zero difference.
    diff = a_mat - b_mat
    first_diff_idx = np.argmax(np.abs(diff), axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    # TTB predicts A (0) if A > B on the discriminating cue, else B (1)
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # Return the proportion of choices that align with TTB's prediction
    return float(np.mean(ttb_pred == resp))
```

**Observed (real) value:** 0.2512 (var=0.0077)
**Predicted under pi_3:** 0.4027 (var=0.0062)
**Predicted under pi_2:** 0.2373 (var=0.0059)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_trials = a_wins != b_wins
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (b_wins > a_wins).astype(int)
    responses = data['response'].values
    
    match = (tally_pred[valid_trials] == responses[valid_trials])
    return float(np.mean(match))
```

**Observed (real) value:** 0.6089 (var=0.0019)
**Predicted under pi_3:** 0.6639 (var=0.0115)
**Predicted under pi_2:** 0.8428 (var=0.0102)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1, 2, and 3, Theory 2 (Tallying) provides an exceptionally tight fit to the observed data, accurately predicting the strong reliance on the sheer count of winning features over validity-weighted integration. Theory 1 (WADD) completely misses these patterns, predicting near-chance or opposite behavior. However, in Experiment 4, the observed Tallying consistency drops to 0.6089, whereas Theory 2 predicts a much higher 0.8428. This suggests that while Tallying is the dominant heuristic, it does not perfectly account for all behavior, as some subjects or trials elicit more compensatory (WADD-like) or validity-sensitive strategies. Theory 1 is fundamentally inadequate as a standalone explanation, but its underlying mechanism might be part of a broader repertoire.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Selection' or 'Mixture' theory. Instead of assuming all subjects use a single heuristic, this theory should posit that decision-makers probabilistically draw from a repertoire of strategies (e.g., Tallying and WADD, or Tallying and Take-The-Best) depending on trial characteristics or individual differences. A mixture model with a parameter controlling the probability of deploying Tallying vs. a validity-weighted strategy will maintain the strong Tallying predictions for Exps 1-3 while naturally capturing the tempered Tallying consistency observed in Exp 4."
}
```

## Usage

```json
{
  "prompt_token_count": 5343,
  "candidates_token_count": 332,
  "total_token_count": 6690
}
```
