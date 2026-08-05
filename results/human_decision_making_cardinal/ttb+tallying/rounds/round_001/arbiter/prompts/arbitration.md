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
Weighted Additive (WADD) theory posits that individuals evaluate options by considering all available features, weighting each feature's cardinal value by its subjective validity or importance. The overall value of an option is the sum of these validity-weighted feature values. Choice is then made by comparing these overall values, with response noise modeled via a softmax function and a base lapse rate. This integrates both cue validity and cardinal magnitudes, distinguishing it from non-compensatory heuristics like Take The Best or unweighted tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    a, b = stim[0], stim[1]
    
    # Calculate weighted additive scores
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    
    # Normalize scores by sum of validities to stabilize softmax scaling across experiments
    sum_val = np.sum(val)
    if sum_val > 0:
        score_a /= sum_val
        score_b /= sum_val
        
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 8, 0, 0]  B=[1, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0]  B=[0, 0, 0, 10]
  trial 4: A=[5, 5, 0, 0]  B=[4, 4, 10, 10]
  trial 5: A=[0, 0, 10, 10]  B=[2, 2, 0, 0]
  trial 6: A=[10, 10, 0, 0]  B=[9, 9, 1, 1]
  trial 7: A=[1, 1, 1, 1]  B=[4, 0, 0, 0]
  trial 8: A=[0, 2, 2, 2]  B=[10, 0, 0, 0]

**Rationale:** To quantitatively dissociate WADD from Tallying, we exploit WADD's sensitivity to both feature validities and cardinal rating magnitudes. Tallying, by contrast, only counts strict feature-wise wins, ignoring both the sizes of the differences and the validities of the features. We use a 4-feature design with a high rating_max (10) and varied validities. The trials are constructed such that one option wins on a single feature by a large margin (strongly favored by WADD), while the other option wins on multiple features by small margins (favored by Tallying). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers one option due to magnitude and validity differences.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of choices that are consistent with WADD on the 5 diagnostic trials where WADD and Tallying make diametrically opposite predictions. WADD will score near 1.0, while Tallying will score near 0.0.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # trial 1: A=[10, 0, 0, 0]  B=[0, 1, 1, 1] -> WADD prefers A (0), Tallying prefers B
        if a[0] == 10 and a[1] == 0 and b[0] == 0:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 2: A=[0, 8, 0, 0]  B=[1, 0, 1, 1] -> WADD prefers A (0), Tallying prefers B
        elif a[0] == 0 and a[1] == 8 and b[0] == 1:
            wadd_consistent += (resp == 0)
            total += 1
        # trial 3: A=[1, 1, 1, 0]  B=[0, 0, 0, 10] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 0 and b[3] == 10:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 7: A=[1, 1, 1, 1]  B=[4, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 1 and a[3] == 1 and b[0] == 4:
            wadd_consistent += (resp == 1)
            total += 1
        # trial 8: A=[0, 2, 2, 2]  B=[10, 0, 0, 0] -> WADD prefers B (1), Tallying prefers A
        elif a[0] == 0 and a[1] == 2 and b[0] == 10:
            wadd_consistent += (resp == 1)
            total += 1
            
    return wadd_consistent / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.8840 (var=0.0047)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: 0.6446 (var=0.0854)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=10):**
  trial 1: A=[5, 5, 5, 5, 5]  B=[10, 4, 4, 4, 4]
  trial 2: A=[9, 9, 4, 4, 4]  B=[5, 5, 5, 5, 5]
  trial 3: A=[10, 0, 10, 0, 10]  B=[0, 10, 0, 10, 0]
  trial 4: A=[8, 8, 2, 2, 5]  B=[2, 2, 8, 8, 5]
  trial 5: A=[3, 3, 3, 3, 3]  B=[2, 2, 2, 2, 10]
  trial 6: A=[1, 9, 9, 9, 9]  B=[10, 8, 8, 8, 8]
  trial 7: A=[7, 2, 7, 2, 7]  B=[6, 8, 6, 8, 6]
  trial 8: A=[5, 6, 5, 6, 5]  B=[6, 5, 6, 5, 6]
  trial 9: A=[4, 8, 4, 8, 4]  B=[5, 2, 5, 2, 5]
  trial 10: A=[8, 8, 8, 0, 0]  B=[9, 9, 9, 10, 10]

**Rationale:** To quantitatively dissociate Tallying from Weighted Additive (WADD) models, we use a 5-feature design with cardinal ratings (up to 10) and varying validities. Tallying ignores both the magnitude of the differences between ratings and the validities of the features, simply counting the number of features on which one option is strictly better than the other. WADD, however, integrates the cardinal magnitudes weighted by the validities. We design trials where one option has a small advantage on a majority of features (favored by Tallying) but the other option has a massive advantage on one or two high-validity features (favored by WADD). We also include trials where Tallying predicts a tie but WADD strongly prefers one option, as well as baseline trials where both models agree.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the option favored by the Weighted Additive (WADD) model on trials where the Tallying heuristic and WADD make strictly opposing predictions. If a subject uses Tallying, they will consistently choose the option favored by the feature count, resulting in a metric value close to 0. If a subject uses WADD, they will consistently choose the option favored by the validity-weighted cardinal sum, resulting in a metric value close to 1. By isolating only the conflict trials and mapping responses to a WADD-compliance score, we achieve maximum discrimination and low within-theory variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    responses = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_diff = np.sum(a > b) - np.sum(b > a)
        wadd_diff = np.sum(a * v) - np.sum(b * v)
        
        if tally_diff > 0 and wadd_diff < 0:
            responses.append(row['response'])
        elif tally_diff < 0 and wadd_diff > 0:
            responses.append(1 - row['response'])
            
    if len(responses) == 0:
        return 0.5
    return float(np.mean(responses))

### RESULTS
- Predicted under pi_3 (simulated): 0.8067 (var=0.0051)
- Predicted under pi_2 (simulated): 0.0000 (var=0.0000)
- Observed on real data: n/a (var=n/a)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[4, 1, 1, 1]  B=[3, 2, 2, 2]
  A=[2, 3, 3, 3]  B=[3, 1, 1, 1]
  A=[5, 5, 1, 1]  B=[5, 4, 2, 2]
  A=[3, 2, 4, 4]  B=[3, 3, 1, 1]
  A=[4, 2, 1, 1]  B=[3, 3, 2, 2]
  A=[1, 4, 4, 4]  B=[2, 2, 2, 2]
  A=[5, 4, 1, 1]  B=[5, 3, 2, 2]
  A=[2, 2, 4, 4]  B=[2, 3, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_choice = None
        # Features are implicitly ordered by validity (highest to lowest)
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        if ttb_choice is not None:
            if row['response'] == ttb_choice:
                ttb_match += 1
            total += 1
    return ttb_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.3308 (var=0.0607)
**Predicted under pi_3:** 0.1652 (var=0.0103)
**Predicted under pi_2:** 0.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[5, 2, 2, 2, 2]  B=[2, 5, 5, 5, 2]
  A=[2, 5, 5, 5, 2]  B=[5, 2, 2, 2, 2]
  A=[5, 5, 2, 2, 2]  B=[5, 2, 5, 5, 5]
  A=[4, 4, 4, 2, 2]  B=[4, 4, 2, 5, 5]
  A=[3, 2, 5, 5, 5]  B=[3, 5, 2, 2, 2]
  A=[4, 3, 3, 3, 3]  B=[3, 4, 4, 4, 4]
  A=[1, 5, 1, 1, 1]  B=[1, 1, 5, 5, 5]
  A=[5, 1, 5, 5, 5]  B=[1, 5, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    def tally_predicts(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(x > y for x, y in zip(a, b))
        b_wins = sum(y > x for x, y in zip(a, b))
        if a_wins > b_wins:
            return 0
        elif b_wins > a_wins:
            return 1
        return -1
    predictions = data.apply(tally_predicts, axis=1)
    return float(np.mean(data['response'] == predictions))
```

**Observed (real) value:** 0.7270 (var=0.0429)
**Predicted under pi_3:** 0.8744 (var=0.0071)
**Predicted under pi_2:** 1.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither WADD (Theory 1) nor Tallying (Theory 2) perfectly captures the observed data, but WADD is consistently closer. In Experiment 1, the observed proportion of WADD-consistent choices is 0.6446, whereas Tallying predicts 0.0000 and WADD predicts 0.8840. In Experiment 4, the observed match to Tallying is 0.7270, while Tallying predicts 1.0000 and WADD predicts 0.8744. Tallying's strict reliance on feature-wise wins completely fails to account for the subjects' sensitivity to large cardinal differences and validities, leading to extreme predictions (0.0 or 1.0). However, WADD also over-predicts the consistency of compensatory choices. The intermediate observed values suggest that human subjects either exhibit significant heterogeneity (some tallying, some using WADD) or employ a hybrid heuristic that considers both feature counts and large magnitude differences.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Strategy Selection' or 'Hybrid' theory to replace Tallying. This new theory should posit that decision-makers rely on a mixture of heuristics depending on the cognitive load or individual differences (e.g., a mixture model where a parameter 'p_wadd' dictates the probability of using a compensatory WADD strategy versus a non-compensatory Tallying strategy on any given trial). Alternatively, propose a 'Threshold' model where subjects default to Tallying unless the cardinal difference on a high-validity feature exceeds a certain threshold, at which point they switch to a compensatory evaluation."
}
```

## Usage

```json
{
  "prompt_token_count": 5523,
  "candidates_token_count": 384,
  "total_token_count": 6781
}
```
