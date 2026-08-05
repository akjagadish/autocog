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
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson 1993).
    # Stimulus is the pair of option feature vectors for the current trial:
    # array-like of shape (2, n_features), row 0 = option A, row 1 = option B.
    # Each option's score is the dot product of its feature vector with the
    # subjective validity weights (which are given directly by the experiment).
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted sum per option (dot product with validities directly)
    scores = stim @ validities

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [1.0, 0.8, 0.6, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 11: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design aims to quantitatively dissociate Tallying from the Weighted Additive (WADD) rule. Validities are chosen to be highly dispersed, distinguishing the feature-counting nature of Tallying from the validity-weighting nature of WADD. Crucial trials pit an option with fewer, but highly valid, features against an option with more, but weakly valid, features. This produces strong divergent predictions (e.g., Tallying prefers A while WADD prefers B). Additionally, trials where Tallying predicts a tie but WADD predicts a preference are included, alongside agreement trials, covering the full spectrum of possible heuristic dynamics.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
On Trial 1, Option A has ratings [0, 0, 1, 1, 1] and Option B has ratings [1, 1, 0, 0, 0]. Tallying evaluates this as A winning on 3 features and B winning on 2 features, leading to a preference for A. In contrast, WADD computes the weighted sums using the validities [1.0, 0.8, 0.6, 0.6, 0.5], giving A a score of 1.7 and B a score of 1.8, leading to a preference for B. This metric calculates the proportion of times the subject chose Option A on this specific trial. Tallying will yield a value significantly greater than 0.5, while WADD will yield a value less than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    # Trial 1 pits an option A with 3 low-validity features against an option B with 2 high-validity features.
    # Tallying prefers A (3 wins vs 2 wins), whereas WADD prefers B (score 1.7 vs 1.8).
    is_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if is_target.sum() == 0:
        return 0.5
    return float((data.loc[is_target, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8300 (var=0.0267)
- Predicted under pi_2_1 (simulated): 0.1650 (var=0.0384)
- Observed on real data: 0.1600 (var=0.0219)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates WADD from Tallying by systematically varying the distribution of features. Validities are spread across a wide range ([0.95, 0.85, 0.75, 0.60, 0.55, 0.50]). Critical trials pit an option with many low-validity features against an option with fewer high-validity features, leading to opposite choices. Additionally, trials where both models agree and trials where Tallying predicts a tie but WADD predicts a strong preference are included to fully map the decision space.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1 and Trial 2, Option A has three low-validity features (wins on features 4, 5, 6) while Option B has two high-validity features (wins on 1 & 2, or 1 & 3). Tallying counts the wins and strictly prefers Option A (3 > 2). WADD, however, weighs the features by validity, and the two high-validity features of B outweigh the three low-validity features of A, leading WADD to prefer Option B. By measuring the choice proportion for A on these specific trials, we expect a value close to 1 for Tallying and close to 0 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify critical trials (Trial 1 and Trial 2) where A is [0, 0, 0, 1, 1, 1]
    # and B has the first feature as 1 (either [1, 1, 0, 0, 0, 0] or [1, 0, 1, 0, 0, 0]).
    is_A_target = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1))
    is_B_starts_1 = data['option_b_ratings'].apply(lambda x: x[0] == 1)
    
    mask = is_A_target & is_B_starts_1
    if not mask.any():
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8600 (var=0.0114)
- Predicted under pi_2_1 (simulated): 0.2058 (var=0.0284)
- Observed on real data: 0.1333 (var=0.0092)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers A (0), WADD prefers B (1)
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        # Trial 2: Tallying prefers B (1), WADD prefers A (0)
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.1311 (var=0.0202)
**Predicted under pi_1:** 0.8544 (var=0.0163)
**Predicted under pi_2_1:** 0.1233 (var=0.0232)

### Experiment 4
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: Tallying prefers A (3 vs 2 wins), WADD prefers B (higher validity sum)
    t1_a = (0, 0, 0, 1, 1, 1)
    
    # Trial 5: Tallying prefers B (3 vs 2 wins), WADD prefers A (higher validity sum)
    t5_a = (1, 1, 0, 0, 0, 0)
    
    t1_mask = a_tuples == t1_a
    t5_mask = a_tuples == t5_a
    
    t1_resp = data.loc[t1_mask, 'response']
    t5_resp = data.loc[t5_mask, 'response']
    
    if len(t1_resp) == 0 or len(t5_resp) == 0:
        return 0.0
        
    p_a_t1 = (t1_resp == 0).mean()
    p_a_t5 = (t5_resp == 0).mean()
    
    # Tallying: P(A|T1) is high, P(A|T5) is low -> Positive difference
    # WADD: P(A|T1) is low, P(A|T5) is high -> Negative difference
    return float(p_a_t1 - p_a_t5)
```

**Observed (real) value:** -0.6650 (var=0.0405)
**Predicted under pi_1:** 0.6958 (var=0.0454)
**Predicted under pi_2_1:** -0.6142 (var=0.0387)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the Weighted Additive model (THEORY 2, pi_2_1) consistently and accurately predicts human behavior, whereas the Tallying heuristic (THEORY 1, pi_1) completely fails. Tallying predicts that subjects will ignore the validities and simply count the number of winning features, leading to predictions that are systematically in the opposite direction of the observed data. Subjects clearly incorporate the provided validities into their decision-making process, weighting high-validity features more heavily than a mere count of low-validity features. Because Tallying is fundamentally mismatched to the data, it should be discarded in favor of a stronger competitor to WADD.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that incorporates validities but operates via a different mechanism than the compensatory Weighted Additive rule. A classic and strong competitor is the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. TTB searches through features in descending order of their validity and stops at the first feature that discriminates between the two options, basing the choice entirely on that single cue without integrating the rest. This will provide a rigorous test of whether subjects are fully integrating all weighted cues (WADD) or relying on the single most important piece of discriminating information (TTB)."
}
```

## Usage

```json
{
  "prompt_token_count": 5964,
  "candidates_token_count": 302,
  "total_token_count": 6833
}
```
