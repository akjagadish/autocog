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
Tallying (Equal-Weight) Theory: People compare two options by simply counting the number of positive cues (features) for each option and choosing the one with the higher count. This theory posits that decision-makers ignore the varying validities of the cues, treating all features as equally important. It is a compensatory strategy because multiple cues can outweigh a single cue, but it is more frugal than a Weighted Additive (WADD) rule because it avoids multiplying by or storing cardinal validities. Response noise is modeled via a softmax function over the tally scores, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: count the number of positive cues for each option
    # Since cues are binary (0 or 1), we can just sum them.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** To dissociate the Tallying (Equal-Weight) theory from the Weighted Additive (WADD) theory, we pit a small number of highly valid cues against a larger number of less valid cues. Tallying ignores cue validities and simply counts the number of positive features, thereby favoring the option with the higher feature count. WADD, on the other hand, computes a validity-weighted sum, allowing a few highly predictive features to outweigh a larger number of weak features. By carefully selecting validities such that the sum of two high-validity cues exceeds the sum of three low-validity cues, we create critical trials where Tallying and WADD make diametrically opposite predictions. We also include trials where both models agree, and trials where feature counts are tied (so Tallying guesses) but WADD has a strong preference.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the critical trials where Tallying (Equal-Weight) and WADD make diametrically opposing predictions. It does so by computing both the unweighted sum of features and the validity-weighted sum, and filtering for trials where the two models favor different options. The metric computes the proportion of choices on these critical trials that align with the Tallying prediction. Data simulated under Tallying will yield a high proportion (close to 1.0, minus lapse rate), while data simulated under WADD will yield a low proportion (close to 0.0, plus lapse rate). By focusing exclusively on the trials that dissociate the theories, the metric maximizes the between-theory difference while minimizing within-theory variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.85, 0.55, 0.5, 0.5])
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    sum_a = np.sum(a_ratings, axis=1)
    sum_b = np.sum(b_ratings, axis=1)
    
    wadd_a = np.dot(a_ratings, validities)
    wadd_b = np.dot(b_ratings, validities)
    
    tally_pred = (sum_b > sum_a).astype(int)
    wadd_pred = (wadd_b > wadd_a).astype(int)
    
    critical_mask = (sum_a != sum_b) & (wadd_a != wadd_b) & (tally_pred != wadd_pred)
    
    if not np.any(critical_mask):
        return 0.5
        
    responses = data['response'].values[critical_mask]
    tally_preds = tally_pred[critical_mask]
    
    return float(np.mean(responses == tally_preds))


### RESULTS
- Predicted under pi_3 (simulated): 0.7333 (var=0.0207)
- Predicted under pi_2 (simulated): 0.4296 (var=0.0575)
- Observed on real data: 0.6783 (var=0.0060)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.55, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To firmly dissociate the Weighted Additive (WADD) theory from the Tallying (Equal-Weight) theory, we pit a small number of high-validity cues against a larger number of low-validity cues. Tallying ignores cue validities and simply counts the number of positive features, thus favoring the option with the higher feature count. WADD computes a validity-weighted sum, allowing fewer, highly predictive features to outweigh a larger number of weak features. By using validities where the sum of the two best cues (0.95 + 0.85 = 1.80) exceeds the sum of the three worst (0.55 + 0.55 + 0.50 = 1.60), we create critical trials where Tallying and WADD make diametrically opposite predictions. We also include trials where feature counts are tied (Tallying guesses, but WADD has a strong preference) and control trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the Weighted Additive (WADD) theory on the critical trials where WADD and Tallying make diametrically opposed predictions. Specifically, it looks at trials where one option has fewer but higher-validity cues (e.g., [1, 1, 0, 0, 0]) and the other has more but lower-validity cues (e.g., [0, 0, 1, 1, 1]). WADD predicts a preference for the former, while Tallying predicts a preference for the latter. The metric will be close to 1 for WADD and close to 0 for Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_consistent += (1 if resp == 0 else 0)
            total += 1
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_consistent += (1 if resp == 1 else 0)
            total += 1
            
    if total == 0:
        return 0.5
    return wadd_consistent / total

### RESULTS
- Predicted under pi_3 (simulated): 0.2508 (var=0.0216)
- Predicted under pi_2 (simulated): 0.5575 (var=0.0876)
- Observed on real data: 0.3467 (var=0.0161)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_cues = data['option_b_ratings'].apply(sum)
    a_cues = data['option_a_ratings'].apply(sum)
    mask = b_cues > a_cues
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.6919 (var=0.0034)
**Predicted under pi_3:** 0.7638 (var=0.0082)
**Predicted under pi_2:** 0.6681 (var=0.0277)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.3892 (var=0.0034)
**Predicted under pi_3:** 0.3602 (var=0.0033)
**Predicted under pi_2:** 0.4569 (var=0.0152)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Tallying (THEORY 1 / pi_3) consistently predicts the observed data better than the Weighted Additive model (THEORY 2 / pi_2). In Experiment 1, the observed metric is 0.678, much closer to Tallying's 0.733 than WADD's 0.430. In Experiment 2, the observed 0.347 is again closer to Tallying's 0.251 than WADD's 0.557. Experiment 4's metric explicitly tracks choices aligning with the highest-validity discriminating cue (the Take-The-Best prediction); the observed value of 0.389 shows subjects are actively choosing against the single best cue in favor of multiple weaker cues, which Tallying captures perfectly (0.360) while WADD overestimates (0.457). Since Tallying is a very strong baseline here, WADD should be replaced with another canonical alternative to fully map the hypothesis space.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose the 'Take-The-Best' (TTB) heuristic as a brand-new theory. TTB is a non-compensatory, lexicographic strategy where the decision-maker searches through cues in descending order of their validity. The search stops at the first cue that discriminates between the two options, and the option with the positive value on that cue is chosen. If no cues discriminate, the model guesses. This provides a stark contrast to both the unweighted compensatory nature of Tallying and the weighted compensatory nature of WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 5442,
  "candidates_token_count": 377,
  "total_token_count": 7087
}
```
