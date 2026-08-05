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
Tallying (Equal Weights) Theory with High Noise: People often ignore the varying validities or importances of different pieces of information, instead simply counting the number of positive features (or cues) favoring each option. However, when faced with difficult compensatory tradeoffs where one option has the best cue but the other has many weaker cues, subjects may experience high decision conflict and rely heavily on guessing. Thus, the choice process is characterized by a high degree of noise, pulling choice probabilities very close to chance (0.50).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Convert stimulus to a numpy array of shape (2, n_features)
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying rule: sum the unweighted positive features for each option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallied scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse (guessing) distribution
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
**Validities (n_features=5):** [0.9, 0.8, 0.5, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design systematically pits Tallying with High Noise against Weighted Additive (WADD) by using 5 features with a steep drop-off in validities. In the critical trial (A: [1,1,0,0,0], B: [0,0,1,1,1]), Tallying predicts a preference for B because it has more positive features (3 vs 2), whereas WADD predicts a preference for A because the two high-validity features outweigh the three low-validity features. Furthermore, Tallying with High Noise predicts that across all trials, choices will be heavily regressed toward 0.50 (guessing) due to high decision conflict and noise, while WADD predicts strong, deterministic choices (probabilities near 1.0 or 0.0) whenever there is a clear difference in the weighted sums.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Tallying with High Noise predicts that choices will be heavily regressed toward 0.50 due to high decision conflict and noise, meaning the choice proportions for each trial type will be close to 0.5. In contrast, WADD predicts much stronger, deterministic choices (probabilities near 1.0 or 0.0) whenever there is a clear difference in weighted sums. By calculating the mean absolute deviation of each trial type's choice proportion from 0.5, we can reliably distinguish the highly noisy, guessing-like behavior of Tallying from the more deterministic behavior of WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # Create a string representation for grouping by trial type
    data = data.copy()
    data['trial_key'] = data.apply(lambda row: str(tuple(row['option_a_ratings'])) + '_' + str(tuple(row['option_b_ratings'])), axis=1)
    
    # Calculate the mean response (probability of choosing B) for each subject and trial type
    subject_trial_means = data.groupby(['subject_id', 'trial_key'])['response'].mean().reset_index()
    
    # Calculate the absolute deviation from 0.5 (guessing)
    subject_trial_means['dev'] = (subject_trial_means['response'] - 0.5).abs()
    
    # Average across trial types for each subject, then average over subjects
    return float(subject_trial_means.groupby('subject_id')['dev'].mean().mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1165 (var=0.0011)
- Predicted under pi_2 (simulated): 0.2760 (var=0.0095)
- Observed on real data: 0.1071 (var=0.0007)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Rationale:** This design uses 6 features to create a quantitative dissociation between Weighted Additive (WADD) and Tallying with High Noise, avoiding the previously explored 5-feature steep drop-off design. By employing three high-validity cues and three low-validity cues, we can construct trials where Tallying predicts a tie (equal number of positive features) but WADD predicts a strong, deterministic preference for the option with high-validity features. Furthermore, we include critical dissociation trials where Tallying strictly prefers the option with more positive features (e.g., 4 vs 3), but WADD prefers the option with fewer, higher-validity features. Under the competing Tallying with High Noise theory, choices will be weakly distributed around 0.50 across all trials, whereas WADD predicts strong preferences that scale with the weighted sum differences.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject's choice aligns with the deterministic prediction of the Weighted Additive (WADD) model. Since the experimental design includes several trials where Tallying predicts a tie (but WADD has a strong preference) and trials where Tallying and WADD have opposite strict preferences, the WADD model will consistently score very high on this metric. In contrast, the competing Tallying with High Noise theory will produce choices that are close to random chance (0.50) on most trials, resulting in a significantly lower score.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    wadd_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        if score_a > score_b:
            pred = 0
        elif score_b > score_a:
            pred = 1
        else:
            continue
        if row['response'] == pred:
            wadd_matches += 1
        total += 1
    return wadd_matches / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.4861 (var=0.0026)
- Predicted under pi_2 (simulated): 0.6055 (var=0.0238)
- Observed on real data: 0.5065 (var=0.0027)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the total number of positive cues for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify critical compensatory trials where Option B has more positive cues overall,
    # but Option A possesses the single most valid discriminating cue.
    mask = sum_b > sum_a
    
    if not mask.any():
        return 0.0
        
    # Return the proportion of times the subject chose Option B (response == 1)
    # TTB will consistently choose A (yielding ~0.0) because it only looks at the top valid cue.
    # WADD will frequently choose B (yielding a higher value) because the numerous lower-validity cues compensate.
    return float(data.loc[mask, 'response'].mean())

```

**Observed (real) value:** 0.4961 (var=0.0021)
**Predicted under pi_3:** 0.5386 (var=0.0041)
**Predicted under pi_2:** 0.7075 (var=0.0196)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # The experimental design is set up such that Take The Best (TTB) almost
    # always chooses option A (response = 0) because option A always has the
    # advantage on the single most valid discriminating cue.
    # Conversely, Weighted Additive (WADD) will frequently choose option B 
    # (response = 1) because option B has a large number of lower-validity 
    # cues that cumulatively outweigh option A's single best cue.
    # Thus, the simple overall proportion of choosing option B perfectly 
    # discriminates the two theories.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Predicted under pi_3:** 0.5337 (var=0.0030)
**Predicted under pi_2:** 0.5729 (var=0.0092)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed data consistently shows weak, noisy preferences closely hovering around chance (0.50). THEORY 1 (pi_3: Tallying with High Noise) captures this extremely well, with predictions closely tracking the observed metrics (e.g., Exp 2 observed 0.5065 vs pi_3 0.4861; Exp 4 observed 0.4996 vs pi_3 0.5337). THEORY 2 (pi_2: WADD) consistently predicts much stronger, deterministic compensatory choices that fail to materialize in the real data, heavily overestimating the degree to which subjects integrate all cues (e.g., Exp 3 observed 0.4961 vs pi_2 0.7075). Given WADD's systematic failure to account for the high level of noise and lack of compensatory trade-offs in the data, it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Take The Best (TTB) with High Noise' theory. Unlike WADD (which assumes compensatory integration of all cues) and Tallying (which assumes equal weighting of all cues), TTB assumes subjects only look at the single most valid discriminating cue to make their decision. To capture the observed data, this theory must incorporate a high lapse rate (guessing) to account for the heavy regression toward 0.50 seen across all experiments. This will provide a strong non-compensatory alternative to Tallying while properly matching the high noise floor."
}
```

## Usage

```json
{
  "prompt_token_count": 5155,
  "candidates_token_count": 355,
  "total_token_count": 6452
}
```
