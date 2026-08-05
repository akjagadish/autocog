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
People use a non-compensatory, lexicographic heuristic known as 'Take-The-Best' (TTB) to make decisions. Instead of integrating all available information (as in weighted additive models) or counting all positive features (as in tallying), decision-makers search through features sequentially in descending order of their validity. The search stops at the very first feature that discriminates between the two options (i.e., one option possesses the feature and the other does not). The option that wins on this single discriminating cue is chosen deterministically, and all remaining lower-validity features are completely ignored. If no features discriminate between the options, the decision-maker guesses randomly. Response noise is modeled purely as a lapse rate (epsilon) where the subject occasionally makes a random guess instead of executing the TTB strategy.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Take-The-Best expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity in descending order
    order = np.argsort(-validities)
    
    # Default to guessing if all features tie
    p_core = np.array([0.5, 0.5])
    
    # Search through features in descending order of validity
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    
    # Blend deterministic choice with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design tests the quantitative dissociation between Take-The-Best (TTB) and Weighted Additive (WADD) models. By using five features with a steeply descending set of validities, we can construct trials where one option wins on the single highest-validity discriminating feature (the only feature TTB considers), while the other option wins on multiple lower-validity features whose combined weight exceeds the highest validity. In these crucial trials, TTB predicts the option with the best single feature, whereas WADD predicts the option with the higher weighted sum of all features. Additional control trials where both models agree, or where the highest-validity feature is tied so TTB relies on the second-best feature but WADD still favors the other option, ensure the full parameter space of both models can be identified.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of trials where the subject's choice aligns with the Take-The-Best (TTB) prediction. Under the TTB theory, this metric will be very close to 1.0 (only lowered by the lapse rate). Under the WADD theory, which disagrees with TTB on 5 out of the 8 trial types in this design, this metric will be significantly lower (around 0.375), creating a massive and stable gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
                
        if pred is not None:
            if resp == pred:
                matches += 1
            total += 1
            
    return matches / total if total > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.8931 (var=0.0064)
- Predicted under pi_2 (simulated): 0.4260 (var=0.0153)
- Observed on real data: 0.8579 (var=0.0072)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the compensatory Weighted Additive (WADD) model from the non-compensatory Take-The-Best (TTB) heuristic. By using a five-feature environment with a linearly descending set of validities, we construct critical trials where Option A wins on the single highest-validity discriminating feature (leading TTB to invariably choose A), but Option B wins on multiple lower-validity features whose combined weight exceeds that of Option A's single advantage (leading WADD to choose B). We also include control trials where both models agree and where they make subtler compensatory trade-offs, ensuring robust estimation of decision noise and strategy parameters across the parameter space.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses Option B on trials where Option B has a greater total number of positive features than Option A. In this experimental design, Take-The-Best (TTB) consistently chooses Option A on these specific trials because Option A always wins on the single most valid discriminating feature. Therefore, TTB predicts a value near 0 (only deviating due to random lapses). In contrast, the Weighted Additive (WADD) model integrates all features, and because Option B has many more positive lower-validity features whose combined weight often exceeds Option A's single advantage, WADD predicts a substantially higher rate of choosing Option B. This creates a large, reliable gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    b_more_than_a = data.apply(lambda row: sum(row['option_b_ratings']) > sum(row['option_a_ratings']), axis=1)
    if not b_more_than_a.any():
        return 0.0
    return float(data.loc[b_more_than_a, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1246 (var=0.0075)
- Predicted under pi_2 (simulated): 0.7087 (var=0.0146)
- Observed on real data: 0.1383 (var=0.0137)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    t1_match = (t1_data['response'] == 0).sum()
    t2_match = (t2_data['response'] == 1).sum()
    
    total = len(t1_data) + len(t2_data)
    
    if total == 0:
        return 0.5
    return float(t1_match + t2_match) / total

```

**Observed (real) value:** 0.1062 (var=0.0063)
**Predicted under pi_3:** 0.1212 (var=0.0080)
**Predicted under pi_2:** 0.3944 (var=0.0773)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_critical = 0
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        
        sum_a = sum(a)
        sum_b = sum(b)
        sum_a_top = sum(a[:3])
        sum_b_top = sum(b[:3])
        
        # Check if this is a critical trial: one option has exactly 2 high-validity features, 
        # while the other has exactly 3 low-validity features.
        if sum_a == 2 and sum_a_top == 2 and sum_b == 3 and sum_b_top == 0:
            total_critical += 1
            if row['response'] == 1:
                tally_choices += 1
        elif sum_b == 2 and sum_b_top == 2 and sum_a == 3 and sum_a_top == 0:
            total_critical += 1
            if row['response'] == 0:
                tally_choices += 1
                
    return tally_choices / total_critical if total_critical > 0 else 0.5
```

**Observed (real) value:** 0.2022 (var=0.0207)
**Predicted under pi_3:** 0.1183 (var=0.0055)
**Predicted under pi_2:** 0.5250 (var=0.0518)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Take-The-Best, pi_3) consistently and dramatically outperforms Theory 2 (Weighted Additive, pi_2). In Experiment 1, 2, and 3, the observed data closely tracks the TTB predictions (e.g., Exp 1: 0.85 observed vs 0.89 TTB vs 0.42 WADD). In Experiment 4, there is a slight deviation where TTB predicts 0.11 but observed is 0.20, suggesting that while TTB captures the vast majority of behavior, there might be a secondary strategy or noise structure not fully captured by a pure deterministic TTB with a uniform lapse rate. WADD consistently fails to capture the non-compensatory nature of the subjects' choices. Since TTB is so dominant, WADD is effectively degenerate in this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Weighted Additive theory with a 'Strategy Mixture' or 'Probabilistic Take-The-Best' theory. Given that pure TTB captures most of the variance but slightly underpredicts deviations in specific cases (like Exp 4), a strong competitor would be a model where subjects predominantly use TTB but with some probability P switch to a simple 'Tallying' (equal-weight) strategy, or where the cue search order is probabilistic (sampled proportional to cue validity) rather than strictly deterministic. This provides a cognitively plausible alternative that can mimic TTB's high success rate while better explaining the residual compensatory-like deviations."
}
```

## Usage

```json
{
  "prompt_token_count": 5374,
  "candidates_token_count": 355,
  "total_token_count": 6637
}
```
