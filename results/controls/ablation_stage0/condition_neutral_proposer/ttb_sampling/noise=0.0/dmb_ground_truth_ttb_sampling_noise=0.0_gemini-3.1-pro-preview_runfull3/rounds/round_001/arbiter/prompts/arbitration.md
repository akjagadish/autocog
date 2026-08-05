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
    import numpy as np
    
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

    validities = np.asarray(parameters["validities"], dtype=float)

    # Weighted sum per option (dot product with validities directly).
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** We employ a 5-feature design with a pronounced gap between two high-validity features and three lower-validity features. This structure allows us to construct strong dissociation trials where Tallying favors an option that wins on the three lower-validity features (winning 3 to 2), while the Weighted Additive (WADD) rule favors the option that wins on the two high-validity features because their combined weight exceeds the sum of the bottom three. We also include trials where Tallying predicts a tie (2 wins vs 2 wins) but WADD strongly favors one option, as well as baseline agreement trials to ensure robust parameter estimation for both models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the starkest dissociation trials in the experiment. In trials where one option has the two highest-validity features (e.g., [1,1,0,0,0]) and the other has the three lowest-validity features (e.g., [0,0,1,1,1]), Tallying strongly prefers the latter because it counts 3 positive features versus 2. In contrast, the Weighted Additive (WADD) rule prefers the former because the combined weight of the top two features (0.95 + 0.85 = 1.80) outweighs the bottom three (0.60 + 0.55 + 0.50 = 1.65). By computing the proportion of times the subject chooses the Tallying-favored option on these specific trials, we expect a very high value under the Tallying theory and a very low value under the WADD theory.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_choices = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_str = ''.join(map(str, a))
        b_str = ''.join(map(str, b))
        
        is_t1 = (a_str == '11000' and b_str == '00111')
        is_t2 = (a_str == '00111' and b_str == '11000')
        
        if is_t1:
            tally_choices.append(1 if row['response'] == 1 else 0)
        elif is_t2:
            tally_choices.append(1 if row['response'] == 0 else 0)
            
    if not tally_choices:
        return 0.5
        
    return float(np.mean(tally_choices))

### RESULTS
- Predicted under pi_1 (simulated): 0.8308 (var=0.0159)
- Predicted under pi_2_1 (simulated): 0.0933 (var=0.0101)
- Observed on real data: 0.1200 (var=0.0109)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.6, 0.55, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]

**Rationale:** This experiment uses a 6-feature design to create a robust quantitative dissociation between WADD and Tallying. The validities are structured with two very high values and four much lower values. This allows us to create 'opposition' trials where Option A wins on a small number of high-validity features (favored by WADD), while Option B wins on a larger number of low-validity features (favored by Tallying). Additionally, we include 'tie' trials where Tallying counts an equal number of wins for both options (predicting indifference), but WADD strongly predicts a preference based on the validities. By varying the magnitude of the WADD difference and the Tallying win-differential across trials, we can precisely identify the choice rule and the noise parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric focuses on an 'opposition' trial (Trial 1) where WADD and Tallying make diametrically opposite predictions. The option [1, 1, 0, 0, 0, 0] is favored by WADD because it possesses the two highest-validity features, yielding a higher weighted sum. However, Tallying favors the competitor [0, 0, 1, 1, 1, 0] because it wins on more features (3 vs 2). The metric calculates the proportion of times the subject chooses the WADD-favored option. Under WADD, this value will be close to 1; under Tallying, it will be close to 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target = [1, 1, 0, 0, 0, 0]
    competitor = [0, 0, 1, 1, 1, 0]
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return (a == target and b == competitor) or (a == competitor and b == target)
        
    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.5
        
    def chose_target(row):
        a = list(row['option_a_ratings'])
        chose_a = (row['response'] == 0)
        return 1.0 if (a == target) == chose_a else 0.0
        
    return float(subset.apply(chose_target, axis=1).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1300 (var=0.0112)
- Predicted under pi_2_1 (simulated): 0.9167 (var=0.0125)
- Observed on real data: 0.8567 (var=0.0164)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    score = 0
    count = 0
    for _, row in data.iterrows():
        A = tuple(row['option_a_ratings'])
        B = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: Tallying prefers B (3 to 2 wins), WADD prefers A (1.85 to 1.65)
        if A == (1, 1, 0, 0, 0) and B == (0, 0, 1, 1, 1):
            score += (resp == 1)
            count += 1
        # Trial 8: Tallying prefers A (2 to 1 wins), WADD prefers B (0.95 to 1.15)
        elif A == (0, 0, 1, 1, 0) and B == (1, 0, 0, 0, 0):
            score += (resp == 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.1625 (var=0.0238)
**Predicted under pi_1:** 0.8525 (var=0.0142)
**Predicted under pi_2_1:** 0.4863 (var=0.0035)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    
    trial_1 = (a_tup == (1, 1, 0, 0, 0)) & (b_tup == (0, 0, 1, 1, 1))
    trial_2 = (a_tup == (0, 0, 1, 1, 1)) & (b_tup == (1, 1, 0, 0, 0))
    
    t1_wadd_aligned = (data['response'] == 0) & trial_1
    t2_wadd_aligned = (data['response'] == 1) & trial_2
    
    wadd_choices = t1_wadd_aligned.sum() + t2_wadd_aligned.sum()
    total_dissociation = trial_1.sum() + trial_2.sum()
    
    return float(wadd_choices / total_dissociation) if total_dissociation > 0 else 0.5
```

**Observed (real) value:** 0.8217 (var=0.0115)
**Predicted under pi_1:** 0.1533 (var=0.0142)
**Predicted under pi_2_1:** 0.9658 (var=0.0033)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Tallying (THEORY 1) consistently and spectacularly fails to capture human behavior on dissociation trials. Subjects overwhelmingly choose the option favored by higher-validity features, even when the competitor possesses a greater absolute number of winning features. This is evidenced by the very low proportion of Tallying-aligned choices in Experiment 1 (0.12 observed vs. 0.83 predicted) and Experiment 3 (0.16 observed vs. 0.85 predicted), and the very high proportion of WADD-aligned choices in Experiment 2 (0.85 observed vs. 0.13 predicted by Tallying) and Experiment 4 (0.82 observed). While WADD (THEORY 2) captures the aggregate direction of these choices much better, proposing a completely new, non-compensatory theory like Take-The-Best (TTB) would provide a stronger, more realistic competitor for WADD.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the 'Take-The-Best' (TTB) heuristic. Unlike Tallying (which gives equal weight to all cues) and WADD (which compensatorily sums all weighted cues), TTB is a lexicographic, non-compensatory heuristic. It searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. This will allow us to arbitrate between a compensatory integration process (WADD) and a fast-and-frugal lexicographic process (TTB) on trials where the top cue points one way but multiple secondary cues point the other."
}
```

## Usage

```json
{
  "prompt_token_count": 5932,
  "candidates_token_count": 384,
  "total_token_count": 7411
}
```
