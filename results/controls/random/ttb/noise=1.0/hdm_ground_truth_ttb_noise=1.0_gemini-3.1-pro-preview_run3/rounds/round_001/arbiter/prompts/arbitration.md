# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_2_1
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
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

    # Weighted sum per option (dot product with per-feature subjective weights).
    scores = stim @ (validities * w)

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
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[2, 7, 7, 7]  B=[10, 6, 6, 6]
  trial 2: A=[10, 2, 2, 2]  B=[2, 3, 3, 3]
  trial 3: A=[0, 0, 10, 10]  B=[10, 10, 0, 0]
  trial 4: A=[10, 10, 0, 0]  B=[0, 0, 10, 10]
  trial 5: A=[0, 5, 5, 5]  B=[10, 0, 0, 1]
  trial 6: A=[10, 0, 0, 1]  B=[0, 5, 5, 5]

**Rationale:** To cleanly dissociate Tallying from WADD, this design exploits Tallying's insensitivity to both feature validities and cardinal feature magnitudes. By setting rating_max to 10 and providing four features with decreasing validities, we construct several distinct trial types. In some trials, one option wins on three out of four features by a small margin (strongly preferred by Tallying), but the other option wins on the most valid feature by a massive margin (strongly preferred by WADD). We also include trials where the models make orthogonal predictions: trials where Tallying sees a tie (2 wins each) but WADD strongly prefers one option due to high validities/magnitudes, and trials where WADD sees a tie (equal weighted sums) but Tallying strongly prefers one option due to it winning on more features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic, specifically isolating trials where Tallying makes a strict prediction (i.e., one option wins on more features than the other). On trials 1 and 2 of this design, Tallying and WADD make diametrically opposed predictions because Tallying favors the option with numerous small wins, whereas WADD favors the option with a single massive win on the most valid feature. On trials 5 and 6, Tallying again makes a strict prediction while WADD evaluates them as roughly equal (depending on subjective weights). Therefore, data generated by the Tallying model will yield values close to 1.0 (minus lapse rate), while data generated by WADD will yield much lower values (around 0.25 on average), strongly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Count how many features option A strictly wins and option B strictly wins
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Only consider trials where one option wins on strictly more features than the other
    mask = a_wins != b_wins
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with more feature wins
    # response == 0 means A was chosen, response == 1 means B was chosen
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    actual = data.loc[mask, 'response']
    
    # Return the proportion of trials where the subject's choice aligns with the Tallying prediction
    return float((tally_pred == actual).mean())

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2_1 (simulated): 0.5081 (var=0.0067)
- Observed on real data: 0.5088 (var=0.0034)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 6, 6, 6]  B=[10, 10, 5, 5, 5]
  trial 2: A=[10, 4, 4, 4, 4]  B=[0, 5, 5, 5, 5]
  trial 3: A=[10, 10, 0, 0, 5]  B=[0, 0, 10, 10, 5]
  trial 4: A=[0, 7, 7, 7, 7]  B=[10, 6, 6, 0, 0]
  trial 5: A=[10, 0, 0, 0, 0]  B=[0, 4, 4, 5, 0]
  trial 6: A=[8, 8, 2, 2, 2]  B=[2, 2, 8, 8, 8]

**Rationale:** To cleanly dissociate the Weighted Additive (WADD) strategy from the Tallying heuristic, we leverage two fundamental differences: Tallying's insensitivity to both feature validities and cardinal feature magnitudes. By setting rating_max to 10 and using 5 features with varying validities, we can construct trials where Tallying's count of strict feature-wise wins points in the opposite direction of WADD's validity-weighted cardinal sum. For example, Option A can win on a majority of features by a small margin (triggering a Tallying preference for A), while Option B wins on the highest-validity features by a massive margin (triggering a WADD preference for B). We also include trials where Tallying sees a tie but WADD strongly prefers one option due to validities and magnitudes, as well as trials where WADD is nearly indifferent but Tallying has a strong preference due to the sheer number of feature wins.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1, Tallying prefers Option A because it wins on more features (3 vs 2), whereas WADD prefers Option B because B has a massive advantage on the most important features. In Trial 2, Tallying prefers Option B (it wins on 4 out of 5 features), whereas WADD strongly prefers Option A due to a massive advantage on the highest-validity feature. The metric computes the difference in the probability of choosing Option A between Trial 1 and Trial 2. This difference is strongly positive for Tallying and strongly negative for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = data['option_a_ratings'].tolist()
    # Trial 1: A=[1, 1, 6, 6, 6]
    is_t1 = np.array([a[0] == 1 for a in a_ratings])
    # Trial 2: A=[10, 4, 4, 4, 4]
    is_t2 = np.array([a[0] == 10 and a[1] == 4 for a in a_ratings])
    
    resp_A = (data['response'] == 0).values
    
    p_A_t1 = resp_A[is_t1].mean() if is_t1.sum() > 0 else 0.5
    p_A_t2 = resp_A[is_t2].mean() if is_t2.sum() > 0 else 0.5
    
    return float(p_A_t1 - p_A_t2)

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2_1 (simulated): -0.2162 (var=0.0574)
- Observed on real data: -0.0100 (var=0.0271)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[5, 6, 6, 6]  B=[10, 4, 4, 4]
  A=[8, 2, 8, 8]  B=[9, 10, 2, 2]
  A=[6, 6, 2, 2]  B=[4, 4, 10, 10]
  A=[2, 8, 8, 8]  B=[10, 6, 6, 6]
  A=[4, 4, 9, 9]  B=[8, 8, 2, 2]
  A=[0, 4, 4, 4]  B=[10, 0, 0, 0]
  A=[7, 7, 3, 3]  B=[6, 6, 10, 10]
  A=[3, 8, 8, 8]  B=[9, 5, 5, 5]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def count_a_wins(row):
        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b)
        
    a_wins = data.apply(count_a_wins, axis=1)
    mask = a_wins >= 3
    
    if mask.sum() == 0:
        return 0.5
        
    return float(data.loc[mask, 'response'].mean())
```

**Observed (real) value:** 0.5042 (var=0.0050)
**Predicted under pi_1:** 0.0000 (var=0.0000)
**Predicted under pi_2_1:** 0.5188 (var=0.0101)

### Experiment 4
**Design**
  A=[1, 6, 6, 6]  B=[10, 5, 5, 5]
  A=[10, 2, 2, 2]  B=[1, 3, 3, 3]
  A=[10, 10, 2, 2]  B=[2, 2, 10, 10]
  A=[9, 9, 8, 1]  B=[8, 8, 7, 10]
  A=[8, 8, 7, 10]  B=[9, 9, 8, 1]
  A=[5, 5, 5, 5]  B=[6, 6, 6, 6]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    a1 = data['option_a_ratings'].apply(lambda x: x[1])
    
    mask_t1 = (a0 == 1)
    mask_t2 = (a0 == 10) & (a1 == 2)
    mask_t4 = (a0 == 9)
    mask_t5 = (a0 == 8)
    
    match_t1 = mask_t1 & (data['response'] == 1)
    match_t2 = mask_t2 & (data['response'] == 0)
    match_t4 = mask_t4 & (data['response'] == 1)
    match_t5 = mask_t5 & (data['response'] == 0)
    
    matches = match_t1 | match_t2 | match_t4 | match_t5
    valid_trials = mask_t1 | mask_t2 | mask_t4 | mask_t5
    
    if valid_trials.sum() == 0:
        return 0.5
        
    return float(matches.sum() / valid_trials.sum())
```

**Observed (real) value:** 0.5156 (var=0.0035)
**Predicted under pi_1:** 0.0000 (var=0.0000)
**Predicted under pi_2_1:** 0.5500 (var=0.0079)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Tallying) predicts extreme metric values (e.g., 1.0 or 0.0) that dramatically fail to align with the observed data, which consistently hovers around 0.5. Theory 2 (WADD) performs much better, predicting values near 0.5 in most cases, though it struggles slightly in Experiment 2. The consistent ~0.5 observed values in metrics designed to isolate Tallying behavior suggest that subjects are decidedly not using a simple unweighted count of feature wins. Instead of Tallying, a strong alternative non-compensatory theory in this domain is 'Take The Best' (TTB), which relies solely on the single most valid discriminating feature and ignores the rest.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic. Under TTB, subjects evaluate features sequentially in descending order of their validities. They stop at the first feature that discriminates between the two options (i.e., where one option has a strictly higher value than the other) and choose the option that wins on that feature. If no features discriminate, they guess. This introduces a purely non-compensatory, lexicographic strategy that strongly contrasts with WADD's compensatory integration and Tallying's unweighted voting."
}
```

## Usage

```json
{
  "prompt_token_count": 5307,
  "candidates_token_count": 306,
  "total_token_count": 6735
}
```
