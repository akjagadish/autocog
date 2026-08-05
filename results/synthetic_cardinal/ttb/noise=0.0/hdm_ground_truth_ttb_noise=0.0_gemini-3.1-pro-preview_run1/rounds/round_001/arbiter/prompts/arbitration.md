# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
Equal-Weight (EQW) Heuristic: People evaluate options by summing their raw feature values, treating all features as equally important. Unlike Tallying, EQW uses cardinal feature magnitudes rather than just ordinal comparisons. Unlike Weighted Additive (WADD), it ignores cue validities, weighting every feature equally. The option with the highest total sum is chosen, subject to softmax decision noise and a random lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"EQW expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Sum the raw feature values across all features for each option
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with numerical stability
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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  trial 2: A=[10, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 8, 8, 8]  B=[10, 7, 7, 7]
  trial 4: A=[10, 10, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[5, 5, 5, 5]  B=[4, 4, 4, 8]
  trial 6: A=[9, 9, 0, 0]  B=[10, 0, 1, 1]

**Rationale:** To dissociate Tallying from Equal-Weight (EQW), this design uses cardinal features (rating_max = 10). Tallying computes preferences based strictly on the number of winning features (ignoring the magnitude of the difference), whereas EQW sums the raw feature values across all attributes. We construct trials where one option wins on a single feature by a massive margin (dominating the EQW sum) while the other option wins on multiple features by small margins (dominating the Tallying count). We also include trials that produce a perfect tie for one heuristic but a strong preference for the other.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the deterministic predictions of the Tallying heuristic on trials where Tallying has a strict preference (i.e., ignoring ties). Because the experimental design specifically pits Tallying against Equal-Weight (EQW) by making their predictions exactly opposite on the majority of these trials, data generated by the Tallying model will yield a metric score close to 1.0 (modulo noise), whereas data generated by the EQW model will yield a score close to 0.0.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_3 (simulated): 0.2258 (var=0.0057)
- Observed on real data: 0.7440 (var=0.0257)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  trial 2: A=[8, 8, 8, 0]  B=[7, 7, 7, 10]
  trial 3: A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  trial 4: A=[10, 10, 0, 0]  B=[0, 0, 2, 2]
  trial 5: A=[5, 5, 5, 5]  B=[6, 6, 6, 0]
  trial 6: A=[2, 2, 0, 0]  B=[0, 0, 10, 10]

**Rationale:** To cleanly dissociate Equal-Weight (EQW) from Tallying, we exploit their core difference: EQW sums feature magnitudes, whereas Tallying counts the number of strictly winning features regardless of magnitude. We use cardinal features (rating_max = 10) to create scenarios where one option wins on a single feature by a massive margin (dominating the EQW sum) while the other option wins on multiple features by small margins (dominating the Tallying count). We also include trials that produce a perfect tie for one heuristic but a strong preference for the other, isolating the specific computational mechanisms (summing magnitudes vs. counting signs) driving choice.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the predictions of the Equal-Weight (EQW) heuristic on trials where EQW has a strict preference. By design, the experiment pits the sum of magnitudes (EQW) against the count of winning features (Tallying). Therefore, data simulated under EQW will yield a high proportion of matches, whereas data simulated under Tallying will yield a low proportion, cleanly discriminating the two theories.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.2000 (var=0.0000)
- Predicted under pi_3 (simulated): 0.8640 (var=0.0069)
- Observed on real data: 0.3680 (var=0.0278)

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
**Predicted under pi_3:** 0.1121 (var=0.0066)

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
**Predicted under pi_3:** 0.1421 (var=0.0130)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data consistently aligns much closer with Tallying (THEORY 1) than Equal-Weight (THEORY 2). In Experiment 1, the observed metric is 0.744 (Tallying predicted 1.0, EQW predicted 0.22). In Experiment 2, the EQW alignment metric is only 0.368 (EQW predicted 0.86, Tallying predicted 0.20). Experiments 3 and 4 further corroborate that the majority of choices align with counting winning features rather than summing magnitudes. However, Tallying's predictions are overly deterministic (predicting 1.0 or 0.0), whereas human data shows intermediate proportions (e.g., 0.744). This suggests that while humans primarily tally, they might also be slightly influenced by cue validities or experience higher decision noise. THEORY 2 (EQW) is fundamentally misaligned with the data and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Weighted Tallying' theory. Like Tallying, this heuristic discards cardinal magnitudes and only looks at strict feature-wise wins. However, instead of treating all features equally, it weights each win by the feature's validity. This maintains the robustness to extreme magnitudes seen in the data, while softening the deterministic predictions of pure Tallying by allowing validities to break ties or flip preferences when the tally is close."
}
```

## Usage

```json
{
  "prompt_token_count": 4453,
  "candidates_token_count": 340,
  "total_token_count": 5644
}
```
