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
Decision-makers evaluate options using a Weighted Additive (WADD) strategy. Instead of relying on a single best cue (like Take The Best) or ignoring cue importance (like Tallying), individuals integrate all available features by weighting each feature according to its validity. The overall value of an option is the sum of its validity-weighted features. Choices are then made probabilistically by comparing these weighted sums, with response consistency governed by a softmax temperature parameter and a base lapse rate. A highly restricted temperature parameter prevents over-sensitivity to small differences in the weighted sums, matching the near-chance behavior of human subjects.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate validity-weighted sums for both options
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
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
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)

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
**Validities (n_features=4):** [0.95, 0.6, 0.87, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0]  B=[1, 0, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[1, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  trial 13: A=[1, 0, 1, 1]  B=[1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 15: A=[1, 0, 1, 0]  B=[1, 1, 0, 0]
  trial 16: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying produces a strict tie (equal number of winning features for both options), but WADD has a strict preference due to differing feature validities. On these trials, Tallying is forced to guess, leading to an expected choice probability of exactly 0.5 regardless of its beta or epsilon parameters. WADD, however, will systematically prefer the option with the higher validity-weighted sum. By measuring the proportion of choices that align with the WADD prediction on these specific tie-breaker trials, we obtain a metric that is theoretically pegged to 0.5 for Tallying but strictly >0.5 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.6, 0.87, 0.55])
    
    wadd_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a == tally_b:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            if wadd_a > wadd_b:
                wadd_match.append(1 if row['response'] == 0 else 0)
            elif wadd_b > wadd_a:
                wadd_match.append(1 if row['response'] == 1 else 0)
                
    return float(np.mean(wadd_match)) if len(wadd_match) > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.5425 (var=0.0123)
- Predicted under pi_2 (simulated): 0.5017 (var=0.0103)
- Observed on real data: 0.4900 (var=0.0624)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.57, 0.58, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1]  B=[1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 11: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 12: A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 14: A=[0, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 15: A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates two specific trials (Trial 2 and Trial 16) where Tallying predicts a perfect tie (each option wins on an equal number of features). Under Tallying, subjects must guess on these trials, making the choice probability exactly 0.5 regardless of the beta parameter. Consequently, between-subject variance for Tallying is minimal, driven entirely by binomial sampling noise. Conversely, WADD predicts a consistent preference on these trials due to the unequal validities of the features (preferring B on Trial 2 and A on Trial 16). By calculating the proportion of WADD-preferred choices on these two trials, we expect Tallying to score exactly 0.5 on average, while WADD will score significantly higher than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_t2 = (a_str == '0110') & (b_str == '1001')
    is_t16 = (a_str == '1011') & (b_str == '0111')
    
    t2_wadd_choices = (data.loc[is_t2, 'response'] == 1).sum()
    t16_wadd_choices = (data.loc[is_t16, 'response'] == 0).sum()
    
    total_trials = is_t2.sum() + is_t16.sum()
    
    if total_trials == 0:
        return 0.5
        
    return float((t2_wadd_choices + t16_wadd_choices) / total_trials)

### RESULTS
- Predicted under pi_3 (simulated): 0.5550 (var=0.0238)
- Predicted under pi_2 (simulated): 0.4650 (var=0.0192)
- Observed on real data: 0.5000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[1, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 1, 1]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # The validities are [0.95, 0.72, 0.73, 0.55]
    # Thus, the descending order of cue validities for Take The Best is:
    # Feature 0 (0.95), Feature 2 (0.73), Feature 1 (0.72), Feature 3 (0.55)
    cue_order = [0, 2, 1, 3]
    matches = []
    
    for a_vals, b_vals, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a = np.array(a_vals)
        b = np.array(b_vals)
        
        # Isolate trials where Tallying predicts a tie (equal number of feature wins)
        if np.sum(a > b) == np.sum(b > a):
            # Take The Best will still make a deterministic prediction based on the first discriminating cue
            for j in cue_order:
                if a[j] > b[j]:
                    matches.append(1.0 if resp == 0 else 0.0)
                    break
                elif b[j] > a[j]:
                    matches.append(1.0 if resp == 1 else 0.0)
                    break
                    
    return float(np.mean(matches)) if matches else 0.5
```

**Observed (real) value:** 0.5500 (var=0.0600)
**Predicted under pi_3:** 0.5583 (var=0.0103)
**Predicted under pi_2:** 0.4942 (var=0.0107)

### Experiment 4
**Design**
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 0]
  A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    score = 0.0
    count = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_list = list(a)
        b_list = list(b)
        # Trial 1: Tallying ties, TTB predicts A (0)
        if a_list == [1, 1, 0, 1] and b_list == [0, 1, 1, 1]:
            score += (1 if r == 0 else 0)
            count += 1
        # Trial 2: Tallying ties, TTB predicts B (1)
        elif a_list == [0, 1, 1, 0] and b_list == [1, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
        # Trial 9: Tallying predicts A, TTB predicts B (1)
        elif a_list == [0, 1, 0, 1] and b_list == [0, 0, 1, 0]:
            score += (1 if r == 1 else 0)
            count += 1
            
    if count == 0:
        return 0.5
    return float(score / count)
```

**Observed (real) value:** 0.5333 (var=0.0267)
**Predicted under pi_3:** 0.4856 (var=0.0137)
**Predicted under pi_2:** 0.3489 (var=0.0083)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, we see a mixed pattern. Experiments 1 and 2 isolated trials where Tallying predicts a tie but WADD predicts a strict preference; the observed data yielded 0.49 and 0.50, exactly matching the Tallying prediction of random guessing (0.50) and contradicting WADD's prediction of a reliable preference. However, Experiments 3 and 4 introduced metrics measuring alignment with the 'Take The Best' (TTB) heuristic. In Experiment 3, on Tallying ties, subjects aligned with TTB 55% of the time, which WADD captured well but Tallying missed completely. In Experiment 4, subjects aligned with TTB 53.3% of the time, whereas Tallying predicted 34.9% and WADD predicted 48.6%. Neither Tallying nor WADD consistently explains the full pattern. Tallying is too rigid and fails when subjects break ties using high-validity cues, while WADD predicts too much sensitivity to all validity weights. A new theory is needed that captures lexicographic decision-making.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose 'Take The Best' (TTB) as the new theory. In TTB, decision-makers rank features by their validities and compare options lexicographically. They check the cue with the highest validity first; if one option is better, they choose it. If they tie, they move to the next most valid cue, and so on. If all cues tie, they guess randomly. This explains why subjects appear to guess on certain trials (if the top cues tie and they don't integrate the rest efficiently) while still showing sensitivity to the most predictive cues on other trials, capturing the deviations seen in Exps 3 and 4."
}
```

## Usage

```json
{
  "prompt_token_count": 6104,
  "candidates_token_count": 409,
  "total_token_count": 7600
}
```
