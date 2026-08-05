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
People make decisions by integrating all available cues, weighting each cue proportionally to its validity (Weighted Additive rule, or WADD). Unlike Take The Best, which stops at the first discriminating cue, WADD is a compensatory mechanism where multiple lower-validity cues can collectively outweigh a single high-validity cue. This explains why subjects often deviate from the non-compensatory predictions of TTB when a large number of lower-validity cues point in the opposite direction.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate weighted additive scores for both options
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
**Validities (n_features=5):** [0.9, 0.8, 0.6, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates WADD from Tallying and TTB. By employing a specific set of feature validities, we create trials where WADD (which integrates all cues proportionally to their validities) conflicts with Tallying (which merely counts the number of winning features, ignoring validities). For instance, in trials where one option possesses the top two validities while the other has three lower-validity cues, WADD will favor the former while Tallying will favor the latter. Additionally, we include trials where Tallying predicts a tie (equal number of winning features) but WADD decisively prefers one option due to the validity weights. Finally, we include trials that dissociate WADD from Take The Best (TTB) by having multiple moderately valid cues outweigh the single most valid cue, forcing TTB and WADD to make opposite predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the WADD prediction on the subset of trials where WADD and Tallying make strictly opposite predictions. Specifically, in trials where one option has the two most valid cues and the other has the three least valid cues, WADD chooses the former (compensatory integration) while Tallying chooses the latter (counting wins). A value close to 1 indicates WADD-like behavior, while a value close to 0 indicates Tallying-like behavior.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_match = 0
    count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        resp = row['response']
        
        # Trial 1: A has top 2 validities (0.9+0.8=1.7), B has bottom 3 (0.6+0.5+0.5=1.6)
        # WADD prefers A (0), Tallying prefers B (1) since B wins 3-2
        if a == (1, 1, 0, 0, 0):
            if resp == 0:
                wadd_match += 1
            count += 1
        # Trial 2: A has bottom 3, B has top 2
        # WADD prefers B (1), Tallying prefers A (0)
        elif a == (0, 0, 1, 1, 1):
            if resp == 1:
                wadd_match += 1
            count += 1
            
    if count == 0:
        return 0.5
    return float(wadd_match / count)

### RESULTS
- Predicted under pi_3 (simulated): 0.6056 (var=0.0172)
- Predicted under pi_2 (simulated): 0.1475 (var=0.0146)
- Observed on real data: 0.1562 (var=0.0200)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Tallying (which simply counts the number of winning features) from WADD (which integrates all cues weighted by their validities). By using a specific set of feature validities, we construct trials where one option possesses the two most valid cues (thus strongly favored by WADD) while the other option possesses three less valid cues (thus favored by Tallying). We also include trials where both heuristics agree, and trials where Tallying predicts a tie (equal number of winning features) but WADD makes a decisive prediction based on the differing validities of those features.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
Trial 1 (A=[1,1,0,0,0], B=[0,0,1,1,1]) is specifically designed to pit Tallying against WADD. Tallying counts 3 winning features for B versus 2 for A, strongly favoring B. WADD, however, favors A because the two features A wins on have the highest validities (0.95 + 0.85 = 1.8) which outweigh the three features B wins on (0.6 + 0.55 + 0.5 = 1.65). By calculating the proportion of A choices on this trial, we get a clear divergence: Tallying yields a value below 0.5, whereas WADD yields a value above 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying predicts B (3 wins vs 2) but WADD predicts A (1.8 vs 1.65)
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    # Return the proportion of times option A was chosen (response == 0)
    return float((data.loc[is_trial_1, 'response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.7233 (var=0.0238)
- Predicted under pi_2 (simulated): 0.1467 (var=0.0111)
- Observed on real data: 0.1533 (var=0.0127)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        # Only consider trials where Tallying does not predict a tie
        if a_wins != b_wins:
            ttb_pred = None
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_pred = 0
                    break
                elif b[i] > a[i]:
                    ttb_pred = 1
                    break
            
            if ttb_pred is not None:
                if resp == ttb_pred:
                    ttb_matches += 1
                total += 1
                
    return float(ttb_matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.1194 (var=0.0110)
**Predicted under pi_3:** 0.1731 (var=0.0077)
**Predicted under pi_2:** 0.1291 (var=0.0066)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        if row['response'] == tally_pred:
            matches += 1
        total += 1
        
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.7608 (var=0.0062)
**Predicted under pi_3:** 0.8319 (var=0.0117)
**Predicted under pi_2:** 0.8690 (var=0.0080)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 2 (Tallying) dramatically outperforms Theory 1 (WADD). In Experiments 1 and 2, which specifically pitted WADD against Tallying by having a few high-validity cues contradict a larger number of low-validity cues, the observed data overwhelmingly favored the Tallying predictions (around 15% match for the WADD prediction, meaning an 85% match for Tallying). Experiment 3 further confirms that a non-compensatory strategy like Take-The-Best (TTB) is not being used, as the TTB match rate is only ~12% when it disagrees with Tallying. Tallying captures the core behavior very well, while WADD systematically fails because it assumes people perfectly integrate the provided cue validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Theory 1 (WADD) is fundamentally degenerate for this dataset because subjects clearly do not integrate all validities in a compensatory manner. Propose a brand-new theory that acts as a stronger competitor to Tallying. For example, you could propose 'Validity-Biased Tallying' (where subjects primarily count wins/losses but use the validity of the most predictive cue to break ties or softly weight the tally), or a 'Take-The-Majority' rule with an attention mechanism that drops low-validity cues. This new theory should remain largely based on counting, but introduce a bounded or simplified use of the validities to see if it can capture the slight deviations from pure Tallying observed in Experiment 4."
}
```

## Usage

```json
{
  "prompt_token_count": 4885,
  "candidates_token_count": 355,
  "total_token_count": 6300
}
```
