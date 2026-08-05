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
Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
    return int(np.random.choice(len(probabilities), p=probabilities))


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
**Validities (n_features=5):** [0.95, 0.85, 0.55, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** This design dissociates the Weighted Additive (WADD) theory from Tallying by manipulating the distribution of cue validities across the options. WADD integrates all features by weighting them according to a subjective transformation of their validities, whereas Tallying ignores validities entirely and simply counts the number of features favoring each option. The trial set includes pairs where one option is supported by a few high-validity cues while the other is supported by a larger number of low-validity cues, leading to opposing predictions. Additionally, several trials present an equal number of winning features for both options (a tie for Tallying), where WADD strongly prefers the option with higher-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the specific trials (T4, T5, and T6) where the Tallying heuristic evaluates the two options as an exact tie (2 winning features each). Under Tallying, subjects must guess on these trials, yielding an expected accuracy of exactly 0.5 with minimal between-subject variance. However, under the WADD theory, these trials present a strict preference because the winning features for one option have systematically higher validities than the winning features for the other. By calculating the proportion of choices that align with WADD's strict preference on these specific trials, we expect exactly 0.5 for Tallying and significantly > 0.5 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify trials where Tallying sees an exact tie (2 wins vs 2 wins)
    t4_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 0))
    t5_mask = (a_tuples == (0, 0, 1, 1, 0)) & (b_tuples == (1, 1, 0, 0, 0))
    t6_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    
    # Under WADD, these trials are NOT ties, because the validities are strictly ordered:
    # w0 > w2 and w1 > w3, etc. 
    # WADD strictly prefers A in T4, B in T5, and A in T6 regardless of the gamma parameter.
    
    c4 = np.sum((data['response'] == 0) & t4_mask)
    c5 = np.sum((data['response'] == 1) & t5_mask)
    c6 = np.sum((data['response'] == 0) & t6_mask)
    
    total = np.sum(t4_mask) + np.sum(t5_mask) + np.sum(t6_mask)
    
    if total == 0:
        return 0.5
        
    return float((c4 + c5 + c6) / total)

### RESULTS
- Predicted under pi_3 (simulated): 0.7222 (var=0.0223)
- Predicted under pi_2 (simulated): 0.4917 (var=0.0061)
- Observed on real data: 0.6989 (var=0.0397)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** This design systematically dissociates Tallying from Weighted Additive (WADD) by contrasting the quantity of winning features against their magnitude (validity). Tallying relies strictly on the count of feature-wise wins, predicting indifference (50/50) when the number of winning features is tied, and favoring the option with more winning features otherwise. WADD, conversely, integrates the subjective validities. We include trials where the option with fewer winning features has much higher validities (WADD prefers the option with fewer wins, Tallying prefers the option with more), as well as 'tie' trials where both options win on an equal number of features but one has higher validities (Tallying predicts exactly 50/50, WADD predicts a strong preference).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts the choice probability of the Tallying-preferred option between Trial 5 and Trial 1. In both trials, one option wins on exactly 3 features and the other on exactly 2. Because Tallying only cares about the unweighted count of wins (a difference of +1 in both cases), it predicts the exact same choice probability across the two trials, yielding a difference of 0. WADD, however, integrates the validities: in Trial 1, the 3-win option is composed of the weakest cues, making WADD prefer the 2-win option; in Trial 5, the 3-win option contains stronger cues, making WADD strongly prefer it. Consequently, WADD predicts a much higher choice rate for the 3-win option in Trial 5 than in Trial 1, yielding a large positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    t1_optA = (0, 0, 1, 1, 1)
    t1_optB = (1, 1, 0, 0, 0)
    
    t5_optA = (0, 1, 1, 1, 0)
    t5_optB = (1, 0, 0, 0, 1)
    
    t1_choices = []
    t5_choices = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_optA and b == t1_optB:
            t1_choices.append(1 if resp == 0 else 0)
        elif a == t1_optB and b == t1_optA:
            t1_choices.append(1 if resp == 1 else 0)
            
        elif a == t5_optA and b == t5_optB:
            t5_choices.append(1 if resp == 0 else 0)
        elif a == t5_optB and b == t5_optA:
            t5_choices.append(1 if resp == 1 else 0)
            
    if not t1_choices or not t5_choices:
        return 0.0
        
    return float(np.mean(t5_choices) - np.mean(t1_choices))

### RESULTS
- Predicted under pi_3 (simulated): 0.2562 (var=0.0685)
- Predicted under pi_2 (simulated): 0.0238 (var=0.0101)
- Observed on real data: 0.1850 (var=0.2183)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    cue_order = np.argsort(-validities)
    
    ttb_matches = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            if row['response'] == ttb_winner:
                ttb_matches += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(ttb_matches / total)

```

**Observed (real) value:** 0.3292 (var=0.0132)
**Predicted under pi_3:** 0.3406 (var=0.0096)
**Predicted under pi_2:** 0.1840 (var=0.0046)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        tally_a = sum(1 for x, y in zip(a, b) if x > y)
        tally_b = sum(1 for x, y in zip(a, b) if y > x)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for x, y in zip(a, b):
            if x > y:
                ttb_pred = 0
                break
            elif y > x:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            agreements.append(1 if resp == tally_pred else 0)
            
    if not agreements:
        return 0.5
    return float(np.mean(agreements))
```

**Observed (real) value:** 0.7356 (var=0.0121)
**Predicted under pi_3:** 0.7931 (var=0.0137)
**Predicted under pi_2:** 0.8589 (var=0.0069)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, THEORY 1 (pi_3, Weighted Additive) consistently outperforms THEORY 2 (pi_2, Tallying). In Experiment 1, subjects show a strong preference on trials that Tallying considers exact ties (observed 0.6989 vs pi_2's 0.4917), aligning well with WADD's prediction (0.7222). In Experiment 2, the choice difference between trials with equal win-counts but different validities is significant (observed 0.1850), refuting Tallying's prediction of 0 but matching WADD's prediction. Experiments 3 and 4 further confirm that subjects do not ignore cue validities, breaking pure Tallying predictions. Because Tallying fundamentally fails to capture the human sensitivity to cue validities demonstrated across these datasets, it is theoretically degenerate in this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory that incorporates both the simplicity of heuristics and the sensitivity to cue validities. For instance, a 'Validity-Thresholded Tallying' theory where subjects only tally features whose validities exceed a subjective threshold, or a 'Sequential Sampling' model where features are sampled with probabilities proportional to their validities until a confidence bound is reached. This would provide a stronger, bounded-rationality competitor to the fully compensatory Weighted Additive model."
}
```

## Usage

```json
{
  "prompt_token_count": 5755,
  "candidates_token_count": 322,
  "total_token_count": 6605
}
```
