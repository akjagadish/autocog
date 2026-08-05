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
Weighted Additive (WADD) Theory with Non-linear Subjective Weights: People evaluate options by considering all available features, weighting each by a subjective transformation of its validity. Rather than using raw validities, people subjectively distort them via an exponent (gamma). A gamma > 1 disproportionately emphasizes higher-validity cues (shifting behavior towards Take The Best), whereas a gamma < 1 flattens the weights (shifting behavior towards Tallying). The subjective value of an option is the weighted sum of its features, and choices are made probabilistically via a softmax over these values, with a lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Weighted additive sum for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design pits WADD against Tallying by manipulating the distribution of feature validities. Tallying ignores validities and simply counts the number of winning features for each option, predicting a tie when the counts are equal and a strict preference when one option has more winning features. WADD, however, weights each feature by a subjective transformation of its validity. We include 'Tallying Tie' trials where Tallying predicts random guessing (50/50), but WADD strongly predicts the option with higher-validity features. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) but the other option wins on fewer, higher-validity features (favored by WADD, depending on the gamma parameter). This creates a quantitative dissociation where WADD predicts a graded probability of choice based on the weighted sum, whereas Tallying predicts either a step-function preference based on counts or a complete tie.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying predicts a strict tie (both options have an equal number of winning features). On these tie trials, Tallying predicts random guessing (50% choice probability). In contrast, WADD uses the validities to break the tie, consistently favoring the option with higher-validity features. The metric calculates the proportion of choices on these tie trials that align with the WADD prediction. Tallying should yield a value around 0.5, while WADD will yield a value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    wadd_match = 0
    n_ties = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            wadd_a = np.sum(a * val)
            wadd_b = np.sum(b * val)
            
            if wadd_a > wadd_b:
                expected = 0
            elif wadd_b > wadd_a:
                expected = 1
            else:
                continue
                
            if row['response'] == expected:
                wadd_match += 1
            n_ties += 1
            
    if n_ties == 0:
        return 0.5
    return wadd_match / n_ties

### RESULTS
- Predicted under pi_3 (simulated): 0.7831 (var=0.0163)
- Predicted under pi_2 (simulated): 0.4850 (var=0.0067)
- Observed on real data: 0.5863 (var=0.0190)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.7, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** This design isolates the effect of cue validities to cleanly dissociate Tallying (which ignores validities and only counts feature wins) from WADD (which weights features by a subjective transformation of their validities). We use four features with a wide spread of validities. The trials include 'Tallying Ties' where each option wins on the same number of features but differs in validity, forcing Tallying to predict random guessing while WADD predicts a strong preference. We also include 'Conflict' trials where one option wins on more features (favored by Tallying) while the other option wins on fewer but higher-validity features (favored by WADD, especially for higher values of gamma). This provides a rigorous quantitative test of whether subjects completely ignore validities as Tallying predicts.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying predicts a tie (equal number of positive features for both options) but WADD predicts a strong preference for the option with the highest validity feature (feature index 0). Specifically, it looks at Trial 1 (1 vs 1 feature) and Trial 6 (2 vs 2 features). For Tallying, the choice probability for the option with feature 1 is exactly 0.5, regardless of beta or epsilon. For WADD, the option with feature 1 always has a higher weighted sum for any gamma > 0, so the choice probability will be significantly > 0.5. This creates a large and stable mean difference between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    target_pairs = [
        ( (1,0,0,0), (0,1,0,0) ),
        ( (1,0,1,0), (0,1,0,1) )
    ]
    
    chosen_has_f1 = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        is_target = False
        for p in target_pairs:
            if (a == p[0] and b == p[1]) or (a == p[1] and b == p[0]):
                is_target = True
                break
                
        if is_target:
            chose_a = (row['response'] == 0)
            chosen_opt = a if chose_a else b
            
            if chosen_opt[0] == 1:
                chosen_has_f1.append(1)
            else:
                chosen_has_f1.append(0)
                
    if len(chosen_has_f1) == 0:
        return 0.5
        
    return float(np.mean(chosen_has_f1))

### RESULTS
- Predicted under pi_3 (simulated): 0.7533 (var=0.0292)
- Predicted under pi_2 (simulated): 0.4892 (var=0.0105)
- Observed on real data: 0.5317 (var=0.0228)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def ttb_predict(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            if b[i] > a[i]:
                return 1
        return -1
    
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if ttb_predict(a, b) == r:
            matches += 1
            
    return float(matches) / len(data) if len(data) > 0 else 0.0
```

**Observed (real) value:** 0.3747 (var=0.0051)
**Predicted under pi_3:** 0.3573 (var=0.0363)
**Predicted under pi_2:** 0.1318 (var=0.0071)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Calculate sum of features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Calculate first feature for each option (most valid cue)
    first_a = data['option_a_ratings'].apply(lambda x: x[0])
    first_b = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Identify trials where Tallying and TTB make strictly opposite predictions
    # Tallying decides by sum(A) vs sum(B)
    # TTB decides by A[0] vs B[0]
    opposing_mask = ((sum_a > sum_b) & (first_a < first_b)) | ((sum_a < sum_b) & (first_a > first_b))
    opposing_data = data[opposing_mask]
    
    if len(opposing_data) == 0:
        return 0.5
        
    # Determine what Tallying would choose: 0 (A) if sum_a > sum_b else 1 (B)
    tallying_choice = (opposing_data['option_a_ratings'].apply(sum) < opposing_data['option_b_ratings'].apply(sum)).astype(int)
    
    # Calculate the proportion of responses that match the Tallying prediction
    match_rate = (opposing_data['response'] == tallying_choice).mean()
    
    return float(match_rate)
```

**Observed (real) value:** 0.6400 (var=0.0095)
**Predicted under pi_3:** 0.7929 (var=0.0253)
**Predicted under pi_2:** 0.8767 (var=0.0063)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither pure Tallying (THEORY 2) nor WADD with non-linear weights (THEORY 1) perfectly captures the data, but the observed metrics consistently fall between their predictions. In Experiments 1 and 2, subjects show a weak preference for higher-validity features on Tallying-tie trials (Observed: ~0.53-0.58), which is higher than Tallying's prediction of random guessing (~0.50) but much lower than WADD's strong preference (~0.75-0.78). In Experiment 3, WADD accurately captures the TTB-match rate, while Tallying severely underpredicts it. In Experiment 4, the observed Tallying match rate on conflict trials (0.64) is lower than both theories' predictions. This pattern strongly suggests a heterogeneous population or a mixture of strategies, where some subjects (or trials) rely on Tallying while others rely on validity-based heuristics like Take-The-Best. Tallying (THEORY 2) is too rigid to accommodate this validity sensitivity.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with a Strategy Mixture Theory. The new theory should propose that decision-makers probabilistically choose between two distinct heuristics: Tallying (counting winning features) and Take-The-Best (relying solely on the highest-validity discriminating feature). The model should include a mixture parameter (e.g., P_tally) that governs the probability of applying Tallying versus TTB on a given trial, alongside response noise. This will allow the model to capture the intermediate valid-cue preferences in Exps 1-2 and the partial Tallying matches in Exp 4, effectively bridging the gap between the two extremes."
}
```

## Usage

```json
{
  "prompt_token_count": 5738,
  "candidates_token_count": 411,
  "total_token_count": 7149
}
```
