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
People evaluate options by computing a weighted sum of all available features, where each feature is weighted by a non-linearly scaled version of its validity centered at chance (0.5). This ensures that uninformative cues contribute nothing, and allows a scaling parameter gamma to control the shift between compensatory and non-compensatory decision-making.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Scale validities non-linearly, centered at chance (0.5)
    centered_val = val - 0.5
    weights = np.sign(centered_val) * (np.abs(centered_val) ** gamma)
    
    # Compute the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
**Validities (n_features=5):** [0.95, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** This design systematically pits the Weighted Additive Model (WADD) against Tallying by contrasting the number of feature wins with the validities of those features. Tallying strictly counts the number of features where one option dominates, completely ignoring the validities. In contrast, WADD scales the features by a non-linear transformation of their validities centered at chance. We constructed a set of 5 features with a steep drop-off in validity. The trial pairs include cases where one option wins on a single highly valid feature while the other wins on multiple less valid features (causing opposing predictions between WADD and Tallying), as well as cases where both options win on an equal number of features (causing Tallying to predict a tie while WADD predicts a strict preference).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying sees a strict tie (an equal number of feature wins for option A and option B). On these trials, Tallying is completely undiscriminating and predicts uniform guessing (50% accuracy). In contrast, WADD uses the validities (centered at chance) to break these ties, leading to a strong directional preference. By computing the proportion of times the subject's choice aligns with WADD's preference on these Tallying-tie trials, we expect a value near 0.5 for data generated by Tallying, and a value substantially above 0.5 for data generated by WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.7, 0.65, 0.6, 0.55])
    w = val - 0.5
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Isolate trials where Tallying sees a tie
        if a_wins == b_wins:
            score_a = np.sum(a * w)
            score_b = np.sum(b * w)
            
            if score_a > score_b:
                target = 0
            elif score_b > score_a:
                target = 1
            else:
                continue
                
            if row['response'] == target:
                match_count += 1
            total += 1
            
    if total == 0:
        return 0.5
        
    return match_count / total

### RESULTS
- Predicted under pi_3 (simulated): 0.6611 (var=0.0165)
- Predicted under pi_2 (simulated): 0.5033 (var=0.0060)
- Observed on real data: 0.6200 (var=0.0045)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.6, 0.55, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design pits Tallying against the Weighted Additive Model (WADD). Tallying simply counts the number of features where an option is superior, entirely ignoring the validities of those features. WADD, on the other hand, scales validities non-linearly and computes a weighted sum. By using a set of validities with a steep drop-off (e.g., 0.95, 0.80, 0.60, 0.55, 0.55), we can construct trials where one option wins on a single highly valid feature while the other wins on multiple low-validity features. Tallying will consistently choose the option with more feature wins, whereas WADD will prefer the option with the higher-validity features. Furthermore, we include trials where both options win on an equal number of features (e.g., 2 vs 2), causing Tallying to predict a tie (guessing), while WADD predicts a deterministic preference based on feature validities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By isolating specific trials where Tallying and the Weighted Additive Model (WADD) make opposite predictions, we can strongly separate the two theories. In Trial 3 and Trial 6, Tallying favors Option B because it wins on more features, whereas WADD strongly prefers Option A due to the high weight of the first feature. In Trial 5, Tallying prefers Option A for having more winning features, but WADD favors Option B. This metric calculates the average alignment with Tallying's predictions across these three highly discriminative trials. Tallying will yield values significantly above 0.5 (approaching 1.0), whereas WADD will yield values significantly below 0.5 (approaching 0.0).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t3 = (a_str == '10001') & (b_str == '01110')
    t5 = (a_str == '01011') & (b_str == '10100')
    t6 = (a_str == '10010') & (b_str == '01101')
    
    score = 0.0
    count = 0
    
    if t3.any():
        score += data.loc[t3, 'response'].mean()
        count += 1
    if t5.any():
        score += (1 - data.loc[t5, 'response']).mean()
        count += 1
    if t6.any():
        score += data.loc[t6, 'response'].mean()
        count += 1
        
    return score / count if count > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.4475 (var=0.0249)
- Predicted under pi_2 (simulated): 0.8442 (var=0.0115)
- Observed on real data: 0.8542 (var=0.0086)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner is not None:
            matches.append(1.0 if resp == ttb_winner else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5711 (var=0.0014)
**Predicted under pi_3:** 0.5142 (var=0.0156)
**Predicted under pi_2:** 0.1771 (var=0.0092)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_preds = np.where(a_wins > b_wins, 0, 1)
    responses = np.array(data['response'].tolist())
    
    matches = (responses[valid_mask] == tally_preds[valid_mask])
    return float(np.mean(matches))

```

**Observed (real) value:** 0.6890 (var=0.0042)
**Predicted under pi_3:** 0.4655 (var=0.0215)
**Predicted under pi_2:** 0.8565 (var=0.0079)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither pure Weighted Additive (WADD) nor pure Tallying perfectly explains the data. In Experiment 2, where Tallying and WADD make opposing predictions, observed behavior strongly aligns with Tallying (0.8542), which pi_2 successfully predicts but pi_3 completely misses. However, Experiment 1 isolates trials where Tallying sees a tie; here, subjects show a systematic preference (0.6200) rather than guessing, which pi_3 captures but pi_2 fails to predict (0.5033). Similarly, Experiment 3 and 4 show intermediate behavior where both validities and tallying play a role. The data strongly suggests a two-stage or hybrid decision process: people predominantly use Tallying, but when Tallying yields a tie, they fall back on feature validities to break it.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying with Validity Tie-Breaker' theory. The theory should state that people primarily compare options by counting feature wins (Tallying). If one option clearly has more wins, it is chosen. However, if the tallies are tied, the decision-maker falls back to using the feature validities (for instance, by picking the option that wins on the single most valid feature, or by computing a weighted sum). This new theory replaces the purely compensatory WADD model and directly accommodates the strong Tallying signal in Exp 2 and 4, while successfully explaining the validity-driven tie-breaking observed in Exp 1."
}
```

## Usage

```json
{
  "prompt_token_count": 5343,
  "candidates_token_count": 358,
  "total_token_count": 6813
}
```
