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
People make decisions using a fully compensatory, rational approach known as the Weighted Additive (WADD) rule. Unlike Take The Best, which stops at the first discriminating cue, or Tallying, which ignores cue validities by weighting all features equally, WADD integrates all available information by computing a weighted sum of the features for each option, where the weights are exactly the cue validities. The option with the highest expected value (weighted sum) is favored, with response noise introduced via a softmax over the expected values and a uniform lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    # Convert stimulus to a float array of shape (2, n_features)
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    # Retrieve validities (weights)
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities does not match number of features.")
        
    a, b = stim[0], stim[1]
    
    # Compute the weighted additive score for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To maximally distinguish Weighted Additive (WADD) from Tallying, we use a 5-feature design with a steep drop-off in validities between the top two and the bottom three features. We include 'conflict' trials where WADD prefers an option that wins on a few high-validity features, while Tallying prefers the alternative because it wins on a strictly greater number of lower-validity features. We also include 'tie' trials where both options win on an equal number of features (making Tallying guess uniformly), but the validities heavily favor one option under WADD. Finally, we include concordant trials with varying margins to help constrain the response noise and softmax temperature parameters for both models.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses the option with fewer, but higher-validity features (e.g., [1, 1, 0, 0, 0]) over the option with more, but lower-validity features (e.g., [0, 0, 1, 1, 1]). WADD strongly favors the former because the weighted sum is greater, whereas Tallying strongly favors the latter because it simply counts the number of positive features. This isolates the fundamental difference between compensatory validities and equal-weight counting.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    conflict_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Check for Trial 1 (conflict trial)
        # A has fewer but higher-validity cues, B has more but lower-validity cues.
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            conflict_choices.append(1 if resp == 0 else 0)
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            conflict_choices.append(1 if resp == 1 else 0)
            
    if not conflict_choices:
        return 0.5
    return float(np.mean(conflict_choices))

### RESULTS
- Predicted under pi_3 (simulated): 0.7415 (var=0.0293)
- Predicted under pi_2 (simulated): 0.1585 (var=0.0221)
- Observed on real data: 0.1908 (var=0.0308)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.6, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Rationale:** To maximally distinguish Tallying from Weighted Additive (WADD) decision making, we use a 6-feature design with a steep drop-off in validities. Tallying completely ignores validities and simply counts the number of winning features, while WADD integrates both the number of winning features and their weights. We include 'conflict' trials where Tallying prefers one option because it wins on more features, but WADD prefers the other because it wins on fewer, much higher-validity features. We also include 'WADD tie' trials where the weighted sum exactly equals the sum of multiple lower-validity features, causing WADD to guess while Tallying strongly prefers the option with more winning features. Finally, 'Tallying tie' trials are included where the number of winning features is equal, causing Tallying to guess, but WADD strongly prefers the option with higher-validity features.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices aligning with Tallying on 'conflict' trials. Specifically, Trial 1 pits an option with more winning features (preferred by Tallying) against an option with fewer, but higher-validity features (preferred by WADD). Tallying will yield a high value on this metric, whereas WADD will yield a low value, offering maximal discrimination between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project lists of binary ratings to strings for safe and easy comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))
    
    # Identify the conflict trial where Tallying and WADD strictly disagree
    # Tallying prefers [0, 0, 0, 1, 1, 1] (3 wins vs 2 wins)
    # WADD prefers [1, 1, 0, 0, 0, 0] (score 1.9 vs 1.6)
    mask1 = (a_str == "000111") & (b_str == "110000")
    mask2 = (a_str == "110000") & (b_str == "000111")
    
    choices_tallying = 0
    total = 0
    
    if mask1.any():
        choices_tallying += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.any():
        choices_tallying += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(choices_tallying / total)

### RESULTS
- Predicted under pi_3 (simulated): 0.2100 (var=0.0206)
- Predicted under pi_2 (simulated): 0.8225 (var=0.0222)
- Observed on real data: 0.8700 (var=0.0172)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    winner = np.zeros(len(data))
    for i in range(len(data)):
        w = -1
        for j in range(a_mat.shape[1]):
            if diff[i, j] > 0:
                w = 0
                break
            elif diff[i, j] < 0:
                w = 1
                break
        winner[i] = w
        
    match = (data['response'].values == winner)
    return float(np.mean(match))
```

**Observed (real) value:** 0.2350 (var=0.0036)
**Predicted under pi_3:** 0.2625 (var=0.0084)
**Predicted under pi_2:** 0.2358 (var=0.0067)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate the number of features each option strictly wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    # Filter for trials where Tallying makes a deterministic prediction (no ties)
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    # Tallying predicts the option with more winning features
    tallying_choice = (b_wins > a_wins).astype(int)
    
    responses = data['response'].values
    
    # Calculate the proportion of choices consistent with Tallying
    consistent = (responses[mask] == tallying_choice[mask]).astype(float)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.8275 (var=0.0100)
**Predicted under pi_3:** 0.8519 (var=0.0102)
**Predicted under pi_2:** 0.8450 (var=0.0087)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 2 (Tallying) substantially outperforms Theory 1 (Weighted Additive). In Experiments 1 and 2, which explicitly pit WADD against Tallying by creating conflict trials (fewer high-validity cues vs. more low-validity cues), human subjects overwhelmingly chose the option with more winning features, aligning closely with Tallying (Observed: ~0.19 and ~0.87) and contradicting WADD (Predicted: ~0.74 and ~0.21). WADD completely fails to capture the core behavioral trend in these conflict scenarios. Given that Tallying successfully explains the variance across these tasks, WADD is degenerate for this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the Weighted Additive (WADD) theory with a 'Take The Best' (TTB) or 'Lexicographic' theory. TTB represents the other extreme of the bounded rationality spectrum: it searches through cues in order of validity, stops at the first cue that discriminates between the options, and entirely ignores all remaining cues. Comparing Tallying (which integrates all cues but ignores validities) against TTB (which strictly adheres to validity order but ignores cue quantity) will provide a stringent test of whether subjects employ non-compensatory stopping rules or non-compensatory weighting rules."
}
```

## Usage

```json
{
  "prompt_token_count": 5293,
  "candidates_token_count": 308,
  "total_token_count": 6587
}
```
