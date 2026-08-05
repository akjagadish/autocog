# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
People use a 'Top-K Tallying' strategy. Instead of relying entirely on a single best cue or integrating all available information, individuals save cognitive effort by tallying feature wins only among the top K most valid features. They ignore the least valid cues entirely. This boundedly rational approach captures the robust, compensatory nature of Tallying while acknowledging cognitive limitations in processing many cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    k = int(round(float(parameters["k"]))) 
    
    # Ensure k is at least 1 and at most n_features
    n_features = len(val)
    k = max(1, min(k, n_features))
    
    # Identify the top K most valid cues
    cue_order = np.argsort(-val, kind="stable")
    top_k_cues = cue_order[:k]
    
    # Tally feature wins only among the top K cues
    a_wins = 0.0
    b_wins = 0.0
    for idx in top_k_cues:
        if a[idx] > b[idx]:
            a_wins += 1.0
        elif b[idx] > a[idx]:
            b_wins += 1.0
            
    scores = np.array([a_wins, b_wins])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= np.sum(probabilities)
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=10):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** To dissociate Top-K Tallying from standard Tallying, we use an environment with 10 features where the advocated theory assumes subjects ignore the least valid cues (since k is parameterized between 4 and 8), while standard Tallying counts all 10 features equally. By creating trials where the top 8 features are either tied in tally or favor Option A, but the bottom 2 features strongly favor Option B, standard Tallying will consistently choose Option B (or predict a tie), whereas Top-K Tallying will either predict a tie or strongly favor Option A, completely ignoring the bottom 2 cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the experimental design, most trials feature Option A having fewer total positive cues than Option B (e.g., 4 vs. 6), but Option A's cues are concentrated among the most valid features. Standard Tallying counts all features equally and will therefore consistently prefer Option B on these trials. In contrast, Top-K Tallying ignores the least valid features, meaning it will either see Option A as superior or see a tie, leading to a significantly higher rate of choosing Option A. By measuring the proportion of Option A choices on these specific trials, we can cleanly separate the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of features for option A and option B
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Select trials where Option A has fewer total positive features than Option B
    # Standard Tallying will consistently choose Option B on these trials.
    # Top-K Tallying will often choose Option A or tie, because Option A's positive features are concentrated in the top K most valid cues.
    mask = a_sums < b_sums
    if not mask.any():
        return 0.5
        
    subset = data[mask]
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float(np.mean(subset['response'] == 0))

### RESULTS
- Predicted under pi_4 (simulated): 0.6597 (var=0.0109)
- Predicted under pi_2 (simulated): 0.1172 (var=0.0062)
- Observed on real data: 0.1283 (var=0.0075)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=10):** [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To dissociate Tallying from Top-K Tallying (where K is bounded between 4 and 8), we use an environment with 10 features of strictly descending validities. Standard Tallying counts all 10 features equally. By contrast, Top-K Tallying completely ignores at least the bottom 2 features (and up to the bottom 6). We construct trials where Option A wins exclusively on the lowest-validity features (e.g., features 8, 9, 10), while Option B wins on a smaller or equal number of the highest-validity features. In these cases, standard Tallying will favor Option A (or predict a tie), whereas Top-K Tallying will ignore Option A's wins entirely and strongly favor Option B (or predict a tie).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In the experimental design, Option A is constructed to have more total feature wins, but its wins are concentrated exclusively in the lowest-validity features (e.g., indices 8 and 9). Option B has fewer feature wins, but they are concentrated in the highest-validity features. Standard Tallying counts all features equally, so it will generally favor Option A (or tie). Top-K Tallying ignores the lowest validity features, so Option A's wins are discarded, and it will strongly favor Option B. Thus, the overall proportion of Option A choices (response == 0) will be significantly higher for Tallying than for Top-K Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    # Return the proportion of times Option A was chosen.
    # Tallying counts all features, meaning Option A wins on most trials (because it has more 1s overall in these specific pairs).
    # Top-K Tallying ignores the lowest validity features where Option A dominates, so Option B wins on most trials.
    return float(np.mean(data['response'] == 0))

### RESULTS
- Predicted under pi_4 (simulated): 0.2042 (var=0.0058)
- Predicted under pi_2 (simulated): 0.7265 (var=0.0054)
- Observed on real data: 0.8588 (var=0.0081)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))
```

**Observed (real) value:** 0.1383 (var=0.0087)
**Predicted under pi_4:** 0.1517 (var=0.0095)
**Predicted under pi_2:** 0.1297 (var=0.0093)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
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
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8286 (var=0.0105)
**Predicted under pi_4:** 0.8433 (var=0.0112)
**Predicted under pi_2:** 0.8479 (var=0.0103)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    df = data.copy()
    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trial 1: A='11000', B='00111' -> WADD prefers A (1.9 vs 1.6), Tally prefers B (2 vs 3)
    # Trial 2: A='00111', B='11000' -> WADD prefers B (1.6 vs 1.9), Tally prefers A (3 vs 2)
    
    t1 = df[(df['A_str'] == '11000') & (df['B_str'] == '00111')]
    t2 = df[(df['A_str'] == '00111') & (df['B_str'] == '11000')]
    
    score = 0.0
    n = 0
    
    if len(t1) > 0:
        score += (t1['response'] == 0).sum()
        n += len(t1)
    if len(t2) > 0:
        score += (t2['response'] == 1).sum()
        n += len(t2)
        
    if n == 0:
        return 0.5
    return float(score / n)

```

**Observed (real) value:** 0.1333 (var=0.0128)
**Predicted under pi_4:** 0.1858 (var=0.0265)
**Predicted under pi_2:** 0.1667 (var=0.0219)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    mask1 = (a_str == '11000') & (b_str == '00111')
    mask2 = (a_str == '00111') & (b_str == '11000')
    
    wadd_chosen = 0
    total = 0
    
    if mask1.sum() > 0:
        wadd_chosen += (data.loc[mask1, 'response'] == 0).sum()
        total += mask1.sum()
        
    if mask2.sum() > 0:
        wadd_chosen += (data.loc[mask2, 'response'] == 1).sum()
        total += mask2.sum()
        
    if total == 0:
        return 0.5
        
    return float(wadd_chosen / total)
```

**Observed (real) value:** 0.1956 (var=0.0240)
**Predicted under pi_4:** 0.1933 (var=0.0360)
**Predicted under pi_2:** 0.1511 (var=0.0177)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 2 (Standard Tallying) consistently captures the observed behavior much better than Theory 1 (Top-K Tallying). In Experiment 1, Top-K Tallying predicts a high rate of choosing Option A, while the observed data matches the low rate predicted by Standard Tallying. In Experiment 2, Top-K Tallying predicts a low rate of choosing Option A, whereas the observed data is very high, again matching Standard Tallying. Experiments 3-6 also show that Standard Tallying is highly competitive. The assumption that people completely ignore the lower validities (Top-K) is strongly contradicted by the data. Theory 1 is therefore degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a brand-new theory that represents a fundamentally different strategy from unweighted Tallying. A classic alternative is the Weighted Additive (WADD) model, where each feature win is weighted by its validity, or Take-The-Best (TTB), where individuals base their decision solely on the single most valid cue that discriminates between the options. Given that Tallying completely ignores validities, proposing a model that fully integrates them (like WADD) or uses them in a lexicographic manner (like TTB) will provide a stronger, theoretically grounded competitor."
}
```

## Usage

```json
{
  "prompt_token_count": 7223,
  "candidates_token_count": 307,
  "total_token_count": 8267
}
```
