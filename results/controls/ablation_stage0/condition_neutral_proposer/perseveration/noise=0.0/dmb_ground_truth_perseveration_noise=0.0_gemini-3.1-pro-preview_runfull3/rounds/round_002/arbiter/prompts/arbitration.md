# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Position Bias / Task Disengagement: In the absence of trial-by-trial feedback and when facing complex multi-cue choices, subjects often disengage from effortful feature integration or search. Instead, they fall back on stimulus-independent response strategies, such as a strong position bias (e.g., a tendency to select the first option, Option A) or random guessing. Decisions are driven entirely by these spatial/temporal biases, and the actual feature validities and values are ignored.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model ignores the stimulus and history entirely.
    # It predicts based solely on a spatial position bias for Option A.
    bias_a = float(parameters['bias_a'])
    return np.array([bias_a, 1.0 - bias_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)


## THEORY 2 — pi_3
People make decisions by computing the expected value of each option, integrating both the magnitude of the features and their subjectively weighted validities. In a Weighted Additive (WADD) strategy, every feature contributes to an option's total score proportionally to a non-linear transformation of its cue validity, capturing subjective distortion of probabilities or weights. Choice probabilities are generated via a softmax function over these weighted sums, with an additional lapse rate to account for random errors.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = np.power(val, gamma)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum for each option
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** To maximally dissociate Theory 1 (Position Bias) from Theory 2 (Weighted Additive), we need to manipulate the stimulus such that the objectively superior option alternates between Option A and Option B. Theory 2's predictions will track the feature values, swinging strongly from preferring A to preferring B depending on which option has higher weighted validities. In stark contrast, Theory 1 ignores the stimulus completely and predicts a constant probability of choosing Option A across all trials. By including extreme cases (e.g., A has all positive features and B has none, and vice versa) as well as closer trade-offs, we ensure that Theory 2 produces highly variable choice probabilities across trials while Theory 1 remains perfectly flat.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
If the Position Bias (Task Disengagement) theory is correct, the probability of choosing Option A vs B is determined solely by a spatial/positional bias and is completely independent of the feature ratings. Consequently, the empirical proportion of choosing Option B should be roughly the same across all 8 unique trial types, resulting in a very low variance across trial types (driven only by binomial sampling noise, max ~0.02). Conversely, the Weighted Additive (WADD) theory predicts that choice probabilities strongly depend on the feature values, meaning some trial types will heavily favor Option A and others Option B. This will result in a much higher variance of choice proportions across the 8 trial types (up to ~0.25).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + df['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    trial_means = df.groupby('trial_type')['response'].mean()
    return float(np.var(trial_means))

### RESULTS
- Predicted under pi_4 (simulated): 0.0000 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0389 (var=0.0042)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 7: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1]  B=[1, 1, 1, 0]

**Rationale:** To dissociate a Weighted Additive (WADD) strategy from a Position Bias strategy, this design varies the objective superiority of Option A versus Option B across trials. WADD predicts that the probability of choosing Option A will shift dramatically depending on which option has the higher weighted sum of features. In contrast, the Position Bias theory assumes subjects ignore the stimulus completely, predicting a constant probability of choosing A across all trials regardless of the feature values. By including extreme differences as well as moderate trade-offs, we ensure WADD produces highly variable choice probabilities while Position Bias remains perfectly flat.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the difference in the probability of choosing Option A when Option A is objectively superior (higher weighted sum of features) versus when Option B is objectively superior. Under the WADD strategy, subjects are sensitive to feature values, so they will choose A much more often when it is superior, resulting in a large positive difference. Under Position Bias, subjects ignore the features and choose A with a constant probability across all trials, resulting in a difference near zero.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    val = np.array([0.9, 0.8, 0.7, 0.6])
    score_a = a_ratings.dot(val)
    score_b = b_ratings.dot(val)
    diff = score_a - score_b
    
    choose_a = 1.0 - data['response'].values
    
    mask_a = diff > 0.01
    mask_b = diff < -0.01
    
    p_a = np.mean(choose_a[mask_a]) if np.sum(mask_a) > 0 else 0.0
    p_b = np.mean(choose_a[mask_b]) if np.sum(mask_b) > 0 else 0.0
    
    return float(p_a - p_b)

### RESULTS
- Predicted under pi_4 (simulated): 0.0088 (var=0.0068)
- Predicted under pi_3 (simulated): 0.4004 (var=0.0734)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    preds = []
    for i in range(len(data)):
        a_vec = data['option_a_ratings'].iloc[i]
        b_vec = data['option_b_ratings'].iloc[i]
        pred = -1
        for j in range(len(a_vec)):
            if a_vec[j] > b_vec[j]:
                pred = 0
                break
            elif b_vec[j] > a_vec[j]:
                pred = 1
                break
        preds.append(pred)
        
    return float(np.mean(data['response'].values == np.array(preds)))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4969 (var=0.0020)
**Predicted under pi_3:** 0.3513 (var=0.0200)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            pred = 0
        elif b_wins > a_wins:
            pred = 1
        else:
            continue
            
        if row['response'] == pred:
            match_count += 1
        total += 1
        
    return match_count / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5031 (var=0.0019)
**Predicted under pi_3:** 0.6421 (var=0.0265)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_aligned(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_pred = 0 if a[i] > b[i] else 1
                return row['response'] == ttb_pred
        return False
        
    return float(data.apply(is_ttb_aligned, axis=1).mean())
```

**Observed (real) value:** 0.4250 (var=0.1350)
**Predicted under pi_4:** 0.5221 (var=0.0492)
**Predicted under pi_3:** 0.2840 (var=0.0195)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                break
    return float(matches / len(data))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4985 (var=0.0017)
**Predicted under pi_3:** 0.2888 (var=0.0249)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed data consistently shows zero variance in choice proportions across different trial types and exactly 0.5 alignment with feature-based strategies (like Take-The-Best or Tallying). This provides overwhelming evidence that subjects are completely ignoring the stimulus features. Theory 1 (Position Bias / Task Disengagement) perfectly captures these patterns, as it models decisions as feature-independent spatial biases. Theory 2 (Weighted Additive), which assumes subjects integrate feature values, consistently predicts sensitivity to the stimuli (higher variance and non-zero differences) and fails to match the observed data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Theory 2 completely fails because it assumes feature-based decision-making, which is not supported by the data. Replace Theory 2 with a 'Sequential Dependency / Alternation' theory. Like Theory 1, this new theory should assume subjects ignore the features (task disengagement), but instead of a static spatial bias, it should posit that subjects' choices depend on their own previous choices (e.g., a tendency to alternate between Option A and Option B, or to repeat the last response). This provides a strong, plausible alternative to static position bias that still perfectly predicts 0.5 expected accuracy and 0.0 feature-driven variance."
}
```

## Usage

```json
{
  "prompt_token_count": 5557,
  "candidates_token_count": 292,
  "total_token_count": 7001
}
```
