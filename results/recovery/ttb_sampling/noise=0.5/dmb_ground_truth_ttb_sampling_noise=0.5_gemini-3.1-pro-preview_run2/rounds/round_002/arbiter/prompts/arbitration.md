# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People use the 'Take The Best' (TTB) heuristic, a non-compensatory lexicographic strategy. They evaluate features sequentially in descending order of their validities and stop at the first feature that discriminates between the options. The option with the higher value on this decisive feature is chosen. If no feature discriminates, they guess. Behavior incorporates response noise and lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    score_a = 0.0
    score_b = 0.0
    
    # Find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


## THEORY 2 — pi_4
Bounded Sequential Feature Sampling: Subjects evaluate features in decreasing order of validity, accumulating unweighted evidence (the net difference in feature values) until a low internal threshold is reached. If the threshold is met, search terminates and a choice is made; if all features are exhausted, the choice is based on whatever net evidence was accumulated. This generalizes 'Take The Best' by allowing for a 'Take The Two' or similar heuristic where more than one discriminating feature may be required to reach a decision.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    threshold = int(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    
    E = 0.0
    # Accumulate evidence until threshold is met
    for idx in order:
        E += (a[idx] - b[idx])
        if abs(E) >= threshold:
            break
            
    # Scores based on accumulated evidence
    scores = np.array([E, -E])
    
    # Softmax for response noise
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    p = np.asarray(probabilities, dtype=float)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from Bounded Sequential Feature Sampling. TTB evaluates features in descending order of validity, stops at the first discriminating feature, and assigns a constant internal score difference (1 vs 0) regardless of the remaining features, with behavior softened by a wide range of response noise (beta in [0.01, 5.0]). The competing Bounded Sequential Feature Sampling accumulates unweighted evidence across features until a threshold (1, 2, or 3) is met, and operates with much lower response noise (beta in [5.0, 20.0]). By pitting the first discriminating feature against varying amounts of opposing evidence from subsequent features, we create trials where the competing model's accumulated evidence E systematically varies (e.g., E = -2, -1, 0, 1, 2). Consequently, the competing model is forced to predict either extreme deterministic choices (when |E| >= 1) or perfect ties (when E = 0). In contrast, TTB consistently favors the option with the best discriminating feature and can accurately capture moderate choice probabilities (e.g., 75%) across all these trial types due to its unconstrained noise parameter.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing Option A between Trial 2 (A=[1,0,0,0,0], B=[0,0,0,0,0]) and Trial 5 (A=[1,0,0,0,0], B=[0,1,1,1,0]). Under Take The Best (TTB), the first discriminating feature in both trials is the first feature, which favors A. Thus, TTB predicts an identical core preference for A in both trials, meaning the difference in choice rates will be near 0 on average, regardless of the subject's noise level. Under Bounded Sequential Feature Sampling (BSFS), subjects with a threshold of 2 or 3 will choose A in Trial 2 but will accumulate enough opposing evidence from the subsequent features to choose B in Trial 5. This results in a large positive difference (~1.0 for those subjects), creating a substantial mean gap between the theories that easily achieves statistical significance.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    data['choose_A'] = (data['response'] == 0).astype(float)
    
    t2_mask = (data['A_str'] == '10000') & (data['B_str'] == '00000')
    t5_mask = (data['A_str'] == '10000') & (data['B_str'] == '01110')
    
    p_a_t2 = data[t2_mask]['choose_A'].mean()
    p_a_t5 = data[t5_mask]['choose_A'].mean()
    
    if pd.isna(p_a_t2): p_a_t2 = 0.5
    if pd.isna(p_a_t5): p_a_t5 = 0.5
    
    return float(p_a_t2 - p_a_t5)

### RESULTS
- Predicted under pi_3 (simulated): 0.0050 (var=0.0415)
- Predicted under pi_4 (simulated): 0.5517 (var=0.2147)
- Observed on real data: -0.0033 (var=0.0403)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate Bounded Sequential Feature Sampling (BSFS) from Take The Best (TTB), we exploit their different stopping rules. TTB always stops at the first discriminating feature, ignoring all subsequent information. BSFS, on the other hand, accumulates unweighted evidence across features until a decision threshold (1, 2, or 3) is reached. By creating trials where the highest-validity feature points to Option A, but multiple subsequent features point to Option B, BSFS (if its threshold is > 1) will continue sampling, eventually overturning the initial evidence and choosing B. TTB will invariably choose A on these same trials. We vary the amount of opposing evidence to distinguish between different possible thresholds in the BSFS model.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In this experimental design, Take The Best (TTB) evaluates the first feature and immediately chooses Option A in almost all trials. Bounded Sequential Feature Sampling (BSFS) with a threshold > 1 will continue sampling and often choose Option B due to the strong opposing evidence in the remaining features. Calculating the mean response (proportion of B choices) will yield a value near 0 for TTB and a significantly higher value for BSFS.

Source:
def metric(data: pd.DataFrame) -> float:
    # The experimental design consists of trials where the first (most valid) feature 
    # always favors Option A, but the subsequent features often favor Option B.
    # TTB will almost exclusively choose Option A (response = 0) because it stops at the first feature.
    # BSFS, depending on its threshold, will often accumulate evidence and choose Option B (response = 1).
    # Therefore, the overall proportion of B choices directly discriminates the two models.
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.3373 (var=0.0227)
- Predicted under pi_4 (simulated): 0.5990 (var=0.1059)
- Observed on real data: 0.3371 (var=0.0041)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_matches.append(1 if row['response'] == 0 else 0)
        elif b_wins > a_wins:
            tally_matches.append(1 if row['response'] == 1 else 0)
    if len(tally_matches) == 0:
        return 0.5
    return float(np.mean(tally_matches))
```

**Observed (real) value:** 0.2967 (var=0.0025)
**Predicted under pi_3:** 0.3267 (var=0.0265)
**Predicted under pi_4:** 0.4307 (var=0.1485)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where Tallying and WADD make opposite predictions
    # A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \
                 data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    is_trial_1_rev = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & \
                     data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
                     
    tally_choices = 0
    total = 0
    
    if is_trial_1.sum() > 0:
        # Tallying favors B (response 1) because it has 3 features vs 2.
        tally_choices += data[is_trial_1]['response'].sum()
        total += is_trial_1.sum()
        
    if is_trial_1_rev.sum() > 0:
        # Tallying favors A (response 0).
        tally_choices += (1 - data[is_trial_1_rev]['response']).sum()
        total += is_trial_1_rev.sum()
        
    if total == 0:
        return 0.5
        
    return float(tally_choices / total)
```

**Observed (real) value:** 0.3600 (var=0.0119)
**Predicted under pi_3:** 0.3200 (var=0.0334)
**Predicted under pi_4:** 0.3484 (var=0.1788)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(5):
            if a[i] > b[i]:
                if r == 0:
                    ttb_matches += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    ttb_matches += 1
                break
        total += 1
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.6716 (var=0.0045)
**Predicted under pi_3:** 0.6977 (var=0.0191)
**Predicted under pi_4:** 0.3592 (var=0.1837)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.65, 0.6, 0.55, 0.5])
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        score_a = np.dot(a, validities)
        score_b = np.dot(b, validities)
        
        if score_a > score_b:
            wadd_choice = 0
        elif score_b > score_a:
            wadd_choice = 1
        else:
            continue
            
        matches.append(1.0 if row['response'] == wadd_choice else 0.0)
        
    return float(np.mean(matches)) if len(matches) > 0 else 0.0
```

**Observed (real) value:** 0.3383 (var=0.0054)
**Predicted under pi_3:** 0.3031 (var=0.0170)
**Predicted under pi_4:** 0.5729 (var=0.1950)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Take The Best, pi_3) consistently and remarkably captures the observed data across all experiments, often matching the observed point estimates almost perfectly (e.g., Exp 1: -0.0033 vs 0.0050; Exp 2: 0.3371 vs 0.3373; Exp 5: 0.6716 vs 0.6977). Theory 2 (Bounded Sequential Feature Sampling, pi_4) systematically mispredicts the outcomes, particularly overestimating the influence of subsequent features when the first feature discriminates, leading to predictions that are far from the observed data in Experiments 1, 2, 5, and 6. Therefore, Theory 1 is strongly supported by the data, and Theory 2 is degenerate.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Theory 2 with a brand-new theory that can serve as a stronger alternative to Take The Best. Since Take The Best (a non-compensatory lexicographic rule) fits the data very well, a competitive alternative could be a Probabilistic Lexicographic Model (where the order of feature evaluation or the stopping rule is stochastic rather than deterministic) or a fully Compensatory Weighted Additive (WADD) model that integrates all features but uses a steep non-linear transformation of validities to mimic lexicographic behavior. This will provide a rigorous test of whether subjects strictly stop at the first feature or just heavily down-weight subsequent features."
}
```

## Usage

```json
{
  "prompt_token_count": 6164,
  "candidates_token_count": 343,
  "total_token_count": 7086
}
```
