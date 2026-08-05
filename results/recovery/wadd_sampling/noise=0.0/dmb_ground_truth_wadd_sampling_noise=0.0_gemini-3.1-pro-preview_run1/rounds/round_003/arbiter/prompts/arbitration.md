# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Strategy Mixture Theory (Tallying-Biased with Softened Determinism): Decision-makers do not universally rely on a single compensatory mechanism. Instead, the population consists of a mixture of strategies using fast-and-frugal heuristics: 'Take-The-Best' (lexicographic) and 'Tallying' (unweighted sum of strict feature-wise wins). The population shows a stronger preference for Tallying over Take-The-Best, but choices are also somewhat stochastic. By softening the determinism of the individual heuristics, extreme choice probabilities are tempered, allowing the model to fit intermediate conflict trial outcomes more robustly without drastically shifting the population mixture.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Order features by validity (highest first)
    order = np.argsort(val)[::-1]
    
    # Take-The-Best (TTB) evaluation
    ttb_a = 0.0
    ttb_b = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying evaluation (count of strict feature-wise wins)
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    beta = float(parameters["beta"])
    
    # TTB probabilities
    z_ttb = beta * np.array([ttb_a, ttb_b])
    p_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb /= np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta * np.array([tally_a, tally_b])
    p_tally = np.exp(z_tally - np.max(z_tally))
    p_tally /= np.sum(p_tally)
    
    # Mixture
    w_tally = float(parameters["w_tally"])
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Response noise (epsilon-greedy)
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To dissociate the Strategy Mixture Theory (Tallying + Take-The-Best) from the Non-linear Rank-Weighted Additive Strategy, we exploit how the two models handle ties in feature counts and varying cue combinations. By using pairs where Tallying predicts a tie while TTB has a clear preference, the Mixture model is forced to predict in the direction of TTB (since the Tallying component provides a uniform probability). In contrast, the Rank-Weighted model, depending on its gamma parameter, can show the opposite preference. For example, if Option A wins on ranks 4 and 1, and Option B wins on ranks 3 and 2, Tallying ties (2 vs 2) and TTB favors A (rank 4 > rank 3). Thus, the Mixture model strictly favors A. However, the Rank model with gamma < 1 (concave rank weighting) will favor B, because 3^g + 2^g > 4^g + 1^g for g < 1. We also include trials pitting a single high-rank cue against multiple lower-rank cues to capture divergence at higher gamma values.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the difference in the probability of choosing Option A between Trial 6 and Trials 1/2. In all of these trials, Tallying predicts a perfect tie (2 wins vs 2 wins), and Take-The-Best strongly favors Option A on the exact same cue rank (rank 2, index 1). Because the Strategy Mixture Theory only relies on these two heuristics, it predicts identical choice probabilities for Trial 6 and Trials 1/2, making the expected difference 0. In contrast, the Non-linear Rank-Weighted Additive Strategy is sensitive to the specific ranks of the lesser cues. In Trial 6, A's ranks (4, 3) cleanly dominate B's ranks (2, 1), leading to a high probability of choosing A. In Trials 1/2, A's ranks (4, 1) are pitted against B's ranks (3, 2), which are much closer in summed rank-weights (and B even wins when gamma < 1), pulling P(A) down toward 0.5. Thus, the Rank-Weighted model predicts a substantially positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_trial_6(row):
        return tuple(row['option_a_ratings']) == (0, 1, 1, 0, 0) and tuple(row['option_b_ratings']) == (0, 0, 0, 1, 1)
        
    def is_trial_1_or_2(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return (a == (1, 1, 0, 0, 1) and b == (1, 0, 1, 1, 0)) or (a == (0, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0))
        
    mask_6 = data.apply(is_trial_6, axis=1)
    mask_12 = data.apply(is_trial_1_or_2, axis=1)
    
    if mask_6.sum() == 0 or mask_12.sum() == 0:
        return 0.0
        
    p_a_6 = (data[mask_6]['response'] == 0).mean()
    p_a_12 = (data[mask_12]['response'] == 0).mean()
    
    return float(p_a_6 - p_a_12)


### RESULTS
- Predicted under pi_5 (simulated): -0.0294 (var=0.0160)
- Predicted under pi_4 (simulated): 0.2538 (var=0.0866)
- Observed on real data: 0.1250 (var=0.1604)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To dissociate the Strategy Mixture Theory (Tallying + Take-The-Best) from the Non-linear Rank-Weighted Additive Strategy, we exploit how the two models handle feature count ties and the trade-off between feature quantity and rank quality. The Strategy Mixture model resolves feature-count ties strictly in the direction of the highest validity cue (TTB) and otherwise leans heavily toward the option with more winning features (Tallying). In contrast, the Rank-Weighted model interpolates between these extremes based on the non-linear scaling parameter gamma. For example, when Tallying predicts a tie (e.g., A wins ranks 5 and 1; B wins ranks 4 and 3), the Mixture model strictly favors A (via TTB), while the Rank-Weighted model with concave weighting (gamma < 1) favors B. Conversely, when an option has more winning features but lower ranks (e.g., A wins ranks 4, 2, 1; B wins ranks 5, 3), the Mixture model leans toward A (due to its Tallying bias), whereas the Rank-Weighted model with convex weighting (gamma > 1) strongly favors B.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric isolates the mean response on Trial 3, where Option A = [0, 1, 0, 1, 1] and Option B = [1, 0, 1, 0, 0]. The Strategy Mixture Theory generally favors Tallying, which counts 3 features for A and 2 for B, leading to a strong preference for Option A (metric value < 0.5). In contrast, the Rank-Weighted Additive Strategy evaluates the non-linear rank sums. For the vast majority of its parameter space (gamma > ~0.7), the high ranks of Option B's features (ranks 5 and 3) easily outweigh Option A's features (ranks 4, 2, 1), leading to a strong preference for Option B (metric value > 0.5). This single-trial mean provides a massive, stable divergence between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(str(int(v)) for v in x))
    t3_mask = a_str == '01011'
    val = data.loc[t3_mask, 'response'].mean()
    if pd.isna(val):
        return 0.5
    return float(val)

### RESULTS
- Predicted under pi_5 (simulated): 0.3863 (var=0.0250)
- Predicted under pi_4 (simulated): 0.6895 (var=0.0910)
- Observed on real data: 0.4547 (var=0.0760)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return -1

    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.2758 (var=0.0200)
**Predicted under pi_5:** 0.3921 (var=0.0129)
**Predicted under pi_4:** 0.3375 (var=0.0395)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        for j in range(len(a)):
            if a[j] > b[j]:
                return resp == 0
            elif b[j] > a[j]:
                return resp == 1
        return False
        
    return float(data.apply(is_ttb_match, axis=1).mean())
```

**Observed (real) value:** 0.2825 (var=0.0090)
**Predicted under pi_5:** 0.4202 (var=0.0092)
**Predicted under pi_4:** 0.3419 (var=0.0265)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    wadd_aligned = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial type 1: A has the 2 high-validity features, B has the 3 low-validity features
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_aligned.append(1.0 if resp == 0 else 0.0)
        # Trial type 2: Flipped
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_aligned.append(1.0 if resp == 1 else 0.0)
            
    if len(wadd_aligned) == 0:
        return 0.5
    return float(np.mean(wadd_aligned))
```

**Observed (real) value:** 0.5083 (var=0.0801)
**Predicted under pi_5:** 0.3900 (var=0.0251)
**Predicted under pi_4:** 0.6683 (var=0.1086)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total
```

**Observed (real) value:** 0.3446 (var=0.0522)
**Predicted under pi_5:** 0.5954 (var=0.0184)
**Predicted under pi_4:** 0.3185 (var=0.0896)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    target_trials = {
        ('10000', '00110'),
        ('01001', '00110'),
        ('10001', '00111')
    }
    
    combined = list(zip(a_str, b_str))
    mask = [pair in target_trials for pair in combined]
    
    if sum(mask) == 0:
        return 0.0
        
    return float((data.loc[mask, 'response'] == 0).mean())
```

**Observed (real) value:** 0.3593 (var=0.0448)
**Predicted under pi_5:** 0.4688 (var=0.0095)
**Predicted under pi_4:** 0.4846 (var=0.0866)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_ratings = data['option_a_ratings'].apply(tuple)
    mask = (a_ratings == (1, 1, 0, 0, 0)) | (a_ratings == (1, 0, 1, 0, 0))
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
    return float((subset['response'] == 0).mean())
```

**Observed (real) value:** 0.4667 (var=0.0658)
**Predicted under pi_5:** 0.4275 (var=0.0153)
**Predicted under pi_4:** 0.6967 (var=0.0834)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the 8 experiments, neither the Strategy Mixture Theory (pi_5) nor the Non-linear Rank-Weighted Additive Strategy (pi_4) consistently captures the observed data. pi_4 performs better on Experiments 1, 3, 4, and 6, while pi_5 is much closer on Experiments 2, 5, and 8. Experiment 5 is particularly revealing: subjects are exactly evenly split (0.50) when choosing between an option with the top 2 cues and an option with the bottom 3 cues. The sum of the raw validities for the top 2 cues (0.95 + 0.85 = 1.80) is very close to the sum of the bottom 3 cues (0.75 + 0.65 + 0.55 = 1.95). Theories relying on log-odds, strict rank-based weighting (which heavily favors one side depending on gamma), or lexicographic heuristics fail to capture this delicate balance. This suggests subjects might be using the raw validities directly as weights in a linear additive fashion.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Raw Validity-Weighted Additive Strategy'. Instead of relying on fast-and-frugal heuristics (like TTB or Tallying) or non-linear rank transformations, this theory posits that decision-makers compute a simple weighted sum of the features, where the weights are directly proportional to the raw validities provided in the instructions. This naturally explains the ~0.50 split in conflict trials where the sum of a few high-validity cues roughly equals the sum of several lower-validity cues, bridging the gap between the heuristic and rank-based models."
}
```

## Usage

```json
{
  "prompt_token_count": 7480,
  "candidates_token_count": 394,
  "total_token_count": 9413
}
```
