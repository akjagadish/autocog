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
Strategy Mixture Theory: The population of decision-makers is heterogeneous, consisting of distinct subgroups that rely on fundamentally different decision rules. Rather than every individual using a noisy version of the same heuristic, about half the population employs a non-compensatory, frugal strategy (Take The Best), while the other half employs a compensatory, unweighted strategy (Tallying). This individual difference accounts for both the ~50% aggregate choice proportions on critical divergence trials and the high (~0.25) between-subject variance, which is characteristic of a Bernoulli distribution of highly deterministic strategies.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a 2xN array.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    strategy = int(parameters["strategy"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if strategy == 0:
        # Take The Best (Non-compensatory)
        cue_order = np.argsort(-val, kind="stable").tolist()
        winner = None
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
                
        if winner is None:
            scores = np.array([0.5, 0.5])
        else:
            scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
            
    else:
        # Tallying (Compensatory)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        scores = np.array([a_wins, b_wins])
        
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_3
Weighted Additive (WADD) Theory: People evaluate options by computing a weighted sum of their features, where the weights correspond to the cue validities above chance (validity - 0.5). This linear compensatory strategy allows multiple lower-validity cues to outweigh a single high-validity cue without over-privileging the top cue. Choices are made probabilistically using a softmax over the weighted sums, combined with a lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Use linear evidence above chance instead of log-odds to prevent over-dominance of the top cue
    weights = val - 0.5
    
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.95, 0.68, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 9: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 11: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses exclusively on Trial 3 (A=[1,0,0,0] vs B=[0,1,0,1]), the only trial where the two sub-strategies of Strategy Mixture (Take-The-Best and Tallying) have strict, opposing preferences. TTB chooses A (relying on the top cue) while Tallying chooses B (relying on the sheer number of cues). This produces a highly variable bimodal distribution of subject means under Strategy Mixture, averaging around 0.5. In contrast, WADD strongly prefers A across the board because the weight of the top cue (0.45) heavily outweighs the sum of the second and fourth cues (0.18 + 0.05 = 0.23), resulting in a point estimate close to 0.9 with very low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask1 = (a_str == '1000') & (b_str == '0101')
    mask2 = (a_str == '0101') & (b_str == '1000')
    
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 0.5
        
    choices = 0
    if mask1.sum() > 0:
        choices += (data.loc[mask1, 'response'] == 0).sum()
    if mask2.sum() > 0:
        choices += (data.loc[mask2, 'response'] == 1).sum()
        
    return float(choices / total)

### RESULTS
- Predicted under pi_4 (simulated): 0.5767 (var=0.2302)
- Predicted under pi_3 (simulated): 0.7600 (var=0.0535)
- Observed on real data: 0.4800 (var=0.2496)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.57, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1]  B=[0, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 1]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[0, 0, 0, 0]  B=[0, 0, 0, 1]
  trial 9: A=[0, 0, 0, 0]  B=[1, 1, 0, 1]
  trial 10: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 11: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 12: A=[1, 1, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[0, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 15: A=[0, 0, 0, 1]  B=[1, 0, 0, 1]
  trial 16: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Trial 14 (A=[1, 1, 0, 0] vs B=[0, 1, 1, 0]) is the only trial where the Tallying strategy (used by half the Mixture population) perceives a tie (both options have 2 positive cues) and therefore guesses at random (50%). The other half of the Mixture population uses Take The Best, which strongly prefers A based on the first cue. Thus, the Strategy Mixture theory predicts an aggregate choice proportion for A of roughly 75% on this trial, with very high between-subject variance (bimodal at 50% and 100%). In contrast, the Weighted Additive (WADD) theory assigns a substantially higher weighted sum to A (0.45 + 0.25 = 0.70) than to B (0.25 + 0.07 = 0.32), predicting a consistent preference for A across all subjects (closer to 90-100%, depending on beta). This metric isolates the choice proportion on Trial 14 to capture this stark divergence.

Source:
def metric(data: pd.DataFrame) -> float:
    a_match = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0))
    b_match = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0))
    target_trials = data[a_match & b_match]
    if len(target_trials) == 0:
        return 0.5
    return float((target_trials['response'] == 0).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.7567 (var=0.0669)
- Predicted under pi_3 (simulated): 0.8333 (var=0.0311)
- Observed on real data: 0.5200 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB winner
        val_order = [0, 2, 1, 3] # Indices in descending order of validity
        ttb_winner = None
        for j in val_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying winner
        a_wins = sum(1 for i in range(4) if a[i] > b[i])
        b_wins = sum(1 for i in range(4) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Check strict disagreement
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            total += 1
            if row['response'] == ttb_winner:
                ttb_match += 1
                
    if total == 0:
        return 0.5
    return ttb_match / total
```

**Observed (real) value:** 0.4667 (var=0.0267)
**Predicted under pi_4:** 0.5211 (var=0.2280)
**Predicted under pi_3:** 0.6144 (var=0.0176)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_tup = data['option_a_ratings'].apply(tuple)
    b_tup = data['option_b_ratings'].apply(tuple)
    mask = (a_tup == (0, 1, 1, 1)) & (b_tup == (1, 0, 1, 0))
    if not mask.any():
        return 0.5
    return float(np.mean(data.loc[mask, 'response'] == 0))
```

**Observed (real) value:** 0.3600 (var=0.2304)
**Predicted under pi_4:** 0.4633 (var=0.2237)
**Predicted under pi_3:** 0.3567 (var=0.0467)

### Experiment 5
**Design**
  A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trial 9, where Option A has ratings [0, 1, 1, 0] and Option B has [1, 0, 0, 0]
    def is_target(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        return a == [0, 1, 1, 0] and b == [1, 0, 0, 0]
    
    mask = data.apply(is_target, axis=1)
    target_trials = data[mask]
    
    if len(target_trials) == 0:
        return 0.5
        
    return float(target_trials['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_4:** 0.5167 (var=0.2247)
**Predicted under pi_3:** 0.3600 (var=0.0471)

### Experiment 6
**Design**
  A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Trial 15 is the only trial where WADD and TTB diverge.
    # A = [0, 1, 1, 0], B = [1, 0, 0, 0]
    # WADD weights are [0.45, 0.27, 0.3, 0.05].
    # WADD score for A = 0.27 + 0.3 = 0.57
    # WADD score for B = 0.45
    # WADD prefers A (response = 0).
    # TTB looks at cue 0 first (validity 0.95), which favors B. 
    # TTB prefers B (response = 1).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0)) & \
           data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data[mask]['response'].mean())
```

**Observed (real) value:** 0.5200 (var=0.2496)
**Predicted under pi_4:** 0.3933 (var=0.2220)
**Predicted under pi_3:** 0.3333 (var=0.0389)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the majority of experiments, the observed data exhibits an aggregate choice proportion near 0.5 but with a between-subject variance near 0.25. This indicates a bimodal distribution where individual subjects are highly deterministic (consistently choosing one option over the other), but the population is evenly split on which option to prefer. THEORY 1 (Strategy Mixture of TTB and Tallying) captures this high variance much better than THEORY 2 (WADD), which consistently predicts low variance and fails to capture the true heterogeneity of the subjects. However, THEORY 1 still struggles with the mean in certain experiments (e.g., Experiment 2 and 6), where even its mixture predicts a strong preference that isn't observed in the data. This suggests that while the population is indeed heterogeneous, the specific mixture of TTB and Tallying might not fully describe the strategies being used. THEORY 2 is fundamentally flawed because a single compensatory strategy with noise cannot produce the observed bimodal distributions.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace THEORY 2 (WADD) with a new theory: 'Heterogeneous Cue Hierarchies'. The core idea is that all subjects use a lexicographic, non-compensatory strategy (like Take-The-Best), but they do not all perfectly internalize the objective cue validities provided in the instructions. Instead, individuals construct their own subjective cue hierarchies. For example, some subjects correctly sort cues by objective validity, while others might sort them by presentation order, or heavily overweight a specific idiosyncratic cue. This theory inherently predicts high between-subject variance (as individuals consistently apply their own distinct hierarchy) and can explain why options that seem objectively dominant (like in Experiment 2) still only receive ~50% choice share, as half the population may be relying on a suboptimal top cue."
}
```

## Usage

```json
{
  "prompt_token_count": 7429,
  "candidates_token_count": 403,
  "total_token_count": 9962
}
```
