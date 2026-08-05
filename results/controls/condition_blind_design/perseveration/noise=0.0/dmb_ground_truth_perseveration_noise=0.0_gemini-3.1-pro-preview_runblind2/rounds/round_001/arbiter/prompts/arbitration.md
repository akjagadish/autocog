# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.74, 0.81, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 4: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1]  B=[0, 1, 0, 1]
  trial 7: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0]  B=[1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1]  B=[0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 13: A=[0, 0, 0, 0]  B=[1, 0, 1, 0]
  trial 14: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 0]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trial 9, which is uniquely discriminating between Take The Best (TTB) and Weighted Additive (WADD) models in this design. On this trial, Option A has ratings [0, 1, 1, 0] and Option B has [1, 0, 0, 0]. TTB consults the most valid cue first (cue 0, validity 0.95), where B=1 and A=0, so it stops immediately and chooses Option B. In contrast, WADD computes the sum of validities above chance: Option A scores (0.74-0.5) + (0.81-0.5) = 0.24 + 0.31 = 0.55, while Option B scores (0.95-0.5) = 0.45. Since 0.55 > 0.45, WADD strictly prefers Option A. Therefore, the proportion of choosing Option B on this trial will be near 1.0 for TTB and near 0.0 for WADD, cleanly separating the theories.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.8167 (var=0.0336)
- Predicted under pi_3 (simulated): 0.3600 (var=0.0471)
- Observed on real data: 0.5200 (var=0.2496)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.77, 0.8, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 15: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
In this design, Trial 15 is the only choice where the two theories predict opposite preferences. Option A has features [0, 1, 1, 0] and Option B has [1, 0, 0, 0]. TTB relies on the most valid cue (cue 0, validity 0.95), which is present for B but absent for A, leading TTB to strongly predict choosing B. Conversely, WADD integrates all cues. The linear evidence weights (validity - 0.5) are [0.45, 0.27, 0.30, 0.05]. Option A's evidence is 0.27 + 0.30 = 0.57, while Option B's evidence is 0.45. Because 0.57 > 0.45, WADD predicts choosing A. Measuring the average response on this specific trial yields a highly discriminative metric: TTB will produce values near 1 (choosing B), and WADD will produce values near 0 (choosing A).

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.8967 (var=0.0188)
- Predicted under pi_3 (simulated): 0.3333 (var=0.0389)
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
**Predicted under pi_1:** 0.8856 (var=0.0104)
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
**Predicted under pi_1:** 0.1467 (var=0.0352)
**Predicted under pi_3:** 0.3567 (var=0.0467)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, the observed data consistently falls between the predictions of the Take The Best (TTB) theory and the Weighted Additive (WADD) theory, often landing near 0.5. Crucially, in Experiments 1 and 2, the observed metric variance is ~0.25, which is the maximum possible variance for a bounded [0,1] metric. This indicates a bimodal distribution where roughly half the subjects consistently choose Option A and the other half consistently choose Option B. Neither TTB nor WADD alone can account for this high between-subject variance; they predict lower variances indicative of a single noisy strategy. WADD (Theory 2) is somewhat closer to the mean in certain experiments (like Exp 4), but Theory 1 (TTB) completely fails to capture the population heterogeneity. A new theory is needed to explicitly model individual differences in strategy selection.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture Theory' (or Individual Differences in Heuristics) which posits that a population consists of distinct subgroups using different decision rules. Specifically, the model should assign each subject either a non-compensatory strategy (like Take The Best) or a compensatory strategy (like WADD or Tallying) based on a population-level mixture parameter. By mixing these two deterministic bounds, the new theory will naturally predict both the ~0.5 point estimates and the ~0.25 between-subject variance observed in the critical trials."
}
```

## Usage

```json
{
  "prompt_token_count": 6282,
  "candidates_token_count": 334,
  "total_token_count": 7553
}
```
