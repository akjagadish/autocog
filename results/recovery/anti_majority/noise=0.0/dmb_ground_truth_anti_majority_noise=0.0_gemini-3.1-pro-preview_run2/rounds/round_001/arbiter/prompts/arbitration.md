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
People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 9: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]

**Rationale:** To dissociate Take The Best (TTB) from the Weighted Additive (WADD) model, we use a 4-feature design with a steep drop-off in validities after the first cue, but closely spaced validities for the remaining cues (e.g., 0.95, 0.75, 0.65, 0.55). TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, on the other hand, integrates all cues. If WADD's gamma parameter is low, it will behave compensatorily and favor options with multiple lower-validity cues over an option with a single high-validity cue. We construct trials that pit the highest-validity discriminating cue against a coalition of multiple lower-validity cues. For instance, an option favored by only the top cue will be chosen by TTB, whereas WADD (with low-to-moderate gamma) will choose the alternative favored by the three remaining cues. We also include trials where the highest cue is tied, shifting the TTB decision to the second cue, which is then pitted against the third and fourth cues to provide a robust test across different levels of the cue hierarchy.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
This metric isolates the specific trials where the Take The Best (TTB) heuristic and the Weighted Additive (WADD) model make opposite predictions. By computing the proportion of trials on which a subject's choices align with TTB on these divergent trials, we obtain a direct measure of the decision strategy. TTB will score near 1.0, while WADD (especially for low-to-moderate gamma values) will score substantially lower, providing a stark contrast between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    diff = a_mat - b_mat
    
    # TTB prediction: heavily weights the first discriminating cue
    # Using powers of 10 ensures strict lexicographical ordering (TTB logic)
    weights = np.array([1000, 100, 10, 1])
    ttb_score = np.dot(diff, weights)
    ttb_pred = np.where(ttb_score > 0, 0, 1)
    
    # WADD prediction (with gamma=1, i.e., linear integration)
    wadd_score = np.dot(diff, val)
    wadd_pred = np.where(wadd_score > 0, 0, 1)
    
    # Identify trials where TTB and baseline WADD disagree
    divergent = ttb_pred != wadd_pred
    
    if not np.any(divergent):
        return 0.5
        
    responses = data['response'].values
    # Calculate proportion of choices matching TTB on these critical trials
    ttb_match = (responses[divergent] == ttb_pred[divergent]).mean()
    
    return float(ttb_match)

### RESULTS
- Predicted under pi_1 (simulated): 0.8418 (var=0.0111)
- Predicted under pi_3 (simulated): 0.7036 (var=0.0145)
- Observed on real data: 0.7236 (var=0.0302)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** To robustly dissociate the Weighted Additive (WADD) theory from Take The Best (TTB), we use a 5-feature design with one highly valid cue and several moderately valid cues. TTB relies strictly on the highest-validity discriminating cue, ignoring all others. WADD, however, integrates all cues, allowing a coalition of lower-validity cues to compensate for the lack of the highest-validity cue. We construct a series of adversarial trials where the top discriminating cue points to Option A, but the majority of remaining cues point to Option B. Across these trials, TTB consistently chooses Option A, while WADD (assuming a moderate gamma parameter) will consistently choose Option B due to the overwhelming additive evidence of the lesser cues. By varying which cues tie and which discriminate, we test the compensatory nature of WADD against the non-compensatory lexicographic stopping rule of TTB at different levels of the cue hierarchy.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, all trials in this experiment feature a conflict where the single most valid discriminating cue points to Option A (response = 0), while the majority of lower-validity cues point to Option B (response = 1). Take The Best (TTB) will consistently choose Option A across all trials, making the mean response close to 0 (only deviating due to lapse noise). Weighted Additive (WADD) integrates all cues, and for a wide range of its gamma parameter, the combined weight of the numerous lower-validity cues will outweigh the single best cue, leading to a much higher proportion of Option B choices. Thus, the overall mean of the 'response' column (proportion of Option B choices) will be substantially higher under WADD than under TTB.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1787 (var=0.0150)
- Predicted under pi_3 (simulated): 0.3127 (var=0.0167)
- Observed on real data: 0.3975 (var=0.0240)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    disagree_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction
        ttb_winner = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1
            
        # Only consider trials where TTB and Tallying make opposite predictions
        if ttb_winner != -1 and tally_winner != -1 and ttb_winner != tally_winner:
            disagree_count += 1
            if row['response'] == ttb_winner:
                ttb_match_count += 1
                
    if disagree_count == 0:
        return 0.5
    return ttb_match_count / disagree_count

```

**Observed (real) value:** 0.7581 (var=0.0332)
**Predicted under pi_1:** 0.8391 (var=0.0094)
**Predicted under pi_3:** 0.6753 (var=0.0250)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

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
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pref)
        
    if len(matches) == 0:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2506 (var=0.0294)
**Predicted under pi_1:** 0.1211 (var=0.0091)
**Predicted under pi_3:** 0.2033 (var=0.0222)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, the observed data consistently falls between the predictions of the pure Take The Best (TTB, pi_1) and Weighted Additive (WADD, pi_3) theories. In Experiment 1, 2, and 4, WADD's predictions are closer to the observed metrics, while in Experiment 3, TTB is closer. The observed metrics indicate that people are neither perfectly non-compensatory (as TTB claims) nor uniformly compensatory (as WADD claims with its current parameterization). The population-level data strongly suggests a mixture of decision strategies, where some decisions or individuals rely on a frugal TTB-like heuristic, and others rely on a more compensatory integration of cues.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture' theory to replace the pure Take The Best model. The new theory should posit that decision-makers do not uniformly employ a single strategy. Instead, there is variability (either within or between subjects) in strategy selection. The model should implement a probabilistic mixture of a non-compensatory heuristic (like TTB) and a compensatory strategy (like Tallying or WADD). By including a mixing parameter that dictates the probability of using TTB versus the compensatory strategy, the model will be able to naturally capture the intermediate metric values observed across all experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5740,
  "candidates_token_count": 304,
  "total_token_count": 6975
}
```
