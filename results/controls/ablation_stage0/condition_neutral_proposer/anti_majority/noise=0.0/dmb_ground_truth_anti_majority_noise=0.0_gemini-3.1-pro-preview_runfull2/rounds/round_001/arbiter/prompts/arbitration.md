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
The Weighted Additive (WADD) model with non-linear weight scaling posits that decision makers integrate all available information by computing a weighted sum of features. However, the weighting of cues is not strictly proportional to their log-odds validity. Instead, decision makers apply a non-linear transformation (parameterized by gamma) to the log-odds, allowing them to stretch the weight differential. This permits WADD to approximate lexicographic (TTB-like) choice when gamma > 1, or more uniform (Tallying-like) weighting when gamma < 1, while remaining a fully compensatory integration process.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid infinity
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # Compute weighted sum for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the weighted scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** To distinguish Take The Best (TTB) from Weighted Additive (WADD) with non-linear weight scaling, we use four features with varying validities. The trials are designed to contrast the single-cue reliance of TTB against the compensatory integration of WADD. In several trials, the option favored by the single most valid discriminating cue is opposed by multiple less valid cues favoring the other option. By varying the number and validity of the opposing cues, we can identify whether participants evaluate cues sequentially and stop at the first discriminator (TTB) or integrate them compensatorily (WADD). Furthermore, the varying margins of compensation provide a gradient that can estimate the non-linear scaling parameter gamma in the WADD model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that favor the 'compensatory' option in trials where a single highly valid cue is opposed by multiple lower-validity cues (Trials 1, 2, and 4). Take The Best (TTB) will almost never choose the compensatory option because it strictly follows the first discriminating cue. Conversely, the Weighted Additive (WADD) model, depending on its non-linear scaling parameter gamma, will frequently choose the compensatory option as the combined weight of the lower-validity cues can outweigh the single best cue. This leads to a substantial difference in the expected value of this metric between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    comp_choices = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: A has best cue, B has all others (compensatory is B)
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            comp_choices += (resp == 1)
            total += 1
        # Trial 2: B has best cue, A has next two (compensatory is A)
        elif a == (0, 1, 1, 0) and b == (1, 0, 0, 0):
            comp_choices += (resp == 0)
            total += 1
        # Trial 4: A has best available cue, B has next two (compensatory is B)
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            comp_choices += (resp == 1)
            total += 1
            
    if total == 0:
        return 0.0
    return float(comp_choices / total)

### RESULTS
- Predicted under pi_1 (simulated): 0.1361 (var=0.0088)
- Predicted under pi_3 (simulated): 0.3717 (var=0.0906)
- Observed on real data: 0.2278 (var=0.0190)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.92, 0.82, 0.72, 0.62, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To distinguish the Weighted Additive (WADD) model with non-linear weight scaling from the lexicographic Take The Best (TTB) model, we use a 5-feature design with a steep drop-off in validities. TTB evaluates cues strictly sequentially and stops at the first discriminator, ignoring all other cues. WADD integrates all cues, but its non-linear scaling parameter (gamma) determines whether it behaves more like TTB (gamma > 1) or more like Tallying (gamma < 1). By pitting the single highest-validity cue against varying numbers of lower-validity cues, we create compensatory trials where WADD (with low-to-moderate gamma) will choose the option favored by the sum of lower-validity cues, while TTB will stubbornly choose the option favored by the single best cue. Varying the threshold of compensation across trials provides a fine-grained measure to estimate WADD's gamma parameter and clearly separate it from TTB's strict one-reason decision making.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 1 and Trial 3, Option A is favored by the single most valid discriminating cue, while Option B is favored by all remaining lower-validity cues. The Take The Best (TTB) model is non-compensatory and strictly follows the single best cue, so it will almost exclusively choose Option A (resulting in a response near 0). The Weighted Additive (WADD) model, however, integrates all cues; depending on its gamma parameter, the sum of the numerous lower-validity cues can outweigh the single best cue, leading to a much higher probability of choosing Option B. This metric isolates these highly compensatory trials and calculates the proportion of B choices, cleanly separating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where Option A has only 1 positive feature (the most valid one) 
    # and Option B has 3 or 4 positive features (all lower validity).
    # This corresponds exactly to Trial 1 and Trial 3 in the design.
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    mask = (a_sum == 1) & (b_sum >= 3)
    if not mask.any():
        return 0.0
    # Return the proportion of times Option B was chosen (response == 1)
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1408 (var=0.0158)
- Predicted under pi_3 (simulated): 0.3633 (var=0.0726)
- Observed on real data: 0.2333 (var=0.0272)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        
        if winner is not None:
            if row['response'] == winner:
                ttb_consistent += 1
            total += 1
            
    return float(ttb_consistent / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.6583 (var=0.0166)
**Predicted under pi_1:** 0.8496 (var=0.0117)
**Predicted under pi_3:** 0.7360 (var=0.0309)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, 1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    matches = (data['response'].values[mask] == tally_pred[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.2750 (var=0.0193)
**Predicted under pi_1:** 0.3260 (var=0.0039)
**Predicted under pi_3:** 0.3937 (var=0.0151)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the observed data consistently falls between the predictions of the strict non-compensatory Take The Best (TTB) model and the compensatory Weighted Additive (WADD) model. For instance, in Experiments 1 and 2, observed compensatory choices (~0.23) are higher than TTB predicts (~0.14) but lower than WADD predicts (~0.37). In Experiment 3, TTB-consistency is lower than TTB predicts but lower than WADD predicts as well. This consistent intermediacy suggests that neither a pure lexicographic rule nor a pure compensatory rule captures the population behavior.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a Strategy Mixture Theory. Instead of assuming a single monolithic decision rule for all participants, posit that the population consists of a mixture of strategies (or individuals probabilistically select between them). Specifically, a proportion 'alpha' of decisions are made using a non-compensatory heuristic (like TTB), while the remaining '1 - alpha' are made using a compensatory strategy (like WADD or Tallying). This mixture approach will naturally capture the intermediate rates of compensatory choices and TTB-consistency observed across the experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5629,
  "candidates_token_count": 275,
  "total_token_count": 7097
}
```
