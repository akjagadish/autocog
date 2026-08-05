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
People evaluate options by computing a weighted sum of their features, where the weights correspond to a non-linear scaling of the log-odds of the validities. This rational Bayesian integration strategy acts as a compensatory mechanism but exhibits strongly non-compensatory behavior when cues are highly diagnostic and the log-odds are exponentiated by a factor gamma. Choice is then made probabilistically based on the difference in these weighted sums.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Clip validities to avoid infinities in log-odds calculation
    val = np.clip(val, 0.001, 0.999)
    
    # Convert validities to log-odds
    log_odds = np.log(val / (1.0 - val))
    
    # Exaggerate weight differences to capture non-compensatory behavior
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # WADD computes the sum of feature values weighted by their scaled log-odds
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.86, 0.84, 0.82, 0.8, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 5-feature design with closely spaced high validities and one lower validity. TTB's strict lexicographic rule dictates that the option with the highest-validity discriminating cue will always be chosen, completely ignoring all lower-validity cues. In contrast, WADD integrates all cues weighted by a non-linear scaling of their log-odds. By setting the validities such that the top validity is only slightly higher than the subsequent ones, we create scenarios where the sum of the weights of the second and third cues securely overpowers the weight of the first cue, even when WADD's scaling parameter (gamma) is high. This directly contrasts TTB's 'one-reason' stopping rule with WADD's compensatory integration.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric calculates the proportion of trials where the subject's choice matches the prediction of the Take The Best (TTB) heuristic. Because the experiment is designed with many 'conflict' trials where TTB and Weighted Additive (WADD) models predict opposite choices, this match rate will be high for data generated by TTB and low for data generated by WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    resp = data['response'].values
    
    diff = a_ratings - b_ratings
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    match = (resp[has_diff] == ttb_pred[has_diff])
    return float(match.mean()) if len(match) > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.8827 (var=0.0091)
- Predicted under pi_3 (simulated): 0.3890 (var=0.0025)
- Observed on real data: 0.8400 (var=0.0088)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.85, 0.82, 0.8, 0.78, 0.75]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from Take The Best (TTB), we use a 5-feature design with a closely spaced set of validities. TTB relies on a strict lexicographic stopping rule, choosing the option favored by the single highest-validity discriminating cue and ignoring all lower-validity cues. In contrast, WADD integrates all cues by weighting them according to a non-linear scaling of their log-odds. By selecting validities that are relatively close to each other, the combined weight of several lower-validity cues in WADD can easily overpower the single highest-validity cue, even when the log-odds are exponentiated by a high gamma parameter. This creates strong compensatory scenarios where WADD predicts one choice (due to the aggregate evidence) while TTB predicts the opposite (due to the single best cue).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In this specific experimental design, Take The Best (TTB) will consistently choose Option A across all 8 trial types because Option A is always favored by the highest-validity discriminating cue. Weighted Additive (WADD), however, compensates for the highest-validity cue by summing the weights of multiple lower-validity cues, leading it to choose Option B on several trial types (e.g., Trial 1, where the four lower cues easily outweigh the top cue even at high gamma). The mean of the 'response' column (proportion of Option B choices) will therefore be near 0 for TTB (driven only by noise/lapses) and significantly higher for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # TTB always chooses Option A (response = 0) across all 8 trials in this design
    # because A always has the single best discriminating cue. 
    # WADD, by contrast, integrates all cues and will frequently choose Option B 
    # (response = 1) because the sum of lower-validity cues often outweighs the single best cue.
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1537 (var=0.0066)
- Predicted under pi_3 (simulated): 0.5608 (var=0.0249)
- Observed on real data: 0.1613 (var=0.0125)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agreements = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
        
        if ttb_winner is not None:
            if resp == ttb_winner:
                agreements += 1
            total += 1
            
    return float(agreements / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.8492 (var=0.0099)
**Predicted under pi_1:** 0.8342 (var=0.0101)
**Predicted under pi_3:** 0.8438 (var=0.0123)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            matches += (resp == 0)
            total += 1
        elif b_wins > a_wins:
            matches += (resp == 1)
            total += 1
    return float(matches / total) if total > 0 else 0.5

```

**Observed (real) value:** 0.1739 (var=0.0108)
**Predicted under pi_1:** 0.1358 (var=0.0099)
**Predicted under pi_3:** 0.1464 (var=0.0088)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Take The Best) consistently provides highly accurate predictions of human choice behavior. In Experiments 1 and 2, which were explicitly designed to pit the non-compensatory TTB strategy against the compensatory Weighted Additive (WADD) strategy of Theory 2, the observed data overwhelmingly favored TTB. Theory 2 predicted that subjects would integrate lower-validity cues to override the top cue, but the low observed rates of Option B choices in Exp 2 and the high TTB match rate in Exp 1 show that subjects almost entirely ignored lower-validity cues when a higher-validity cue discriminated. Thus, the compensatory assumption of Theory 2 is fundamentally misaligned with the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Since strict cue integration (WADD) fails and the non-compensatory TTB rule is highly successful, the new theory should explore a different boundedly rational or heuristic approach. Propose a 'Probabilistic Take The Best' or 'Take The Best with Sequential Sampling' theory, where cue search is guided probabilistically by validity rather than being strictly deterministic, or a hybrid model where subjects use TTB but fall back to a simple Tallying (counting positive features) if the top cues are perceived as equally valid or too noisy. This provides a stronger, more nuanced competitor to the deterministic TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 5345,
  "candidates_token_count": 309,
  "total_token_count": 6327
}
```
