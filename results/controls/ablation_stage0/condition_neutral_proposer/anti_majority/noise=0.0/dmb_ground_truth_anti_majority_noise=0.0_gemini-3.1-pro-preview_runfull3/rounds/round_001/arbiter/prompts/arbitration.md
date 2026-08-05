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
People use a Weighted Additive (WADD) strategy to make decisions, where each feature is weighted by its subjective importance (a non-linear transformation of its validity). The total score for an option is the sum of the weighted features it possesses. This compensatory mechanism allows multiple lower-validity cues to sometimes outweigh a single high-validity cue, naturally interpolating between Take The Best (when validity differences are heavily magnified) and Tallying (when validities are ignored). Response noise and lapses account for stochasticity in choice.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights using a power function.
    # gamma = 0 yields equal weights (Tallying); gamma -> inf yields lexicographic weights (Take The Best).
    weights = val ** gamma
    
    # Compute weighted additive scores for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To maximally distinguish Take The Best (TTB) from Weighted Additive (WADD), we must exploit WADD's compensatory nature driven by its gamma parameter. While TTB strictly decides based on the single highest-validity discriminating cue, WADD integrates all cues weighted by a non-linear transformation of their validities. We propose a 5-cue environment with linearly spaced validities. The trial pairs are designed to pit the highest-validity discriminating cue (favoring Option A) against multiple lower-validity cues (favoring Option B). By varying the number and specific validities of the compensating cues, we create a gradient of thresholds where WADD will switch from Option B to Option A as gamma increases. TTB, on the other hand, will rigidly select Option A across all these critical pairs regardless of how many lower-validity cues are stacked against it.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, every trial in the experiment pits the single highest-validity discriminating cue (which always favors Option A) against multiple lower-validity cues (which favor Option B). Take The Best (TTB) strictly follows the highest-validity discriminating cue, so it predicts that subjects will almost always choose Option A (response = 0). Weighted Additive (WADD) integrates all cues, so for a wide range of its gamma parameter, the combined weight of the lower-validity cues will exceed the single high-validity cue, leading to choices for Option B (response = 1). The overall mean of the response column directly measures the proportion of compensatory (Option B) choices, which will be near 0 for TTB and significantly higher for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1415 (var=0.0107)
- Predicted under pi_3 (simulated): 0.3475 (var=0.0281)
- Observed on real data: 0.4021 (var=0.0113)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Rationale:** To robustly dissociate WADD from TTB, we use a 6-feature environment with a steady gradient of validities. The trials are constructed to pit the highest-validity discriminating cue (favoring Option A) against varying combinations of lower-validity cues (favoring Option B). TTB will deterministically choose Option A in all these critical trials because it relies solely on the first discriminating cue. In contrast, WADD's compensatory nature allows it to choose Option B when the sum of the subjectively weighted lower-validity cues exceeds the weight of the highest-validity cue. By varying the number and specific validities of the compensating cues, we map out the exact threshold of the gamma parameter at which WADD switches from a compensatory (Tallying-like) to a non-compensatory (TTB-like) strategy, creating a stark quantitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, every trial pits a single high-validity cue (favoring Option A) against multiple lower-validity cues (favoring Option B). Since TTB strictly follows the first discriminating cue, it will deterministically favor Option A on every trial, choosing B only through random lapses. In contrast, WADD's compensatory nature allows the sum of multiple lower-validity cues to outweigh the single high-validity cue for lower values of the gamma parameter. Measuring the overall proportion of Option B choices across all trials provides a highly stable, low-variance estimate of compensatory behavior, maximizing the quantitative dissociation between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    # In all 8 trials of this design, Option A is constructed to be the Take-The-Best (TTB) choice
    # (it always possesses the highest-validity discriminating cue). Option B is the compensatory
    # choice (it possesses multiple lower-validity cues).
    # This metric simply calculates the overall proportion of times Option B was chosen.
    # TTB will only choose B due to lapse noise (epsilon), keeping this metric low.
    # WADD will choose B systematically when gamma is low enough to allow compensation,
    # leading to a significantly higher mean.
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1227 (var=0.0108)
- Predicted under pi_3 (simulated): 0.3542 (var=0.0643)
- Observed on real data: 0.2479 (var=0.0104)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_consistent = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        if ttb_winner == resp:
            ttb_consistent += 1
            
    return float(ttb_consistent / len(data))
```

**Observed (real) value:** 0.6533 (var=0.0443)
**Predicted under pi_1:** 0.8408 (var=0.0108)
**Predicted under pi_3:** 0.6640 (var=0.0320)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tallying_consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        if row['response'] == tally_choice:
            tallying_consistent += 1
        total += 1
        
    if total == 0:
        return 0.5
    return float(tallying_consistent / total)
```

**Observed (real) value:** 0.3696 (var=0.0477)
**Predicted under pi_1:** 0.1496 (var=0.0077)
**Predicted under pi_3:** 0.3227 (var=0.0319)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 2 (WADD) consistently outperforms Theory 1 (TTB). TTB rigidly predicts that subjects will always follow the highest-validity discriminating cue, leading to extreme predictions (e.g., ~15% Option B choices in Exp 1 and 2, and ~84% TTB-consistency in Exp 3) that fail to match the observed data. The data shows significant compensatory behavior (Option B choices around 25-40% in Exps 1 and 2, and Tallying-consistency around 37% in Exp 4). WADD naturally captures this intermediate behavior through its gamma parameter, which interpolates between TTB and Tallying. Since TTB is fundamentally incapable of capturing the compensatory trade-offs observed in the data, Theory 1 should be completely replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Selection / Adaptive Toolbox' theory. Instead of assuming all subjects use a single strategy, this theory should posit that decision-makers probabilistically select between distinct heuristics (e.g., Take The Best and Tallying) on a trial-by-trial basis, or that there is a population-level mixture of TTB-users and Tallying-users. This provides an alternative way to achieve intermediate compensatory behavior (mixing strict non-compensatory choices with equal-weight compensatory choices) that contrasts with WADD's unified non-linear weighting mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 5283,
  "candidates_token_count": 333,
  "total_token_count": 6341
}
```
