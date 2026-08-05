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
Decision makers use a Weighted Additive (WADD) strategy to choose between options, computing a weighted sum of cue values. To reflect the unequal reliance on cues, the provided cue validities are transformed non-linearly using a power parameter (gamma). This allows the decision maker to tune the spread between high and low validities, smoothly interpolating between Tallying (equal weights) and Take The Best (where the best cue dominates). The option with the higher weighted sum is chosen, with response noise introduced via a softmax function and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities to tune the spread between high and low validities
    weights = val ** gamma
    
    # Compute weighted sum of cues for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** To strongly dissociate Take The Best (TTB) from the Weighted Additive (WADD) strategy, we employ a 4-feature design with linearly spaced validities. TTB relies exclusively on the single highest-validity discriminating cue, ignoring all others. WADD, depending on its gamma parameter, integrates all cues. We include 'compensatory' trials where the highest-validity cue favors option A, but all remaining cues favor option B. For TTB, option A is always the deterministic choice. For WADD, the accumulation of lower-validity cues will easily override the top cue unless gamma is extremely high. We also include a trial where the sum of validities for A and B are perfectly balanced (e.g., cues 1 and 4 vs. cues 2 and 3), leading WADD to predict a tie (when gamma=1) or a slight preference, whereas TTB maintains a strong preference for the option with cue 1.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject chooses Option A on 'compensatory' trials (Trials 1 and 2). On these trials, Option A possesses only the single highest-validity cue available, while Option B possesses all remaining lower-validity cues. The Take The Best (TTB) strategy will deterministically choose Option A because it bases its decision exclusively on the single best discriminating cue. In contrast, the Weighted Additive (WADD) strategy will aggregate the evidence from the lower-validity cues, which together outweigh the single best cue for most values of the gamma parameter, thus leading WADD to predominantly choose Option B. Consequently, this metric will be close to 1 for TTB and much closer to 0 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_trial(row):
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        # Trial 1: A has the best cue, B has all the rest
        if a == [1, 0, 0, 0] and b == [0, 1, 1, 1]:
            return True
        # Trial 2: A has the second best cue, B has the rest
        if a == [0, 1, 0, 0] and b == [0, 0, 1, 1]:
            return True
        return False

    mask = data.apply(is_target_trial, axis=1)
    subset = data[mask]
    if len(subset) == 0:
        return 0.5
    
    # Return the proportion of times Option A was chosen (response == 0)
    return float((subset['response'] == 0).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8350 (var=0.0172)
- Predicted under pi_3 (simulated): 0.4969 (var=0.0682)
- Observed on real data: 0.8287 (var=0.0147)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.75]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]

**Rationale:** To effectively dissociate the Weighted Additive (WADD) strategy from Take The Best (TTB), we employ a 5-feature design with closely spaced validities. TTB evaluates options solely based on the highest-validity discriminating cue, ignoring the number or weight of all lower-validity cues. In contrast, WADD integrates all cues weighted by their validities (transformed by gamma). The trials are designed as 'compensatory' choices where the single best discriminating cue favors one option (TTB's choice), but the sum of the remaining validities strongly favors the other option (WADD's choice, for low-to-moderate gamma values). We also include trials where WADD predicts a near-tie but TTB has a deterministic preference, maximizing the quantitative divergence between the two models.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
On Trial 1, Option A is favored by the single best cue (validity 0.95), while Option B is favored by all other cues (validities 0.9, 0.85, 0.8, 0.75). Take The Best will deterministically choose Option A (response 0) because it only considers the first discriminating cue. Weighted Additive, on the other hand, will integrate all cues and, for most values of gamma, strongly favor Option B (response 1) because the sum of the remaining validities outweighs the first. Measuring the proportion of B choices on this specific trial provides a highly stable and strongly discriminative metric between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 where A = [1, 0, 0, 0, 0] and B = [0, 1, 1, 1, 1]
    is_trial_1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    if not is_trial_1.any():
        return 0.0
    return float(data.loc[is_trial_1, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1126 (var=0.0110)
- Predicted under pi_3 (simulated): 0.8126 (var=0.0166)
- Observed on real data: 0.1811 (var=0.0113)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Features are pre-sorted by validity due to the design
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred != -1:
            matches.append(1.0 if row['response'] == ttb_pred else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.6369 (var=0.0028)
**Predicted under pi_1:** 0.8289 (var=0.0104)
**Predicted under pi_3:** 0.5778 (var=0.0301)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def tally_predict(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_w = np.sum(a > b)
        b_w = np.sum(b > a)
        if a_w > b_w:
            return 0
        elif b_w > a_w:
            return 1
        else:
            return -1
            
    preds = data.apply(tally_predict, axis=1)
    valid_trials = preds != -1
    
    if valid_trials.sum() == 0:
        return 0.5
        
    match = (preds[valid_trials] == data.loc[valid_trials, 'response']).mean()
    return float(match)

```

**Observed (real) value:** 0.5855 (var=0.0012)
**Predicted under pi_1:** 0.3718 (var=0.0030)
**Predicted under pi_3:** 0.6198 (var=0.0427)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, Take The Best (pi_1) accurately captures behavior on strongly compensatory trials (Exps 1 and 2), where subjects overwhelmingly choose the option favored by the single best cue, ignoring the remaining cues. However, in Experiment 4, the observed data shows a much higher match with Tallying/Weighted Additive (pi_3) than TTB predicts, and in Experiment 3 the match with TTB is substantially lower than pi_1 expects. This indicates that neither a pure TTB nor a pure WADD strategy is task-invariant. Instead, subjects appear to exhibit a mix of these behaviors, perhaps relying on TTB by default but switching to a compensatory strategy (like Tallying) when the top cues are less discriminative or when cognitive load permits.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Strategy Selection' or 'Mixture' theory. This theory should posit that decision-makers maintain a repertoire of strategies (e.g., Take The Best and Tallying/WADD). On any given trial or across subjects, there is a probability of selecting the fast-and-frugal TTB strategy versus a more compensatory strategy. Alternatively, propose a 'Threshold' model where evidence is accumulated in order of validity, but lower-validity cues can override the top cue if their combined weight exceeds a certain threshold. This would allow the model to capture the strong TTB-like behavior in Exps 1 and 2 while accommodating the compensatory behavior observed in Exps 3 and 4."
}
```

## Usage

```json
{
  "prompt_token_count": 5527,
  "candidates_token_count": 349,
  "total_token_count": 6746
}
```
