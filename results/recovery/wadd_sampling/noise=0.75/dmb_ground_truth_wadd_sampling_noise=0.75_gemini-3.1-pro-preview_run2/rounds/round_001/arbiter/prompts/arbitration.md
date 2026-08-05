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
People make decisions by computing a subjective value for each option through a Weighted Additive (WADD) process. They take a weighted sum of the features for each option, where the weight of each feature is the log-odds of its provided cue validity. This transformation appropriately scales probabilities into additive evidence. The probability of choosing an option is then determined by a softmax over the options' subjective values, combined with a lapse rate for random errors. This compensatory mechanism allows multiple lower-validity cues to jointly outweigh a single high-validity cue, naturally predicting the graded, intermediate behavior observed on conflict trials compared to non-compensatory heuristics like Take The Best or Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds to represent additive evidence
    # Clip to avoid division by zero or log(0)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.75, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 3: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we use a 4-feature design where the validities are chosen such that the log-odds of the lower-validity cues sum to more than the log-odds of the highest-validity cue. For instance, with validities [0.9, 0.8, 0.75, 0.6], the log-odds are approximately [2.20, 1.39, 1.10, 0.41]. When Option A possesses only the most valid cue (0.9) and Option B possesses the second and third most valid cues (0.8 and 0.75), TTB will deterministically choose Option A because it only considers the first discriminating cue. In contrast, WADD will choose Option B because the combined log-odds weight of the two lower cues (1.39 + 1.10 = 2.49) strictly outweighs the highest cue (2.20). By systematically varying whether the lower cues can overcome the highest discriminating cue, we can robustly separate the non-compensatory 'one-reason' logic of TTB from the compensatory, evidence-summing logic of WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the 'conflict' trials where Take The Best (TTB) and Weighted Additive (WADD) predict opposite choices. It calculates the proportion of times the subject's choice aligns with the WADD prediction on these specific trials. For a subject following TTB, this metric will be close to 0 (modulo response noise), whereas for a subject following WADD, it will be significantly higher than 0.5. Since half of the trials in this design are conflict trials, the metric is computed over a large number of trials per subject, ensuring low within-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    w = np.array([2.19722458, 1.38629436, 1.09861229, 0.40546511])
    
    wadd_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * w)
        wadd_b = np.sum(b * w)
        wadd_winner = 0 if wadd_a > wadd_b else 1
        
        if ttb_winner is not None and ttb_winner != wadd_winner:
            subject_choice = row['response']
            wadd_choices.append(1 if subject_choice == wadd_winner else 0)
            
    if len(wadd_choices) == 0:
        return 0.5
        
    return float(np.mean(wadd_choices))

### RESULTS
- Predicted under pi_1 (simulated): 0.1442 (var=0.0139)
- Predicted under pi_3 (simulated): 0.6379 (var=0.0059)
- Observed on real data: 0.5275 (var=0.0041)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.88, 0.78, 0.72, 0.68, 0.62]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) models, we employ a 5-feature design. The validities are carefully calibrated such that the log-odds of combinations of lower-validity cues exceed the log-odds of the single most valid discriminating cue. For instance, with validities [0.88, 0.78, 0.72, 0.68, 0.62], the log-odds weights are approximately [1.99, 1.26, 0.94, 0.75, 0.49]. In trial pairs where Option A possesses only the single highest discriminating cue, TTB will deterministically choose Option A. However, Option B is endowed with multiple lower-validity cues whose combined log-odds strictly outweigh Option A's single cue, causing WADD to select Option B. By systematically varying the dispersion of these compensating cues across different levels of the validity hierarchy, we can robustly separate the non-compensatory 'one-reason' logic of TTB from the compensatory, evidence-summing logic of WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, every trial in this experiment features a conflict where Option A possesses the single most valid discriminating cue (making it the deterministic choice for Take The Best), whereas Option B possesses a combination of lower-validity cues whose combined log-odds strictly outweigh Option A's single cue (making it the preferred choice for Weighted Additive). Because the 'response' column encodes choosing Option A as 0 and Option B as 1, the mean response directly represents the proportion of choices consistent with the compensatory WADD model over the non-compensatory TTB model. TTB's predicted mean will be strictly below 0.5 (driven only by noise), while WADD's predicted mean will be strictly above 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1527 (var=0.0092)
- Predicted under pi_3 (simulated): 0.6508 (var=0.0056)
- Observed on real data: 0.5571 (var=0.0045)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction (assumes features are sorted by descending validity)
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Tallying prediction
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            tally_pred = 0
        elif sum_b > sum_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        # Focus strictly on conflict trials where the theories make opposite predictions
        if ttb_pred is not None and tally_pred is not None and ttb_pred != tally_pred:
            matches.append(1 if row['response'] == ttb_pred else 0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))

```

**Observed (real) value:** 0.4775 (var=0.0054)
**Predicted under pi_1:** 0.8421 (var=0.0086)
**Predicted under pi_3:** 0.4117 (var=0.0076)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        tally_pred = 0 if a_wins > b_wins else (1 if b_wins > a_wins else None)
        
        val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        cue_order = np.argsort(-val)
        ttb_pred = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_pred = 0
                break
            elif b[j] > a[j]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            if row['response'] == tally_pred:
                tally_match += 1
            total += 1
            
    return tally_match / total if total > 0 else 0.5
```

**Observed (real) value:** 0.5340 (var=0.0037)
**Predicted under pi_1:** 0.1371 (var=0.0093)
**Predicted under pi_3:** 0.4159 (var=0.0049)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics on conflict trials consistently hover around 0.5 (e.g., 0.5275, 0.5571, 0.4775, 0.5340). Theory 1 (Take The Best) incorrectly predicts highly extreme values on these metrics (e.g., ~0.15 or ~0.85) because it deterministically favors the highest-validity cue. Theory 2 (Weighted Additive) predicts more moderate values (e.g., ~0.41 to ~0.65) and is generally closer to the observed data, but still systematically deviates from the near-0.5 empirical mean. The fact that aggregate behavior is split almost exactly 50/50 on conflict trials pitting one high-validity cue against multiple lower-validity cues strongly suggests either high population heterogeneity (some subjects use TTB, others use WADD/Tallying) or trial-by-trial strategy variability.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture' theory. Instead of assuming all subjects use a pure non-compensatory heuristic (TTB) or a pure compensatory strategy (WADD), this theory should posit that individuals stochastically draw from a repertoire of strategies (e.g., TTB and Tallying/WADD), or that the population is a mixture of heuristic and compensatory decision-makers. A mixture parameter will allow the model to capture the ~0.5 aggregate observed metric on conflict trials, effectively bridging the gap between the extreme predictions of pure TTB and the weighted predictions of WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 5914,
  "candidates_token_count": 374,
  "total_token_count": 7141
}
```
