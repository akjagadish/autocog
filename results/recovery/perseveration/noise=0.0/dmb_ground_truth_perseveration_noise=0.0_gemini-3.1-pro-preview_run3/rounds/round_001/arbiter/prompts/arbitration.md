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
Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 4: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from a Weighted Additive (WADD) strategy that is subject to strong spatial biases and lapses, we design trials where the single highest-validity discriminating cue strongly opposes the sum of the remaining lower-validity cues. In TTB, the choice is deterministically driven by the first discriminating cue in descending order of validity, completely ignoring the cumulative weight of all other cues. In WADD, the sum of the lower-validity cues can easily outweigh the single highest-validity cue. By symmetrically alternating whether Option A or Option B possesses the highest-validity cue or the bulk of the lower-validity cues, we can also separate true WADD integration from a simple Option A positional bias.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, TTB and an unbiased WADD predict exactly the opposite choice on 6 out of the 8 unique trial types. On these critical trials, TTB's choice is driven by the single highest-validity discriminating cue, whereas WADD's choice is driven by the sum of the remaining lower-validity cues. This metric calculates the proportion of subject choices that agree with the TTB prediction specifically on these divergent trials. TTB will produce an agreement score between 0.75 (maximum lapse) and 1.0 (deterministic). WADD, on the other hand, will either consistently choose its preferred option (agreement ~ 0.0), or, if dominated by a strong spatial bias for Option A or Option B, will choose A or B uniformly across these trials (since the TTB prediction alternates symmetrically between A and B, yielding an agreement of ~ 0.5). Thus, WADD's agreement is strictly bounded below 0.5, creating a massive, robust gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # TTB predictions
    diff = a_mat - b_mat
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(4):
            if diff[i, j] > 0:
                ttb_preds[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_preds[i] = 1
                break
                
    # WADD predictions (without spatial bias)
    score_a = a_mat @ validities
    score_b = b_mat @ validities
    wadd_preds = (score_b > score_a).astype(int)
    
    # Isolate trials where TTB and WADD (unbiased) predict opposite choices
    mask = ttb_preds != wadd_preds
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values
    agreement = (responses[mask] == ttb_preds[mask]).mean()
    return float(agreement)

### RESULTS
- Predicted under pi_1 (simulated): 0.8558 (var=0.0065)
- Predicted under pi_3 (simulated): 0.4614 (var=0.0072)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Rationale:** This design dissociates WADD (with its potential for spatial bias and feature integration) from Take The Best (TTB). We include 'compensatory' trials where the highest-validity cue points to one option while the sum of the remaining cues points to the other, separating TTB's non-compensatory one-reason decision making from WADD's integration. Crucially, we also include 'tie' trials where both options have identical feature profiles; TTB must guess (predicting exactly 50/50), whereas the advocated WADD model will reveal any underlying spatial/positional bias (e.g., a baseline preference for Option A).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the difference in the probability of choosing Option A between Trial 1 (where the single best cue favors B, but the sum of the rest favors A) and Trial 2 (where the single best cue favors A, but the sum of the rest favors B). Take The Best (TTB) follows the top cue, meaning it will rarely choose A in Trial 1 and often choose A in Trial 2, yielding a strongly negative difference (approaching -1). The advocated WADD model integrates all cues, meaning it favors A in Trial 1 and B in Trial 2, yielding a positive difference (or 0 if spatial bias dominates). This creates a massive, consistent gap between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (0, 1, 1, 1) and \
               tuple(int(x) for x in row['option_b_ratings']) == (1, 0, 0, 0)
               
    def is_t2(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (1, 0, 0, 0) and \
               tuple(int(x) for x in row['option_b_ratings']) == (0, 1, 1, 1)
               
    t1_mask = data.apply(is_t1, axis=1)
    t2_mask = data.apply(is_t2, axis=1)
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.5
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)

### RESULTS
- Predicted under pi_1 (simulated): -0.7200 (var=0.0474)
- Predicted under pi_3 (simulated): 0.1400 (var=0.0701)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        ttb_preds.append(pred)
        
    ttb_preds = np.array(ttb_preds)
    responses = data['response'].values
    
    return float(np.mean(ttb_preds == responses))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_1:** 0.8575 (var=0.0070)
**Predicted under pi_3:** 0.4029 (var=0.0166)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Calculate number of feature-wise wins for each option
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    responses = data['response'].values
    
    # Identify trials where Tallying has a strict preference
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    # Check if subject's response is consistent with Tallying's preference
    consistent = (tally_prefers_a & (responses == 0)) | (tally_prefers_b & (responses == 1))
    strict_trials = tally_prefers_a | tally_prefers_b
    
    if np.sum(strict_trials) == 0:
        return 0.5
        
    # Return the proportion of Tallying-consistent choices on strict trials
    return float(np.sum(consistent[strict_trials]) / np.sum(strict_trials))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_1:** 0.1459 (var=0.0108)
**Predicted under pi_3:** 0.5791 (var=0.0214)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metrics consistently sit exactly at 0.5000 (or 0.0000 for difference metrics) with a between-subject variance of exactly 0.0000. This highly specific pattern indicates that subjects are not using the features to guide their choices at all; instead, every single subject is choosing exactly at random (50/50) on every trial. Theory 1 (Take The Best) predicts strong, feature-driven deterministic choices, completely failing to match the empirical data. Theory 2 (WADD with bias/lapse) gets closer due to its lapse parameter, but still predicts some feature integration and variance. Because the data overwhelmingly points to pure random guessing across the board, we need a theory that explicitly models complete task disengagement.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Random Guessing' or 'Complete Disengagement' theory. The observed data across all experiments perfectly aligns with subjects choosing completely at random, ignoring all feature values and validities. The new theory should postulate that subjects do not engage with the multi-attribute decision task at all, and instead simply flip a coin for every choice. The corresponding model should have no free parameters (or a fixed lapse rate of 1.0) and simply return [0.5, 0.5] for all stimuli."
}
```

## Usage

```json
{
  "prompt_token_count": 5715,
  "candidates_token_count": 310,
  "total_token_count": 6697
}
```
