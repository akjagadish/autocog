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
Weighted Additive Model (WADD). Decision-makers integrate all available information by computing a sum of each option's features weighted by their respective validities, transformed into log-odds. The option with the higher weighted sum is chosen, providing a fully compensatory decision rule. Response noise enters through a softmax over the weighted sums with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights to act as normative Bayesian evidence
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over weighted sums with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from the Weighted Additive Model (WADD), this design exploits the compensatory nature of WADD versus the non-compensatory nature of TTB. We use five features with validities set such that the log-odds weight of the highest-validity cue is less than the sum of the log-odds weights of several lower-validity cues. In most of these trials, one option possesses the single highest discriminating validity cue (leading TTB to strictly prefer it), while the other option possesses a combination of lower-validity cues whose combined log-odds evidence outweighs the single best cue (leading WADD to strictly prefer the opposite option). We also include trials where the top cue is tied, forcing TTB to the second cue while WADD integrates the remaining evidence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Take The Best (TTB) against the Weighted Additive Model (WADD) in every single trial. Because of the specific compensatory structure of the cues and their log-odds weights, TTB and WADD predict exactly opposite choices for all 8 trial pairs. This metric simply computes the proportion of choices that align with the TTB prediction (the option favored by the first discriminating cue in descending order of validity). Data simulated under TTB will yield values close to 1 (modulo noise), whereas data simulated under WADD will yield values close to 0.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
        
        if ttb_pred is not None and row['response'] == ttb_pred:
            ttb_matches += 1
            
    return float(ttb_matches / total)

### RESULTS
- Predicted under pi_1 (simulated): 0.8502 (var=0.0120)
- Predicted under pi_3 (simulated): 0.3042 (var=0.0231)
- Observed on real data: 0.6950 (var=0.1026)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.8, 0.75, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** To maximally distinguish the Weighted Additive Model (WADD) from Take The Best (TTB), this design uses a 5-cue structure where the validities are set such that the highest-validity cue is highly predictive but its log-odds weight is outweighed by the sum of several lower-validity cues. In all trials, Option A possesses the single best discriminating cue, causing TTB to strictly prefer Option A. However, Option B possesses a combination of lower-validity cues that collectively provide stronger evidence in terms of log-odds, causing WADD to strictly prefer Option B. By varying which cues are tied and which lower-validity cues Option B has, we ensure robust quantitative dissociation across different levels of cue conflict.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design is perfectly constructed to pit TTB against WADD on every single trial. Option A always possesses the single highest-validity discriminating cue (causing TTB to reliably choose A, response=0), whereas Option B always possesses a combination of lower-validity cues that collectively outweigh Option A's cue in log-odds terms (causing WADD to reliably choose B, response=1). Therefore, the overall proportion of Option B choices (the mean of the 'response' column) will be close to 0 under TTB and close to 1 under WADD, providing maximum and robust discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1644 (var=0.0130)
- Predicted under pi_3 (simulated): 0.6940 (var=0.0168)
- Observed on real data: 0.5200 (var=0.2496)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    is_1 = a_tuples == (1, 0, 0, 0)
    is_3 = a_tuples == (0, 1, 1, 1)
    is_5 = a_tuples == (1, 1, 0, 0)
    is_6 = a_tuples == (0, 0, 1, 1)
    
    correct = 0
    total = 0
    
    if is_1.any():
        correct += (data.loc[is_1, 'response'] == 0).sum()
        total += is_1.sum()
    if is_3.any():
        correct += (data.loc[is_3, 'response'] == 1).sum()
        total += is_3.sum()
    if is_5.any():
        correct += (data.loc[is_5, 'response'] == 0).sum()
        total += is_5.sum()
    if is_6.any():
        correct += (data.loc[is_6, 'response'] == 1).sum()
        total += is_6.sum()
        
    if total == 0:
        return 0.5
    return float(correct / total)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_1:** 0.8250 (var=0.0123)
**Predicted under pi_3:** 0.4153 (var=0.0068)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    tally_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            tally_choice = 0
        elif b_wins > a_wins:
            tally_choice = 1
        else:
            continue
            
        tally_match.append(row['response'] == tally_choice)
        
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_1:** 0.1575 (var=0.0102)
**Predicted under pi_3:** 0.5614 (var=0.0029)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data consistently falls between the predictions of TTB (pi_1) and WADD (pi_3), often hovering near 0.5 (as seen in Experiments 2, 3, and 4) with surprisingly low variance in some cases. TTB predicts extreme outcomes (e.g., 0.85, 0.16, 0.82) because it is strictly non-compensatory, while WADD predicts the opposite extremes due to its fully compensatory nature. The reality is that the average human behavior in these binary choice tasks reflects neither strict one-reason decision making nor optimal Bayesian evidence integration. The consistent ~0.50 observed metrics in several experiments suggest either high response noise, a mixture of strategies across trials, or a simpler heuristic that relies on counting rather than weighting (which was partially tested but also yielded 0.5). Neither theory adequately captures this central tendency.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Selection' or 'Mixed Heuristics' theory. Instead of assuming all subjects use TTB or WADD exclusively, this theory should propose that decision-makers probabilistically sample between a non-compensatory heuristic (like TTB) and a simple compensatory heuristic (like equal-weights Tallying) on a trial-by-trial basis, depending on the difficulty of the choice. This would naturally pull the population-level predictions closer to 0.5 in highly conflicting trials while retaining some sensitivity to the dominant cue, better matching the observed data across all four experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5213,
  "candidates_token_count": 353,
  "total_token_count": 6386
}
```
