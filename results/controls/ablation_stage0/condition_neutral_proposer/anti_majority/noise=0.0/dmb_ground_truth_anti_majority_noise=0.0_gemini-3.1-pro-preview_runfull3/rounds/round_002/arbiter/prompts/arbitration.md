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
People possess an adaptive toolbox of decision heuristics, primarily Take The Best (TTB) and Tallying, and select among them probabilistically. TTB is a non-compensatory, one-reason decision rule that relies solely on the most valid discriminating cue, while Tallying is an equal-weight compensatory rule that simply counts the number of cues favoring each option. By probabilistically mixing these distinct strategies on a trial-by-trial basis (or reflecting a population-level mixture of strategy users), decision-makers exhibit intermediate compensatory behavior. This avoids the need for a unified, complex non-linear cue weighting mechanism like Weighted Additive (WADD) models. Decision stochasticity is accounted for by the probabilistic strategy selection, strategy-specific softmax response noise, and an overarching lapse rate. The mixture is heavily biased toward TTB, reflecting the empirical dominance of one-reason decision making in these environments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) --- 
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # --- Tallying ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # --- Strategy Mixture ---
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** To maximally distinguish the TTB/Tallying mixture model (THEORY 1) from the WADD model (THEORY 2), we exploit the fact that the mixture model relies only on the rank of the highest valid discriminating cue (for TTB) and the unweighted count of winning cues (for Tallying). WADD, on the other hand, is sensitive to the specific absolute validities of all cues. We use a 5-feature environment with linearly spaced validities and construct a series of pairs where the structural signals for TTB and Tallying are held perfectly constant. For example, in several pairs, Option A wins on the single highest-validity discriminating cue (triggering TTB), while Option B wins on exactly two lower-validity cues (triggering Tallying). THEORY 1 predicts identical choice probabilities across all these pairs because the TTB and Tallying outputs are identical. THEORY 2 (WADD) predicts systematically varying choice probabilities because the actual validities of the compensating cues change across pairs (e.g., cues 2 and 3 vs. cues 4 and 5).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The TTB/Tallying Mixture model relies on two invariant signals: the rank of the highest discriminating cue (for TTB) and the unweighted count of winning cues (for Tallying). In both Trial 6 and Trial 8, Option A and Option B each have exactly one cue, meaning Tallying is tied. Furthermore, in both trials, Option A's cue is higher in validity than Option B's, so TTB strictly prefers A. Because the Mixture model's TTB implementation is purely rank-based and ignores absolute validity magnitudes, it predicts exactly the same probability of choosing A in Trial 6 and Trial 8. Thus, the expected value of P(A|t6) - P(A|t8) is exactly 0 under the Mixture model.

In contrast, WADD integrates absolute validities using a power function (gamma). At moderate to high gamma values (which dominate the adversary's prior), the weights of lower-validity cues vanish exponentially toward zero. In Trial 6, the cues are the highest available (0.95 vs 0.85), so their difference remains large and WADD strongly prefers A (P(A) ~ 1.0). In Trial 8, the cues are much lower (0.75 vs 0.65). At high gamma, these values become infinitesimally small, making their absolute difference effectively zero. Consequently, WADD becomes indifferent and predicts a probability near 0.5 for Trial 8. This leads WADD to consistently predict a massive positive difference (often > 0.3) for P(A|t6) - P(A|t8), perfectly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project lists to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 6: A has cue 1 (0.95), B has cue 2 (0.85)
    t6_A = (1, 0, 0, 0, 0)
    t6_B = (0, 1, 0, 0, 0)
    
    # Trial 8: A has cue 3 (0.75), B has cue 4 (0.65)
    t8_A = (0, 0, 1, 0, 0)
    t8_B = (0, 0, 0, 1, 0)
    
    mask_6 = (a_tuples == t6_A) & (b_tuples == t6_B)
    mask_8 = (a_tuples == t8_A) & (b_tuples == t8_B)
    
    if not mask_6.any() or not mask_8.any():
        return 0.0
        
    # Calculate probability of choosing A (response == 0)
    pA_t6 = (data.loc[mask_6, 'response'] == 0).mean()
    pA_t8 = (data.loc[mask_8, 'response'] == 0).mean()
    
    return float(pA_t6 - pA_t8)

### RESULTS
- Predicted under pi_4 (simulated): 0.0033 (var=0.0228)
- Predicted under pi_3 (simulated): 0.1833 (var=0.0475)
- Observed on real data: 0.0133 (var=0.0126)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Rationale:** To perfectly dissociate THEORY 1 (WADD) from THEORY 2 (TTB/Tallying mixture), we construct a series of choice pairs where the structural signals for TTB and Tallying are held perfectly constant, while the absolute validities of the compensating cues vary. In all pairs, Option A is favored by the single highest-validity discriminating cue (triggering TTB to choose A with a constant score difference of 1 vs 0), while Option B is favored by exactly two lower-validity cues (triggering Tallying to choose B with a constant score difference of 2 vs 1). THEORY 2 will predict the exact same probability of choosing A across all these pairs because its components only care about rank (TTB) and unweighted counts (Tallying). Conversely, THEORY 1 (WADD) sums the non-linearly transformed validities, so it will predict systematically different choice probabilities as the actual validities of Option B's compensating cues shift from high to low.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Theory 2 predicts a constant choice probability across all 8 trial types, as TTB and Tallying signals are structurally identical across pairs (A always wins TTB 1-0, B always wins Tallying 2-1). Thus, any variance in choice rates across the 8 trial types is purely due to binomial noise. Theory 1 (WADD) integrates the exact validities, predicting systematically different choice probabilities across the 8 pairs as the compensating cues shift from strong to weak. By calculating the standard deviation of the mean choice rates across the 8 trial types, we capture this difference: Theory 2 will have a low standard deviation (only noise), while Theory 1 will have a significantly higher standard deviation (true probability variance + noise).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Create a hashable surrogate for the trial type using Option B's ratings
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the mean response (probability of choosing B) for each trial type
    trial_means = data.groupby(b_str)['response'].mean()
    
    # Calculate the standard deviation of these means
    return float(np.std(trial_means))

### RESULTS
- Predicted under pi_4 (simulated): 0.0105 (var=0.0012)
- Predicted under pi_3 (simulated): 0.0503 (var=0.0036)
- Observed on real data: 0.1327 (var=0.0146)

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
**Predicted under pi_4:** 0.7275 (var=0.0221)
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
**Predicted under pi_4:** 0.2754 (var=0.0178)
**Predicted under pi_3:** 0.3227 (var=0.0319)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4021 (var=0.0113)
**Predicted under pi_4:** 0.2535 (var=0.0108)
**Predicted under pi_3:** 0.3475 (var=0.0281)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # In all 8 trials of this design, Option A is constructed to be the Take-The-Best (TTB) choice
    # (it always possesses the highest-validity discriminating cue). Option B is the compensatory
    # choice (it possesses multiple lower-validity cues).
    # This metric simply calculates the overall proportion of times Option B was chosen.
    # TTB will only choose B due to lapse noise (epsilon), keeping this metric low.
    # WADD will choose B systematically when gamma is low enough to allow compensation,
    # leading to a significantly higher mean.
    return float(data['response'].mean())
```

**Observed (real) value:** 0.2479 (var=0.0104)
**Predicted under pi_4:** 0.2737 (var=0.0273)
**Predicted under pi_3:** 0.3542 (var=0.0643)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (TTB/Tallying mixture) perfectly captures Experiment 1, correctly predicting near-zero difference between trials where the absolute validities shift but the structural ranks remain the same. However, it completely fails Experiment 2, predicting almost zero variance across trials where subjects clearly showed sensitivity to the specific compensating cues. Theory 2 (WADD) captures some of the variance in Experiment 2 (and fits better on Exps 3, 4, 5) but drastically fails Experiment 1 by predicting a large difference where humans show none. Because both theories capture distinct, mutually exclusive aspects of the data, a completely new theory is needed to bridge the gap.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that blends the strict rank-based nature of Take-The-Best with the compensatory sensitivity of WADD. For example, a 'Mixture of TTB and WADD' (rather than Tallying) or a 'Threshold-based WADD' where cues only contribute to a compensatory sum if their validities exceed a certain threshold. A TTB/WADD mixture would allow strict one-reason decision making to dominate (explaining Exp 1's near-zero difference), while the WADD component would evaluate the exact validities of the compensating cues when TTB is not used (explaining Exp 2's high variance)."
}
```

## Usage

```json
{
  "prompt_token_count": 6880,
  "candidates_token_count": 313,
  "total_token_count": 8741
}
```
