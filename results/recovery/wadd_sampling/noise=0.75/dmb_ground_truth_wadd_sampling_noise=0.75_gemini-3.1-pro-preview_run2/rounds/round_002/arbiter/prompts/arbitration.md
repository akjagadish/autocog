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
Strategy Mixture Theory: People do not rely exclusively on a single decision strategy. Instead, individuals or trials draw from a repertoire of strategies, specifically a mixture of a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). The probability of using TTB versus Tallying is governed by a mixture parameter, allowing the model to capture intermediate choice proportions on conflict trials where the two strategies make opposite predictions. Tallying treats all cues equally, ensuring it provides a genuine compensatory contrast to TTB.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Tallying (Compensatory) Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_comp = np.array([a_wins, b_wins])
    
    z_comp = beta * (scores_comp - np.max(scores_comp))
    e_comp = np.exp(z_comp)
    p_comp = e_comp / np.sum(e_comp)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_comp
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.93, 0.9, 0.88, 0.52, 0.52, 0.52]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Strategy Mixture Theory (TTB + Tallying) from the Weighted Additive (WADD) theory by exploiting the non-linear transformation of validities into log-odds. We use a 6-feature design with one very high validity cue (0.93), two high validity cues (0.90, 0.88), and three very low validity cues (0.52). Because WADD uses log-odds weights, the sum of the two high-validity cues (log-odds ~4.18) strictly outweighs the highest validity cue combined with all low-validity cues (log-odds ~2.82). In critical trials, Option A possesses the highest validity cue and several low validity cues, while Option B possesses the two high validity cues. TTB will choose A (because it has the single best cue). Tallying will also choose A (because A has more total cues). Consequently, any mixture of TTB and Tallying will strictly prefer A. However, WADD will strongly prefer B, providing a definitive qualitative and quantitative dissociation.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
The metric isolates the specific subset of 'critical' trials (Trials 1, 2, and 7) where the Strategy Mixture theory and WADD theory make completely opposite, strong predictions. In these trials, Option A possesses the single most valid cue and a higher total number of cues, leading both TTB and Tallying (and thus any mixture of them) to strongly prefer Option A. Conversely, Option B possesses the second and third most valid cues, which under WADD's log-odds weighting strictly sum to a higher subjective value than Option A's cues, leading WADD to strongly prefer Option B. By computing the proportion of Option B choices on these specific trials, the metric drives a massive mean difference (near 0 for Mixture, near 1 for WADD) with very low within-theory variance.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify critical trials where TTB and Tallying both prefer Option A, but WADD prefers Option B.
    # These are trials where Option A has the highest validity cue (A[0] == 1), Option B has the next two highest (B[1] == 1, B[2] == 1),
    # and Option A has more positive cues overall (sum(A) > sum(B)).
    is_critical = data.apply(
        lambda row: sum(row['option_a_ratings']) > sum(row['option_b_ratings']) 
                    and row['option_a_ratings'][0] == 1 
                    and row['option_b_ratings'][1] == 1,
        axis=1
    )
    if not is_critical.any():
        return 0.5
    
    # Return the proportion of times Option B was chosen on these critical trials.
    # Strategy Mixture will be close to 0 (since both TTB and Tallying prefer A).
    # WADD will be close to 1 (since the log-odds of cues 2 and 3 outweigh cue 1 and the minor cues).
    return float(data.loc[is_critical, 'response'].mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.1897 (var=0.0224)
- Predicted under pi_3 (simulated): 0.7826 (var=0.0190)
- Observed on real data: 0.4738 (var=0.0085)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.85, 0.84, 0.83, 0.55, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) theory from the Strategy Mixture Theory (TTB + Tallying), we use a 5-feature design with three high-validity cues with very similar validities and two low-validity cues. Because WADD uses the log-odds of validities as weights, the combined weight of the second and third most valid cues strictly outweighs the single most valid cue combined with all low-validity cues. In critical trials, Option A possesses the highest validity cue alongside multiple low-validity cues, while Option B possesses only the second and third most valid cues. TTB will select Option A because it has the best cue. Tallying will also select Option A because it has more cues in total (e.g., 3 vs. 2). Consequently, any mixture of TTB and Tallying must strictly prefer Option A. In stark contrast, WADD will strongly prefer Option B, providing a definitive dissociation. Additional trials pit TTB and Tallying against each other in ways that align WADD with one or the other, ensuring robust parameter estimation across theories.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times the subject selects the option with the feature profile [0, 1, 1, 0, 0] when it is available (which occurs in Trials 1, 2, and 3). Under the WADD theory, the combined weight of cues 2 and 3 (the second and third most valid cues) strictly outweighs the combined weight of cue 1 and the low-validity cues, leading to a strong preference for the [0, 1, 1, 0, 0] option. In contrast, the Strategy Mixture Theory (TTB + Tallying) will strongly prefer the alternative option, because TTB favors the option with the best cue (cue 1), and Tallying either favors the alternative (which has more cues in total) or is indifferent. Thus, WADD predicts a value close to 1, while the Mixture theory predicts a value close to 0.

Source:
def metric(data: pd.DataFrame) -> float:
    target_profile = (0, 1, 1, 0, 0)
    matches = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == target_profile:
            total += 1
            if row['response'] == 0:
                matches += 1
        elif b == target_profile:
            total += 1
            if row['response'] == 1:
                matches += 1
    if total == 0:
        return 0.5
    return float(matches / total)

### RESULTS
- Predicted under pi_4 (simulated): 0.2092 (var=0.0138)
- Predicted under pi_3 (simulated): 0.8163 (var=0.0124)
- Observed on real data: 0.4550 (var=0.0058)

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
**Predicted under pi_4:** 0.4392 (var=0.0459)
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
**Predicted under pi_4:** 0.5032 (var=0.0323)
**Predicted under pi_3:** 0.4159 (var=0.0049)

### Experiment 5
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.5275 (var=0.0041)
**Predicted under pi_4:** 0.5413 (var=0.0378)
**Predicted under pi_3:** 0.6379 (var=0.0059)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5571 (var=0.0045)
**Predicted under pi_4:** 0.5035 (var=0.0409)
**Predicted under pi_3:** 0.6508 (var=0.0056)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, we see a striking pattern: the observed metric values hover around 0.45-0.47, squarely in the middle of the predictions made by the Strategy Mixture Theory (TTB + Tallying, ~0.20) and the Weighted Additive Theory (WADD, ~0.80). In these critical trials, Option A is favored by both TTB and Tallying, while Option B is favored by WADD. The fact that subjects choose Option B nearly half the time indicates that a significant portion of the behavior is driven by a WADD-like compensatory process, but not entirely, as WADD overpredicts Option B choices. The TTB + Tallying mixture fails because both of its component strategies pull in the same direction on these trials, preventing it from predicting the ~50% of choices that align with WADD. Neither theory adequately captures the data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory: 'Strategy Mixture of TTB and WADD'. Since the data consistently falls halfway between the predictions of a purely non-compensatory approach (TTB) and a purely compensatory approach with log-odds weighting (WADD), the new theory should posit that decision-makers draw from a repertoire containing both TTB and WADD. By using a mixture parameter to blend TTB (which selects the option with the highest valid cue) and WADD (which computes subjective value via log-odds weighted sums), this theory will naturally predict the intermediate (~0.50) choice proportions observed on conflict trials where TTB and WADD make opposite predictions."
}
```

## Usage

```json
{
  "prompt_token_count": 6967,
  "candidates_token_count": 368,
  "total_token_count": 8544
}
```
