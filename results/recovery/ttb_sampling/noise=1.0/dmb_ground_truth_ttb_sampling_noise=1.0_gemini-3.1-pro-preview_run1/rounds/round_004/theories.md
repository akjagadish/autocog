# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

**Rationale:** Following the arbiter's diagnosis, this theory implements a Strategy Mixture model where individuals stochastically alternate between a lexicographic strategy (Take-The-Best) and an equal-weight compensatory strategy (Tallying). By mixing these strategies with a stable probability `p_ttb` centered around 0.5 (sampled from [0.4, 0.6]), the model reproduces the ~0.50 choice rates on conflict trials without collapsing into pure randomness. The narrow range for `p_ttb` ensures that the between-subject variance remains low, faithfully matching the empirical distributions across all experiments.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `p_ttb`: `[0.4, 0.6]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** Non-linear Weighted Additive (WADD) Theory with Power Weights and Max-Normalized Validities: Decision-makers integrate all available cues simultaneously in a compensatory manner. The subjective weight assigned to each cue is determined by a power transformation of its objective validity, scaled by a parameter gamma. To ensure gamma operates consistently across experiments with varying validity scales, the objective validities are first normalized by their maximum value. This stable, bounded non-linear mapping ensures that cues with zero validity receive zero subjective weight, while allowing the model to naturally balance the top cue against the remaining cues on conflict trials. A temperature-parameterized softmax choice rule is used to smoothly control stochasticity.

**Rationale:** Following the critic's advice on Iteration 7, I expanded the gamma range to [0.0, 20.0] to provide more headroom for overweighting the top cue. I also replaced the precision parameter beta with a temperature parameter tau (range [0.01, 5.0]) in the softmax choice rule. Optimizing temperature rather than precision often yields a smoother loss landscape for tuning stochasticity, which should help the model better capture the near-chance choices and low consistency on conflict trials.

**Parameters:**
  - `tau`: `[0.01, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `gamma`: `[0.0, 20.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Max-normalize validities to make gamma scale-invariant across experiments
    v_norm = validities / np.max(validities)
    
    # Power transformation of normalized validities to subjective weights
    weights = v_norm ** gamma
    weights = weights / np.sum(weights)
    
    # Compute weighted sum of features for each option
    scores = stim @ weights
    
    # Standard softmax choice rule with temperature
    z = scores / tau
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Sequential Probabilistic Cue Integration: Decision-makers evaluate options by sequentially sampling cues with a probability proportional to their validity via a softmax function. Upon sampling a cue, evidence is accumulated for the option it favors. If the absolute difference in accumulated evidence reaches a decision threshold, the process terminates and the favored option is chosen. If a maximum number of samples is reached without crossing the threshold, the decision-maker falls back to a noisy integration of all accumulated evidence. By constraining the max samples and softmax temperature, the model avoids degenerate over-sampling of a single cue, naturally interpolating between non-compensatory heuristics and compensatory tallying.

**Rationale:** Applying the minimal edit requested by the critic: reducing the upper bound of 'gamma' to 3.0 to further prevent the softmax sampling from becoming too deterministic and over-predicting TTB-like behavior in Experiments 6, 8, and 10. Additionally, slightly expanding 'max_samples' to allow values up to 6, enabling a slightly deeper evidence integration before falling back to the noisy tally. The core mechanism remains identical.

**Parameters:**
  - `theta`: `{1, 2, 3, 4, 5}`
  - `max_samples`: `{2, 3, 4, 5, 6}`
  - `gamma`: `[0.0, 3.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    theta = int(parameters["theta"])
    max_samples = int(parameters["max_samples"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    cue_diffs = a - b
    
    # Normalize validities to max 1 for stable gamma scaling across experiments
    v_norm = validities / np.max(validities)
    
    # Softmax for cue sampling probabilities (gamma acts as inverse temperature)
    z_p = gamma * v_norm
    z_p = z_p - np.max(z_p)
    e_p = np.exp(z_p)
    p = e_p / np.sum(e_p)
        
    # Monte Carlo approximation of the choice probabilities
    N = 1000
    samples = np.random.choice(len(validities), size=(N, max_samples), p=p)
    step_evidence = cue_diffs[samples]
    cum_evidence = np.cumsum(step_evidence, axis=1)
    
    hit_A = cum_evidence >= theta
    hit_B = cum_evidence <= -theta
    hit_any = hit_A | hit_B
    
    ever_hit = np.any(hit_any, axis=1)
    first_hit_idx = np.argmax(hit_any, axis=1)
    
    # Decisions for simulated runs that hit the threshold
    final_ev_hit = cum_evidence[np.arange(N), first_hit_idx]
    prob_A_hit = (final_ev_hit > 0).astype(float)
    prob_A_hit = np.where(final_ev_hit == 0, 0.5, prob_A_hit)
    
    # Decisions for simulated runs that exhausted max_samples without hitting threshold
    final_ev_no_hit = cum_evidence[:, -1]
    z = beta * final_ev_no_hit
    z = np.clip(z, -100, 100)  # Prevent overflow in exp
    prob_A_no_hit = 1.0 / (1.0 + np.exp(-z))
    
    # Aggregate overall probability of choosing A
    prob_A_overall = np.where(ever_hit, prob_A_hit, prob_A_no_hit)
    expected_p_A = np.mean(prob_A_overall)
    
    p_core = np.array([expected_p_A, 1.0 - expected_p_A])
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
