# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Decision makers exhibit strategy variability, probabilistic alternating between a simple heuristic and a compensatory strategy. Specifically, subjects use a Strategy Mixture of Tallying (counting the number of positive cues for each option, ignoring cue validities) and Weighted Additive (WADD, computing a compensatory score based on log-odds of cue validities). This mixture allows the model to capture both fast, unweighted evidence accumulation and more deliberative, validity-weighted integration on different trials.

**Rationale:** The arbiter noted that pure TTB is too rigid and suggested exploring Tallying or a Strategy Mixture. Given that Tallying alone is a strong baseline (as seen in its leaderboard performance), a mixture model that probabilistically alternates between Tallying (simple cue counting) and WADD (compensatory validity-weighted integration) provides a flexible and structurally distinct hypothesis. This model allows for individual differences in strategy reliance via the w_mix parameter, capturing behavior that is sometimes heuristic and sometimes deliberative.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `w_mix`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    # WADD component
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # Tallying probabilities
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # WADD probabilities
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture
    p_core = w_mix * p_tally + (1.0 - w_mix) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Choice-Inertia Weighted Additive Model: Decision makers evaluate options using a compensatory Weighted Additive (WADD) strategy, but their final valuation is biased by their choice on the immediately preceding trial. This history-dependent inertia acts as an autoregressive bias on the chosen response side (e.g., a motor or spatial bias), allowing the model to capture sequential dependencies such as the tendency to repeat or alternate responses independently of the option features.

**Rationale:** Following the arbiter's feedback, this theory introduces a sequential dependence mechanism ('Choice Inertia') built on top of a compensatory Weighted Additive (WADD) base. By adding a history-dependent bias parameter (`inertia`) to the score of the option corresponding to the previously chosen side, the model can capture trial-to-trial autocorrelations commonly observed in human data (e.g., motor repetition or alternation biases). This mechanistic addition directly targets the failure of previous memoryless models to account for sequential effects in the experimental metric.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `inertia`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    # WADD base valuation
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    scores = np.dot(stim, weights)
    
    # Add choice inertia from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        inertia = float(parameters["inertia"])
        scores[last_choice] += inertia
        
    # Softmax and lapse
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Validity-Scaled Confidence and Choice Inertia: Decision makers use a fast-and-frugal lexicographic heuristic, comparing options sequentially on features ordered by their stated validities. They stop at the first feature that discriminates between the options. However, the confidence (and therefore determinism) of their choice scales with the validity of the cue that resolved the decision. The final choice is also subject to an autoregressive motor/spatial bias (inertia) from the immediately preceding trial.

**Rationale:** Addressing the critic's feedback: the previous iteration used a constant score increment (1.0) whenever a cue discriminated between the options, which ignores the fact that choices resolved by highly valid cues are typically executed with lower noise than those resolved by weak tie-breakers. By incrementing the score by the validity of the discriminating cue (e.g., `scores[0] += val[j]`), the softmax temperature naturally produces more deterministic choices for strong cues and more stochastic choices for weak ones, while preserving the strict non-compensatory sequential search of the Take-The-Best heuristic.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `inertia`: `[-5.0, 5.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    scores = np.zeros(2)
    
    # Take-The-Best heuristic: find the first discriminating cue
    for j in cue_order:
        if a[j] > b[j]:
            scores[0] += val[j]
            break
        if b[j] > a[j]:
            scores[1] += val[j]
            break
            
    # Add choice inertia from the previous trial
    if history and "response" in history and len(history["response"]) > 0:
        last_choice = int(history["response"][-1])
        inertia = float(parameters["inertia"])
        scores[last_choice] += inertia
        
    # Softmax and lapse
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_core)
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
