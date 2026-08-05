# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_6` — SURVIVED ✓

**Description:** Probabilistic Cue Integration with Attention Lapses and Choice Inertia: Subjects integrate evidence from cues by probabilistically attending to them based on validity. To account for behavioral overdispersion and sequence effects, the final choice probability is a mixture of the evidence-based decision, random lapses, and explicit choice inertia (stickiness) from the previous trial.

**Rationale:** Added a `stickiness` parameter to explicitly model choice inertia at the probability level, as recommended by the critic. Because the Monte Carlo integration smooths out the probabilities, directly injecting trial-by-trial auto-correlation via `stickiness` forces the output probabilities to become more polarized according to the subject's previous choices. This will naturally inflate the variance of simulated choice proportions and better capture the overdispersion (high JSD) seen in human data. I ensured that the mixture weights are properly normalized so that the output remains a valid probability distribution.

**Parameters:**
  - `attention_base`: `[0.1, 1.0]`
  - `attention_gamma`: `[-5.0, 10.0]`
  - `history_bias`: `[-10.0, 10.0]`
  - `temperature`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `stickiness`: `[0.0, 0.99]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    attention_base = float(parameters["attention_base"])
    attention_gamma = float(parameters["attention_gamma"])
    history_bias = float(parameters["history_bias"])
    temperature = float(parameters["temperature"])
    epsilon = float(parameters["epsilon"])
    stickiness = float(parameters["stickiness"])
    
    # Probability of attending to each cue depends on base rate and its validity
    p_attend = np.clip(attention_base * (val ** attention_gamma), 0.0, 1.0)
    
    prev_resp = None
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
    
    bias = 0.0
    p_prev = np.array([0.5, 0.5])
    if prev_resp == 0:
        bias = history_bias
        p_prev = np.array([1.0, 0.0])
    elif prev_resp == 1:
        bias = -history_bias
        p_prev = np.array([0.0, 1.0])
        
    a, b = stim[0], stim[1]
    n_features = len(val)
    
    # Monte Carlo simulation of probabilistic attention
    n_sims = 2000
    attend_mask = np.random.rand(n_sims, n_features) < p_attend
    evidence = weights * (a - b)
    
    # Sum evidence only for attended cues and add history bias
    total_evidence = np.sum(attend_mask * evidence, axis=1) + bias
    
    # Softmax conversion to probability of choosing A
    z = total_evidence / temperature
    z = np.clip(z, -100, 100)
    p_A_sims = 1.0 / (1.0 + np.exp(-z))
    
    p_A = np.mean(p_A_sims)
    p_B = 1.0 - p_A
    
    p_core = np.array([p_A, p_B])
    
    # Normalize weights to ensure valid probabilities even if stickiness + epsilon > 1
    w_core = max(0.0, 1.0 - stickiness - epsilon)
    total_w = w_core + stickiness + epsilon
    
    return (w_core * p_core + stickiness * p_prev + epsilon * np.array([0.5, 0.5])) / total_w
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Noisy Evidence Accumulation with History Bias and Leakage: Decision making is a noisy sequential sampling process where evidence from cues is accumulated in order of validity. The accumulation is subject to moment-to-moment noise and evidence decay (leakage), producing trial-to-trial variability and primacy/recency effects. The starting point of evidence accumulation is biased by the previous choice, naturally capturing sequence-aware divergence. Wide individual differences in noise, threshold, leak, and history bias account for the high variance in behavior across subjects.

**Rationale:** Following the critic's feedback, we added a `leak` parameter to represent evidence decay during the sequential accumulation process. By decaying accumulated evidence at each step before adding new evidence (`accumulated = accumulated * (1.0 - leak) + step`), the model naturally captures primacy/recency effects in cue processing. This allows for highly idiosyncratic choices without needing to artificially scale weights or restrict the general lapse rate. We also narrowed the `noise_std` upper bound to 5.0 to ensure that noise does not completely overwhelm the signal on every trial.

**Parameters:**
  - `threshold`: `[0.01, 10.0]`
  - `noise_std`: `[0.1, 5.0]`
  - `history_bias`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `leak`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    noise_std = float(parameters["noise_std"])
    history_bias = float(parameters["history_bias"])
    epsilon = float(parameters["epsilon"])
    leak = float(parameters["leak"])
    
    # Determine previous response to set accumulation starting bias
    prev_resp = None
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
    
    bias = 0.0
    if prev_resp == 0:
        bias = history_bias
    elif prev_resp == 1:
        bias = -history_bias
        
    a, b = stim[0], stim[1]
    
    # Vectorized Monte Carlo simulation of the noisy accumulation process
    n_sims = 1000
    accumulated = np.full(n_sims, bias)
    decided = np.zeros(n_sims, dtype=bool)
    choices = np.zeros(n_sims, dtype=int)
    
    for j in cue_order:
        mu = weights[j] * (a[j] - b[j])
        # Moment-to-moment noise in evidence accumulation
        step = np.random.normal(mu, noise_std, size=n_sims)
        
        # Apply leakage to previously accumulated evidence before adding the new step
        accumulated = np.where(decided, accumulated, accumulated * (1.0 - leak) + step)
        
        hit_A = (accumulated >= threshold) & ~decided
        hit_B = (accumulated <= -threshold) & ~decided
        
        choices[hit_A] = 0
        choices[hit_B] = 1
        
        decided = decided | hit_A | hit_B
        if np.all(decided):
            break
            
    # For simulations that didn't cross the threshold, decide based on final accumulated evidence
    if not np.all(decided):
        undecided = ~decided
        undecided_A = undecided & (accumulated > 0)
        undecided_B = undecided & (accumulated < 0)
        undecided_tie = undecided & (accumulated == 0)
        
        choices[undecided_A] = 0
        choices[undecided_B] = 1
        
        ties = np.sum(undecided_tie)
        if ties > 0:
            choices[undecided_tie] = np.random.choice([0, 1], size=ties)
            
    p_A = np.mean(choices == 0)
    p_B = 1.0 - p_A
    
    p_core = np.array([p_A, p_B])
    n_opts = len(p_core)
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Mixture of Heuristics: Subjects possess a repertoire of distinct decision heuristics—specifically, Take-The-Best (lexicographic, strictly ordered by validity with early stopping) and Tallying (simple unweighted counting of positive features). On any given trial, a subject probabilistically selects one of these strategies. This mixture of a strictly non-compensatory strategy and a completely flat compensatory strategy, combined with baseline random lapses and history bias (stickiness to the previous choice), captures the complex, overdispersed choice patterns observed across different cue configurations.

**Rationale:** Following the arbiter's guidance, this theory abandons continuous cue integration in favor of a probabilistic mixture of two discrete heuristics: Take-The-Best (non-compensatory) and Tallying (flat compensatory). By mixing these deterministic strategies at the individual level (via `p_ttb`), alongside random lapses (`epsilon`) and choice inertia (`stickiness`), the model naturally generates the complex, overdispersed choice behaviors without needing a temperature parameter. This satisfies the arbiter's mechanistic requirements while remaining highly parsimonious.

**Parameters:**
  - `p_ttb`: `[0.0, 1.0]`
  - `epsilon`: `[0.0, 1.0]`
  - `stickiness`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) heuristic
    cue_order = np.argsort(-val, kind="stable").tolist()
    p_ttb_A = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb_A = 1.0
            break
        elif b[j] > a[j]:
            p_ttb_A = 0.0
            break
            
    # Tallying heuristic
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tly_A = 1.0
    elif b_wins > a_wins:
        p_tly_A = 0.0
    else:
        p_tly_A = 0.5
        
    # Mixture of heuristics
    p_ttb_weight = float(parameters["p_ttb"])
    p_core_A = p_ttb_weight * p_ttb_A + (1.0 - p_ttb_weight) * p_tly_A
    p_core = np.array([p_core_A, 1.0 - p_core_A])
    
    # History bias (stickiness)
    prev_resp = None
    if history and "response" in history and len(history["response"]) > 0:
        prev_resp = history["response"][-1]
        
    p_prev = np.array([0.5, 0.5])
    if prev_resp == 0:
        p_prev = np.array([1.0, 0.0])
    elif prev_resp == 1:
        p_prev = np.array([0.0, 1.0])
        
    epsilon = float(parameters["epsilon"])
    stickiness = float(parameters["stickiness"])
    
    # Normalize weights to ensure valid probabilities
    w_core = max(0.0, 1.0 - stickiness - epsilon)
    total_w = w_core + stickiness + epsilon
    
    p_final = (w_core * p_core + stickiness * p_prev + epsilon * np.array([0.5, 0.5])) / total_w
    return p_final
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
