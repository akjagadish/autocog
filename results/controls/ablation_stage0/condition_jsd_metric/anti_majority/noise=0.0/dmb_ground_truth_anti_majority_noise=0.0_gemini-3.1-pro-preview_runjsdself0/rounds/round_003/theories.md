# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** People make decisions by sequentially accumulating evidence from cues in order of their validity. Evidence is weighted by the log-odds of each cue's validity. The accumulation stops and a choice is made as soon as the absolute difference in accumulated evidence between the two options exceeds an internal threshold. A low threshold leads to frugal 'Take The Best' behavior, while a high threshold leads to compensatory 'Weighted Additive' behavior. This unified mechanism accounts for varying degrees of cue integration depending on task context and individual differences.

**Rationale:** Following the arbiter's recommendation, this theory implements a sequential evidence accumulation model with a stopping threshold. By consulting cues in order of validity and accumulating log-odds weights, the model naturally interpolates between Take The Best (low threshold, stopping at the first discriminating cue) and Weighted Additive (high threshold, integrating all cues). This unifies the previously distinct heuristic and compensatory models into a single parameterized framework capable of capturing the diverse, context-dependent human behaviors observed across the experiments.

**Parameters:**
  - `threshold`: `[0.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Monotonic transformation of validities to log-odds
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Cues are consulted in descending order of validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    a, b = stim[0], stim[1]
    score_a = 0.0
    score_b = 0.0
    
    # Sequential evidence accumulation
    for j in cue_order:
        score_a += weights[j] * a[j]
        score_b += weights[j] * b[j]
        diff = abs(score_a - score_b)
        # Stop if the accumulated difference exceeds the threshold
        # (and ensure we don't stop on a zero difference if threshold is 0)
        if diff >= threshold and diff > 1e-6:
            break

    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
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


### slot 2 — `pi_5` — SURVIVED ✓

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

### `pi_6` → slot 1 (via `new_theory`)

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
