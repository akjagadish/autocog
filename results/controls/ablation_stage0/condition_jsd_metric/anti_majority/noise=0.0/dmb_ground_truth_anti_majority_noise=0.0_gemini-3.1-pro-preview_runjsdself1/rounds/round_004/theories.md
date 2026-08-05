# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_6` — KILLED ✗

**Description:** Probabilistic Evidence Accumulation with Baseline Evidence: Decision-makers sample cues probabilistically proportional to their log-odds validities. Each sampled cue casts a vote for the option it favors, and decision-makers also possess a baseline level of prior evidence for both options. The process continues until a relative evidence threshold is reached, producing compensatory behavior and graded choice probabilities based on the relative accumulated evidence. The inclusion of baseline evidence smooths out extreme probability predictions and naturally handles cases where no cues favor an option.

**Rationale:** Building on the accepted base from iteration 1, we retain the log-odds validities and the power-law choice rule. Following the critic's advice, we introduce a new baseline evidence parameter `c` added to both options' accumulated evidence before applying the power-law rule. This completely eliminates the need for hardcoded zero-evidence branches and smoothly regularizes extreme probability predictions when one option has little to no evidence, increasing the robustness of the Probabilistic Evidence Accumulation model.

**Parameters:**
  - `theta`: `[0.1, 20.0]`
  - `c`: `[0.01, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Log-odds weights for proportional sampling probabilities
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # Total sampling weight favoring each option
    v_a = np.sum(weights * (a > b))
    v_b = np.sum(weights * (b > a))
    
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    c = float(parameters["c"])
    
    # Add baseline constant to smooth out extreme ratios and avoid zero-evidence edge cases
    v_a_eff = v_a + c
    v_b_eff = v_b + c
    
    # Probability of reaching the relative threshold theta first via power-law rule
    log_p_a_unnorm = theta * np.log(v_a_eff)
    log_p_b_unnorm = theta * np.log(v_b_eff)
    max_log = max(log_p_a_unnorm, log_p_b_unnorm)
    exp_a = np.exp(log_p_a_unnorm - max_log)
    exp_b = np.exp(log_p_b_unnorm - max_log)
    p_a = exp_a / (exp_a + exp_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Decision-makers use a bounded evidence accumulation process that integrates sequential dependencies. They default to a non-compensatory heuristic (Take The Best) but smoothly transition to a compensatory strategy (Weighted Additive) when opposing evidence exceeds a soft threshold. Furthermore, choices are subject to sequential dependencies: the baseline preference for an option is dynamically adjusted based on the previous trial's choice, capturing choice inertia or auto-correlation. This moving baseline influences the final decision probabilities alongside the strategy-specific evidence.

**Rationale:** The new theory directly implements the arbiter's suggestion to incorporate sequential dependencies by adding a choice inertia mechanism alongside the bounded evidence accumulation process (soft compensatory check). The evaluation metric computes JSD conditioned on the previous trial's response, implying that the history of choices influences the current choice probabilities. By adding a dynamic baseline shift (`inertia`) to the logits of the previously chosen option, the model naturally produces auto-correlation in choices, improving its ability to match the conditional JSD metrics across experiments.

**Parameters:**
  - `threshold`: `[0.0, 10.0]`
  - `k`: `[0.1, 10.0]`
  - `beta_ttb`: `[0.1, 20.0]`
  - `beta_wadd`: `[0.1, 20.0]`
  - `inertia`: `[-5.0, 5.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import scipy.special
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Calculate log-odds weights for WADD and opposing evidence
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # --- Take The Best (TTB) Phase ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is not None:
        # Calculate opposing evidence: sum of weights where the TTB loser beats the TTB winner
        if winner_ttb == 0:
            opposing_evidence = np.sum(weights * (b > a))
        else:
            opposing_evidence = np.sum(weights * (a > b))
            
        k = float(parameters["k"])
        threshold = float(parameters["threshold"])
        # Soft transition: probability of using WADD is a logistic function of opposing evidence
        p_wadd = float(scipy.special.expit(k * (opposing_evidence - threshold)))
    else:
        # If no cue discriminates, default to WADD (which will tie)
        p_wadd = 1.0
        
    # --- Sequential Dependency (Inertia) ---
    inertia = float(parameters["inertia"])
    z_inertia = np.array([0.0, 0.0])
    if history and "response" in history and len(history["response"]) > 0:
        last_resp = int(history["response"][-1])
        z_inertia[last_resp] = inertia
        
    # --- Decision Formulation ---
    # WADD probabilities with inertia
    scores_wadd = np.array([np.sum(weights * a), np.sum(weights * b)])
    beta_wadd = float(parameters["beta_wadd"])
    z_w = beta_wadd * scores_wadd + z_inertia
    z_w = z_w - np.max(z_w)
    e_w = np.exp(z_w)
    p_core_wadd = e_w / np.sum(e_w)
    
    # TTB probabilities with inertia
    if winner_ttb is not None:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
    beta_ttb = float(parameters["beta_ttb"])
    z_t = beta_ttb * scores_ttb + z_inertia
    z_t = z_t - np.max(z_t)
    e_t = np.exp(z_t)
    p_core_ttb = e_t / np.sum(e_t)
    
    # Mix strategies based on soft threshold
    p_core = p_wadd * p_core_wadd + (1.0 - p_wadd) * p_core_ttb
    
    # Apply lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_7` → slot 1 (via `new_theory`)

**Description:** Decision-makers evaluate options using a Leaky Competing Accumulator (LCA) process with dynamic attention switching. Attention deterministically shifts between features in decreasing order of their validity over a sequence of time steps. At each step, the attended feature provides evidence that updates a preference state for each option. These preference states are subject to leakage and lateral inhibition. The starting state of the accumulators is biased by the previous trial's choice (inertia). By limiting the maximum number of accumulation steps, the model ensures that the initial sequential bias is not washed out by over-accumulation, allowing it to better capture sequential dependencies.

**Rationale:** Following the critic's advice on the rejected Iteration 3, I reverted to strictly positive inertia (expanding the upper bound slightly to 3.0) and reduced the upper bound of `num_steps` from 20.0 to 10.0. This prevents the sequential inertia effect from being washed out by over-accumulation in the leaky process, addressing the under-prediction of deviations in Experiments 4, 6, and 8 without introducing the harmful negative inertia.

**Parameters:**
  - `lambda_leak`: `[0.0, 0.5]`
  - `gamma_inhibition`: `[0.0, 0.5]`
  - `inertia`: `[0.0, 3.0]`
  - `num_steps`: `[1.0, 10.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    v = np.clip(val, 0.5001, 0.9999)
    weights = np.log(v / (1.0 - v))
    
    # Order features by descending validity for dynamic attention switching
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    lambda_leak = float(parameters["lambda_leak"])
    gamma_inhibition = float(parameters["gamma_inhibition"])
    inertia = float(parameters["inertia"])
    num_steps = int(float(parameters["num_steps"]))
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Starting point bias based on previous choice (Inertia)
    if history and len(history.get("response", [])) > 0:
        prev_resp = history["response"][-1]
        x_A = inertia if prev_resp == 0 else 0.0
        x_B = inertia if prev_resp == 1 else 0.0
    else:
        x_A = 0.0
        x_B = 0.0
        
    # LCA accumulation with dynamic attention
    for t in range(num_steps):
        idx = cue_order[t % n_features]
        w = weights[idx]
        
        # Input from the currently attended feature
        I_A = a[idx] * w
        I_B = b[idx] * w
        
        # Update accumulators with leak and lateral inhibition
        new_x_A = max(0.0, x_A + I_A - lambda_leak * x_A - gamma_inhibition * x_B)
        new_x_B = max(0.0, x_B + I_B - lambda_leak * x_B - gamma_inhibition * x_A)
        
        x_A = new_x_A
        x_B = new_x_B
        
    scores = np.array([x_A, x_B])
    
    # Softmax choice probability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```
