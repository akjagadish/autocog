# Round 10 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** Conflict-Driven Strategy Selection: Decision-makers adaptively select their decision strategy based on the dissimilarity of the options' total cue counts. When options are highly dissimilar in the number of positive cues (indicating high conflict or complexity), individuals abandon compensatory processing and fall back entirely on a simple non-compensatory heuristic (Take The Best). Conversely, when the total cue counts are similar, individuals attempt to integrate all available information using a compensatory strategy (Tallying). This is modeled as a probabilistic mixture of TTB and Tallying, where the probability of using TTB scales directly with the absolute difference in total cue counts.

**Rationale:** Following the critic's advice, we revert to the exact Iteration 2 mechanism but adjust the parameter ranges to find a better global compromise. We constrain `beta_ttb` and `beta_tally` to `[0.1, 10.0]` to ensure both strategies maintain reasonable stochasticity and do not collapse into pure determinism. We also increase the upper bound of `epsilon` to `0.5` to allow for a higher baseline lapse rate. This gentle retuning preserves the stable core formulation of Iteration 2 while preventing the destructive extremes observed in Iteration 6.

**Parameters:**
  - `beta_ttb`: `[0.1, 10.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `gamma`: `[0.0, 10.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB Probabilities
    z_ttb = beta_ttb * scores_ttb
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta_tally * scores_tally
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally_dist = e_tally / e_tally.sum()
    
    # Conflict-Driven Weight
    gamma = float(parameters["gamma"])
    
    # Dissimilarity in total cue counts
    diff_cues = abs(np.sum(a) - np.sum(b))
    
    # Probability of using TTB increases linearly with diff_cues
    w_ttb = min(1.0, gamma * diff_cues / max(1.0, float(n_features)))
    
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb_dist + (1.0 - w_ttb) * p_tally_dist
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_12` — KILLED ✗

**Description:** Context-Dependent Strategy Selection with Reverse TTB (Bottom-Up Heuristic): Decision-makers adapt their decision strategy based on the overall equality of evidence. When the total number of positive cues is unequal, they rely on a mixture of Tallying and standard Take-The-Best (evaluating from most to least valid). However, when options present a tied sum of positive cues, it triggers a 'Bottom-Up' or 'Reverse TTB' heuristic. In this mode, subjects evaluate cues starting from the least valid to the most valid, resolving the tie by favoring the option that possesses discriminating lower-validity cues. This naturally explains the paradoxical avoidance of the highest-validity cue in tied-sum scenarios, as the highest-validity cue is evaluated last and is thus overridden by differences in the lower-validity cues.

**Rationale:** Following the arbiter's insight, this model directly implements a 'Bottom-Up' strategy (Reverse TTB) to account for the paradoxical avoidance of the highest-validity cue in tied-sum trials. When the total sum of cues is equal, a context-dependent mechanism heavily boosts the probability of using Reverse TTB, causing the decision-maker to resolve the conflict by looking at the lowest-validity cues first. This correctly predicts that the option with the highest-validity cue will frequently be rejected in these specific conflict scenarios, addressing the mechanistic failures of purely compensatory or top-down models.

**Parameters:**
  - `w_ttb`: `[0.0, 5.0]`
  - `w_tally`: `[0.0, 5.0]`
  - `w_bottom_up`: `[0.0, 5.0]`
  - `gamma_tie`: `[0.0, 10.0]`
  - `gamma_diff`: `[0.0, 10.0]`
  - `beta`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.2]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Orderings for Top-Down (TTB) and Bottom-Up (Reverse TTB)
    cue_order_top_down = np.argsort(-val, kind="stable").tolist()
    cue_order_bottom_up = np.argsort(val, kind="stable").tolist()
    
    # 1. Standard TTB (Top-Down)
    winner_ttb = 0.5
    for j in cue_order_top_down:
        if a[j] > b[j]:
            winner_ttb = 1.0
            break
        elif b[j] > a[j]:
            winner_ttb = 0.0
            break
            
    # 2. Bottom-Up TTB (Reverse TTB)
    winner_bottom_up = 0.5
    for j in cue_order_bottom_up:
        if a[j] > b[j]:
            winner_bottom_up = 1.0
            break
        elif b[j] > a[j]:
            winner_bottom_up = 0.0
            break
            
    # 3. Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        winner_tally = 1.0
    elif b_wins > a_wins:
        winner_tally = 0.0
    else:
        winner_tally = 0.5
        
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Strategy mixture logits
    z_strats = np.array([
        float(parameters["w_ttb"]),
        float(parameters["w_tally"]),
        float(parameters["w_bottom_up"])
    ])
    
    # Context-dependent boosts
    if sum_a == sum_b:
        z_strats[2] += float(parameters["gamma_tie"])  # Boost Bottom-Up when sums are tied
    else:
        z_strats[0] += float(parameters["gamma_diff"]) # Boost Standard TTB when sums differ
        
    e_strats = np.exp(z_strats - np.max(z_strats))
    p_strats = e_strats / np.sum(e_strats)
    
    # Expected probability of choosing Option A
    p_a = (p_strats[0] * winner_ttb + 
           p_strats[1] * winner_tally + 
           p_strats[2] * winner_bottom_up)
    
    # Convert to choice probabilities with temperature
    beta = float(parameters["beta"])
    z = beta * np.array([p_a, 1.0 - p_a])
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_13` → slot 2 (via `new_theory`)

**Description:** Weighted Additive with Power-Function Validity Scaling (WADD-PWR): Decision-makers integrate all available cues but weight them non-linearly using a power function of their stated validities. A single parameter, gamma, controls the weighting scheme: positive values approximate Take-The-Best by heavily favoring the most valid cues, zero yields Tallying by weighting all cues equally, and negative values approximate Reverse TTB by favoring less valid cues. The power function naturally anchors a validity of 1.0 to a weight of 1.0, providing built-in numerical stability across the full spectrum of strategies without requiring artificial normalization.

**Rationale:** Following the critic's advice, we replace the exponential weighting scheme with a power-function parameterization `w = val ** gamma` while preserving the expanded range for gamma `[-20.0, 20.0]`. Because validities are bounded in [0.5, 1.0], raising them to a power naturally anchors the maximum possible validity to 1.0 (since 1.0 ** gamma = 1.0), inherently solving the absolute scaling issues that plagued the exponential normalizations. When gamma > 0, it approximates TTB; when gamma == 0, it yields Tallying; and when gamma < 0, lower validities are weighted more heavily, naturally capturing Reverse TTB. This provides a clean, continuous WADD-NL integration mechanism.

**Parameters:**
  - `gamma`: `[-20.0, 20.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```
