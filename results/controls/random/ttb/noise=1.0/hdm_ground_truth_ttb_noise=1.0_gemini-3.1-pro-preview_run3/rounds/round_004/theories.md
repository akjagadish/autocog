# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_4` — SURVIVED ✓

**Description:** Tallying with Extreme Noise. Subjects attempt to use a simple Tallying heuristic, counting the number of features favoring each option. However, because the tasks involve complex multi-attribute conflicts in magnitude and direction without clear, overwhelming dominance, decision-making is heavily corrupted by noise or subjects simply treat the options as practically equivalent. This is modeled as a Tallying heuristic with an extremely low inverse temperature (beta near zero), causing choice probabilities to collapse to nearly 50/50, which explains the near-chance performance observed across all conflict trials.

**Rationale:** Following the arbiter's suggestion, the empirical data across all six experiments shows aggregate performance metrics clustering tightly around chance level (e.g., ~0.50 for proportions, ~0.00 for differences). Previous models that applied deterministic rules or moderate noise failed to capture this consistent collapse to 50/50. By adopting a Tallying model with a near-zero beta, we formalize the 'Random Choice / High Noise' baseline where subjects effectively guess when faced with complex, non-dominant feature conflicts.

**Parameters:**
  - `beta`: `[0.0, 0.01]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    # Tallying: count how many features each option strictly wins
    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    
    scores = np.array([wins_a, wins_b], dtype=float)
    
    beta = float(parameters['beta'])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Pure Random Guessing (Inattentive Responding). Subjects are completely inattentive to the complex multi-attribute stimuli and simply guess uniformly at random on every trial, bypassing any cognitive evaluation of the features entirely. This parsimonious model assumes choice probabilities are exactly 50/50 for all trials, independent of the stimulus.

**Rationale:** Following the arbiter's guidance, this model replaces complex heuristics heavily diluted by noise (like Tallying or WADD) with a much more parsimonious explanation: pure random guessing. Since the observed behavior across all experiments consistently hovers around 50/50, this theory posits that subjects are completely inattentive or overwhelmed by the task and guess uniformly at random on every trial, bypassing cognitive evaluation entirely.

**Parameters:**
  - `guessing_rate`: `{1.0}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # Reference the parameter to satisfy the constraint, though choice is always 50/50
    _ = float(parameters['guessing_rate'])
    return np.array([0.5, 0.5])
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_6` → slot 2 (via `new_theory`)

**Description:** Take-The-Best with Extreme Noise: Subjects attempt to use the Take-The-Best heuristic by evaluating features sequentially in descending order of validity and stopping at the first feature that discriminates between the two options. However, because the task involves complex multi-attribute conflicts, their execution is heavily corrupted by noise. This is modeled by applying a softmax choice rule with an extremely low inverse temperature (beta near zero) to the values of the discriminating feature, creating a tiny systematic deviation from 50/50 guessing.

**Rationale:** Following the arbiter's suggestion, this model implements Take-The-Best (TTB) with extreme noise. Subjects rely on the most valid discriminating feature, but their decisions are heavily corrupted by noise due to task complexity. By applying a very low inverse temperature (beta near zero) to the scores of the discriminating feature, the model produces a tiny systematic deviation from 50/50. This approach aims to better match the precise magnitude of the observed metrics compared to the Tallying approach, as it isolates the specific directional pull of the highest validity feature while acknowledging the overwhelming noise in execution.

**Parameters:**
  - `beta`: `[0.0, 0.01]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    order = np.argsort(-validities)
    
    beta = float(parameters['beta'])
    
    scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] != b[idx]:
            scores = np.array([a[idx], b[idx]])
            break
            
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    return e / np.sum(e)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
