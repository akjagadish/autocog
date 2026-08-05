# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5` — SURVIVED ✓

**Description:** Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.

**Rationale:** Following the arbiter's suggestion, this model instantiates the 'Extreme Spatial Bias / Key Perseveration' theory. A pure random guessing model (pi_4) fails to capture the variance structure of the data, particularly in Experiment 6 where it yields a squared deviation of ~0.0 instead of 0.25. By assuming subjects are individually deterministic but perfectly split at the population level (50% Always-A, 50% Always-B), we perfectly recover the 0.5 accuracy metrics (since the experimental designs are balanced, always picking one option yields exactly 50% agreement with any balanced strategy like TTB or Tallying), the 0.0 differences in conditional probabilities, and the exact 0.25 squared deviation in Experiment 6 (since P(A) is 1.0 or 0.0 for each subject, yielding (1-0.5)^2 = 0.25 or (0-0.5)^2 = 0.25).

**Parameters:**
  - `preferred_option`: `{0, 1}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    # The subject adopts a deterministic response strategy, always choosing the same option.
    pref = int(parameters["preferred_option"])
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_6` — KILLED ✗

**Description:** First-Trial Inertia / Rational Disengagement: Subjects do not possess an innate spatial bias prior to the experiment. Instead, they actively evaluate the very first trial using a simple heuristic (like Tallying or Take-The-Best) to make an initial choice. However, due to the lack of trial-by-trial feedback and the repetitive nature of the multi-attribute stimuli, they rationally disengage after this first trial to minimize cognitive effort. For all subsequent trials, they exhibit 'first-trial inertia' by simply repeating their initial motor action. This produces 100% perseveration at the individual level, but grounds the origin of the bias in an initial stimulus-driven evaluation rather than an arbitrary trait.

**Rationale:** Following the arbiter's recommendation, this theory replaces the a priori spatial bias assumption of Theory 1 with a 'First-Trial Inertia' mechanism. Subjects evaluate the first trial using a simple heuristic (parameterized as either Tallying or Take-The-Best) to make a reasoned initial choice. Because the task provides no feedback and features are complex, they subsequently disengage and repeat this initial choice for the rest of the block. Since trial order is randomized, the first trial varies across subjects, leading to a natural ~50/50 split in the population between 'Always-A' and 'Always-B' responders. This maintains the perfect 100% perseveration fit observed in the metrics while providing a more plausible, stimulus-grounded mechanistic origin.

**Parameters:**
  - `heuristic_type`: `{0, 1}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # If this is the first trial, evaluate using a simple heuristic
    if len(history.get("response", [])) == 0:
        state = np.asarray(state, dtype=float)
        a, b = state[0], state[1]
        
        heuristic_type = int(parameters["heuristic_type"])
        score_a, score_b = 0, 0
        
        if heuristic_type == 0:
            # Tallying (equal weights)
            score_a = np.sum(a)
            score_b = np.sum(b)
        else:
            # Take-The-Best (lexicographic)
            val = np.asarray(parameters["validities"], dtype=float)
            order = np.argsort(val)[::-1]
            for idx in order:
                if a[idx] > b[idx]:
                    score_a = 1
                    break
                elif b[idx] > a[idx]:
                    score_b = 1
                    break
                    
        if score_a > score_b:
            return np.array([1.0, 0.0])
        elif score_b > score_a:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])
            
    else:
        # Rational disengagement: repeat the initial motor action
        first_resp = history["response"][0]
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_7` → slot 2 (via `new_theory`)

**Description:** Random First Choice Inertia: Subjects face task ambiguity or lack motivation at the onset of the experiment. Instead of possessing an innate spatial bias or evaluating the first trial's features, they make a completely random guess on the very first trial. To minimize cognitive effort on all subsequent trials, they deterministically repeat this initial random motor action. This leads to 100% individual-level perseveration that is entirely uncorrelated with any stimulus features.

**Rationale:** Following the arbiter's guidance, this theory replaces the innate spatial trait (Theory 1) and the first-trial heuristic evaluation (Theory 2) with a 'Random First Choice Inertia' mechanism. The subject guesses randomly on the first trial due to task ambiguity, and then repetitively executes that same motor action for the rest of the block to minimize cognitive effort. This replicates the 100% perseveration seen in the empirical data while correctly predicting that the choice is uncorrelated with the first trial's actual stimulus features.

**Parameters:**
  - `dummy`: `{0}`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # A dummy parameter to satisfy the interface requirement
    _ = parameters.get("dummy")
    
    # On the very first trial, subjects guess randomly (50/50)
    if len(history.get("response", [])) == 0:
        return np.array([0.5, 0.5])
    else:
        # On all subsequent trials, they deterministically repeat the first trial's motor action
        first_resp = history["response"][0]
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```
