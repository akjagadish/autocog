# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_1` — SURVIVED ✓

**Description:** People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — KILLED ✗

**Description:** Tallying-Biased Parity (Strict Parity Dominance): Decision makers primarily rely on a strict Cue-Parity rule (preferring the option with an odd number of winning cues). However, instead of lapsing into purely uniform random noise, their secondary considerations consist of occasionally falling back to a standard validity-weighted tallying strategy. To ensure that tallying acts only as a weak bias and does not override parity in cases of option dominance, the fallback probability is strictly bounded to a low range.

**Rationale:** Following the latest critic feedback, we tightly bounded `epsilon` to [0.0, 0.3] and restricted `beta_tally` to [0.1, 10.0]. This minimal edit ensures that the validity-weighted tallying strategy functions strictly as a weak secondary lapse, preventing it from overriding the primary Cue-Parity rule in scenarios where one option clearly dominates the raw cue count (like in Experiment 6). This preserves the strong parity invariants while still allowing validities to exert the necessary monotonic pull.

**Parameters:**
  - `validities`: `validities`
  - `beta_parity`: `[0.1, 20.0]`
  - `beta_tally`: `[0.1, 10.0]`
  - `epsilon`: `[0.0, 0.3]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying-Biased Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Core Strategy: Strict Cue-Parity
    a_wins = int(np.sum(a > b))
    parity_winner = 0 if (a_wins % 2 == 1) else 1
    parity_scores = np.array([1.0, 0.0]) if parity_winner == 0 else np.array([0.0, 1.0])
    
    # Fallback Strategy: Validity-weighted Tallying
    val_a = np.sum(validities * (a > b))
    val_b = np.sum(validities * (b > a))
    tally_scores = np.array([val_a, val_b])
    
    beta_parity = float(parameters["beta_parity"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Parity probabilities via softmax
    z_p = beta_parity * (parity_scores - parity_scores.max())
    e_p = np.exp(z_p)
    p_parity = e_p / e_p.sum()
    
    # Tallying probabilities via softmax
    z_t = beta_tally * (tally_scores - tally_scores.max())
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Mixture of the two strategies
    return (1.0 - epsilon) * p_parity + epsilon * p_tally
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

**Description:** Top-K Parity: Decision makers rely on a strict Cue-Parity rule but apply it selectively to a stable subset of the most valid cues. Instead of using a floating validity threshold or rounding a fraction (which can arbitrarily drop a single cue and flip the parity sum in 6- and 12-cue tasks), they evaluate the top ceil(k_frac * N) cues. This ensures that for tasks with 5 or 6 cues, all cues are evaluated, preserving multi-cue parity effects without collapsing to random parity flips.

**Rationale:** Following the critic's advice, I built on the successful iter 5 base but replaced `int(np.round(k_frac * n_cues))` with `math.ceil(k_frac * n_cues)`. In experiments with 5 or 6 cues, a `k_frac` of 0.9 previously rounded to 4 and 5 cues respectively, dropping exactly 1 cue. This single dropped cue artificially flipped the parity sum, causing massive deviations in Exps 5 and 8. By using `math.ceil`, `math.ceil(0.9 * 5) = 5` and `math.ceil(0.9 * 6) = 6`, ensuring all cues are evaluated in these tasks and preventing the artificial parity flips, while still allowing a tiny fraction of cues to be dropped in much larger tasks.

**Parameters:**
  - `validities`: `validities`
  - `k_frac`: `[0.9, 1.0]`
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    import math
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Top-K Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    k_frac = float(parameters["k_frac"])
    
    n_cues = len(validities)
    # Use math.ceil to avoid arbitrarily dropping exactly 1 cue in 5- or 6-cue tasks
    k = max(1, math.ceil(k_frac * n_cues))
    
    # Get indices of the top K validities
    # np.argsort sorts ascending, so we take the last k elements
    top_k_indices = np.argsort(validities)[-k:]
    
    a_filtered = a[top_k_indices]
    b_filtered = b[top_k_indices]
    
    # Strict Cue-Parity on the filtered subset of cues
    a_wins = int(np.sum(a_filtered > b_filtered))
    winner = 0 if (a_wins % 2 == 1) else 1
    
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary score
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
