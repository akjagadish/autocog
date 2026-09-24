# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

**Rationale:** This theory replaces pi_1's one-reason stopping rule with full compensatory integration, so it does not wrongly commit to the first discriminating cue. In Experiment 1, that matters because the five lower-validity features can jointly outweigh the single high-validity feature, and in Experiment 2 signed evidence across the full feature vector must drive choice. The model also differs from plain Tallying by allowing feature weights to be validity-graded through kappa. The kappa prior is symmetric around zero and small, so the population-level expected weight vector is exactly the equal-weight vector, which captures the fact that both observed experiments sit close to ordinary Tallying while still giving the theory a substantive validity-weighting mechanism.

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `kappa`: `[-0.10, 0.10]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_6` — SURVIVED ✓

**Description:** Tally-first choice with a salience-gated and size-regularized validity tie-break. On every trial the unweighted signed tally of the feature differences is the default choice signal. When the tally is clear, choice follows it through a moderate softmax plus an explicit lapse. When the tally is tied or near-tied, validity is allowed to matter only if the task's validity distribution itself advertises a clear elite: either a small elite block of two to three maximally valid cues separated from the remaining cues, or a large validity gap combined with a bounded elite-block size. In those gated cases people compute a restricted sub-tally over the elite cues and choose its favored option stochastically. Larger elite blocks receive an additional lapse, preventing overconfident validity-following for wide top tiers. In large stimulus arrays, near-zero tallies are treated as low-confidence and are heavily regularized with extra choice noise, so deterministic regression blows-up do not occur.

**Rationale:** The previous pi_5 model failed most severely in Experiment 8 because its validity-free tally route was nearly deterministic, producing enormous z_d values. The proposed model fixes this with moderate tally gain and strong trial-level lapse, and in large stimulus arrays near-zero tallies receive especially heavy lapse because those choices are treated as low-confidence. Validity is not continuously mixed into every trial, avoiding pi_3's distortion of the Experiment 2 tally-only slope. The validity route is gated by static validity salience plus the current tally tie, not by history-window tie prevalence. The gate is calibrated to fire for the small elite block in Experiment 4 and the large-gap bounded elite block in Experiment 3, while remaining off in Experiments 1, 2, 5, 6, 7, and 8, whose human validity-following metrics are near chance. Adding extra lapse for larger elite blocks keeps Experiment 3's validity-following below the metric cap instead of saturating it.

**Parameters:**
  - `beta_tally`: `[2.0, 4.0]`
  - `epsilon_tally`: `[0.02, 0.08]`
  - `beta_near`: `[0.3, 0.8]`
  - `epsilon_near`: `[0.78, 0.97]`
  - `near_tally_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `[0.0, 0.5]`
  - `salience_gap_hi`: `[0.30, 0.45]`
  - `salience_gap_lo`: `[0.04, 0.06]`
  - `top_tol`: `{0.01}`
  - `top_size_small_min`: `{2}`
  - `top_size_small_max`: `{3}`
  - `top_size_block_max`: `{6}`
  - `beta_sub`: `[0.35, 0.50]`
  - `epsilon_sub`: `[0.08, 0.14]`
  - `epsilon_sub_size`: `[0.16, 0.20]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['salience_gap_hi'])
    gap_lo = float(parameters['salience_gap_lo'])
    size_min = int(parameters['top_size_small_min'])
    size_max = int(parameters['top_size_small_max'])
    block_max = int(parameters['top_size_block_max'])

    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    salience = 1.0 if (block_clear or small_elite) else 0.0

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        top_diff = diff[top_mask]
        sub_a = float(np.sum(top_diff > 0.0))
        sub_b = float(np.sum(top_diff < 0.0))
        sub_tally = sub_a - sub_b

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])
        eps_sub_size = float(parameters['epsilon_sub_size'])
        eps_sub_eff = eps_sub + eps_sub_size * max(0.0, float(top_size) - 3.0)
        eps_sub_eff = float(np.clip(eps_sub_eff, 0.0, 0.95))

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            probs = (1.0 - eps_sub_eff) * core + eps_sub_eff * np.array([0.5, 0.5])
        else:
            probs = np.array([0.5, 0.5])
        return probs

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.array([0.5, 0.5])
    return probs / total
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** People choose by a confidence-gated unweighted signed tally. On clear tallies, choice follows the tally through a moderate softmax plus a small lapse. When the tally is tied or nearly tied, people may switch to a restricted sub-tally only if the advertised cue validities themselves flag a small coherent elite (two to three cues separated from the rest, or a large validity gap with at most six top cues). Critically, that restricted block is not defined by validity alone: cues are re-ranked by a mixture of advertised validity and serial display position, where later cues receive exponentially greater recency weight. This allows late relevant cues to dominate early equally-valid cue blocks, producing reversals like Experiment 10 while preserving validity use in Experiments 3 and 4. Large-feature environments with small tallies are treated as low confidence and heavily regularized with near-chance lapse, preventing deterministic regression blow-ups.

**Rationale:** The default remains the signed tally, which already reproduces Experiments 1, 2, 5, and 7. The new contribution is a recency-weighted restricted sub-tally: when a tie occurs and the validity configuration advertises a small elite, the model selects a small block by re-ranking cues with validity * exp(rho * position). Setting rho high makes late blocks dominate early equally-valid blocks, which fixes the negative Experiment 10 contrast while leaving late high-validity blocks in Experiments 3 and 4 strongly preferred. The gate is deliberately narrow: small elite size is capped at 3 and requires at least a 0.05-0.10 gap, so Experiment 6's four-cue late block and Experiments 7 and 8 do not trigger validity use. Large-feature, small-tally trials receive epsilon_near of 0.80-0.97, keeping Experiments 8 and 9 near chance and avoiding the variance blow-ups of earlier theories. All logits are bounded, probabilities are clipped, and the use of moderate beta values regularizes per-subject regression estimates.

**Parameters:**
  - `beta_tally`: `[0.6, 1.0]`
  - `epsilon_tally`: `[0.02, 0.08]`
  - `beta_near`: `[0.3, 0.6]`
  - `epsilon_near`: `[0.80, 0.97]`
  - `near_tally_bound`: `{1.5}`
  - `n_low_conf`: `{15}`
  - `tau_tie`: `{0.5}`
  - `top_tol`: `{0.001}`
  - `small_size_min`: `{2}`
  - `small_size_max`: `{3}`
  - `gap_lo`: `[0.05, 0.10]`
  - `gap_hi`: `[0.30, 0.45]`
  - `block_max`: `{6}`
  - `rho`: `[0.60, 1.00]`
  - `beta_sub`: `[0.60, 1.00]`
  - `epsilon_sub`: `[0.05, 0.10]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))
    positions = np.arange(n_features, dtype=float)

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    def lapse_mix(core, eps):
        probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
        probs = np.clip(probs, 0.005, 0.995)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.array([0.5, 0.5])
        return probs / total

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['gap_hi'])
    gap_lo = float(parameters['gap_lo'])
    size_min = int(parameters['small_size_min'])
    size_max = int(parameters['small_size_max'])
    block_max = int(parameters['block_max'])

    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    salience = 1.0 if (small_elite or block_clear) else 0.0

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        rho = float(parameters['rho'])

        # Stable recency-weighted cue relevance. Subtracting the max position
        # prevents overflow while preserving the ranking and normalized weights.
        combined = np.asarray(validities, dtype=float) * np.exp(
            rho * (positions - positions.max())
        )

        order = np.argsort(-combined)
        selected = order[:top_size]
        if selected.size == 0:
            return np.array([0.5, 0.5])

        sel_weights = combined[selected]
        denom = float(np.sum(sel_weights))
        if denom <= 1e-12:
            weights = np.ones_like(sel_weights)
        else:
            weights = sel_weights / denom * float(selected.size)

        sub_tally = float(np.dot(weights, diff[selected]))

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            return lapse_mix(core, eps_sub)

        return np.array([0.5, 0.5])

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    return lapse_mix(core, eps)
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```
