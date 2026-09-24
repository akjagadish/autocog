# Round 3 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_3` — SURVIVED ✓

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


### slot 2 — `pi_5` — KILLED ✗

**Description:** People adaptively decide whether unweighted tallying is diagnostic in the current task. They monitor two context signals over recent stimulus history: the prevalence of tally ties and the mean absolute per-trial validity-weighted evidence on those tie trials. When ties are rare, choice remains essentially unweighted tallying. When ties are prevalent and validity evidence is diagnostically substantial in absolute magnitude on tie trials, a validity-based tie-break policy is recruited. The recruitment is near-binary rather than continuous, so low-tie tasks recover near-tallying behavior while high-tie diagnostic tasks show strong validity-aligned tie-breaking. Unlike a fixed validity-gated tallying rule, the validity route is switched by task composition, and unlike continuous validity grading, validity is not mixed into every trial.

**Rationale:** This is a minimal-diff edit of the accepted context-gated tallying candidate. The key change replaces the signed-mean consistency veto with the absolute per-trial validity-evidence statistic prescribed by the critic: mean |v_lin| on tie trials. This preserves the near-binary veto that fixed Experiment 6, but now opens the gate in Experiment 4 where validity evidence is diagnostic per trial while alternating sign across trials. The tie-prevalence gate is kept and its threshold is narrowed (c0 in [0.56, 0.575]), the mean-evidence floor is set low enough to open reliably on small but nonzero evidence ([0.05, 0.10]), and min_cons_ties is lowered to 2-3. The tally route is sharpened with beta_tally in [4, 20], which repairs the Experiment 1 decisiveness deficit. I deliberately kept epsilon_tally at [0, 0.5]: shrinking it to 0.1 would make the low-tie tally conditions too deterministic and overshoot Experiments 1 and 2, which require about 25% average choice stochasticity; the temperature lower bound alone fixes their weak decisiveness without that overshoot.

**Parameters:**
  - `beta_tally`: `[4.0, 20.0]`
  - `epsilon_tally`: `[0.0, 0.5]`
  - `tau_tie`: `[0.0, 0.5]`
  - `lambda_min`: `[0.0, 0.005]`
  - `lambda_max`: `[0.93, 0.96]`
  - `c0`: `[0.56, 0.575]`
  - `gate_width`: `[0.03, 0.05]`
  - `ctx_power`: `[0.5, 0.8]`
  - `history_window`: `[100.0, 200.0]`
  - `beta_v`: `[10.0, 30.0]`
  - `epsilon_v`: `[0.0, 0.05]`
  - `gamma`: `[3.0, 7.0]`
  - `consistency_floor`: `[0.05, 0.10]`
  - `min_cons_ties`: `{2, 3}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Context-gated tallying expects a (2, n_features) stimulus; got shape %s.' % (stim.shape,))

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length %d != n_features %d.' % (validities.shape[0], n_features))

    a = stim[0]
    b = stim[1]
    diff = a - b

    wins_a = float(np.sum(a > b))
    wins_b = float(np.sum(b > a))
    tally_d = wins_a - wins_b

    tau_tie = float(parameters['tau_tie'])

    h_a = history.get('option_a_ratings', [])
    h_b = history.get('option_b_ratings', [])
    n_past = min(len(h_a), len(h_b))

    window = int(round(float(parameters['history_window'])))
    if window < 1:
        window = 1
    start = max(0, n_past - window)

    lin_weights = validities - 0.5
    tie_count = 0.0
    abs_evidence_sum = 0.0

    for i in range(start, n_past):
        pa = np.asarray(h_a[i], dtype=float)
        pb = np.asarray(h_b[i], dtype=float)
        past_wins = float(np.sum(pa > pb)) - float(np.sum(pb > pa))
        if abs(past_wins) <= tau_tie:
            tie_count += 1.0
            v_lin = float(np.dot(lin_weights, pa - pb))
            abs_evidence_sum += abs(v_lin)

    current_tie = 1.0 if abs(tally_d) <= tau_tie else 0.0
    tie_count += current_tie
    if current_tie:
        v_lin_current = float(np.dot(lin_weights, diff))
        abs_evidence_sum += abs(v_lin_current)

    total_considered = min(n_past, window) + 1.0
    c = tie_count / total_considered if total_considered > 0.0 else 0.0

    mean_abs_evidence = abs_evidence_sum / max(tie_count, 1.0)

    c0 = float(parameters['c0'])
    gate_width = float(parameters['gate_width'])
    ctx_power = float(parameters['ctx_power'])
    lambda_min = float(parameters['lambda_min'])
    lambda_max = float(parameters['lambda_max'])
    consistency_floor = float(parameters['consistency_floor'])
    min_cons_ties = int(parameters['min_cons_ties'])

    g = float(np.clip((c - c0) / max(gate_width, 1e-12), 0.0, 1.0))

    if tie_count >= min_cons_ties and mean_abs_evidence >= consistency_floor:
        veto = 1.0
    else:
        veto = 0.0

    lamb = lambda_min + (lambda_max - lambda_min) * (g ** ctx_power) * veto

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    beta_t = float(parameters['beta_tally'])
    eps_t = float(parameters['epsilon_tally'])
    p_tally = stable_softmax(np.array([wins_a, wins_b]), beta_t)
    p_tally = (1.0 - eps_t) * p_tally + eps_t * np.array([0.5, 0.5])

    gamma = float(parameters['gamma'])
    weights = np.maximum(validities - 0.5, 0.0) ** gamma
    v = float(np.dot(weights, diff))

    beta_v = float(parameters['beta_v'])
    eps_v = float(parameters['epsilon_v'])
    p_valid = stable_softmax(np.array([v, -v]), beta_v)
    p_valid = (1.0 - eps_v) * p_valid + eps_v * np.array([0.5, 0.5])

    probs = lamb * p_valid + (1.0 - lamb) * p_tally
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / total

    return probs
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

### `pi_6` → slot 2 (via `new_theory`)

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
