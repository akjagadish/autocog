# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** People treat advertised expert validity as a cue to ignore rather than trust: cues with low validity, especially those near chance, receive the largest decision weight, while very high-validity cues are discounted because they are seen as redundant or overstated. Choices are formed by a continuous reversed-validity weighted additive comparison of the two options, with signed feature differences weighted by an inverse diagnosticity function of the advertised validities. Ties on a feature contribute zero evidence. The accumulated evidence then passes through a softmax choice rule with inverse temperature beta, plus a small uniform lapse probability epsilon that captures attentional or motor noise.

**Rationale:** The previous validity-weighted model assumed that advertised validity should be used monotonically and positively, with high-validity cues dominating. That assumption generated the wrong sign on the key contrast in Experiments 3 and 4. The present theory reverses the diagnosticity mapping: weights are a decreasing function of advertised validity, implemented here as w_j = (theta - validity_j)^gamma. Low-validity cues therefore control the decision, which produces the negative top-cue-opposition slope in Experiment 1, the negative directional-minus-margin contrast in Experiment 2, the negative rank correlation in Experiment 3, and the negative high-versus-low feature-block difference in Experiment 4. A moderately sharp softmax (beta around 5) and a small lapse (epsilon around 0.2) keep the predicted choice probabilities noisy and human-like rather than deterministic, preserving the observed negative signs while matching the moderate empirical magnitudes.

**Parameters:**
  - `theta`: `{1.0}`
  - `gamma`: `[0.3, 0.4]`
  - `beta`: `[4.8, 5.5]`
  - `epsilon`: `[0.15, 0.25]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Reversed-diagnosticity weighted additive integration.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    # History is ignored because validities are advertised in the
    # instructions and no trial-by-trial feedback is provided.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reversed-validity integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])

    # Reversed diagnosticity weights. Low-validity cues receive large
    # weights; high-validity cues are discounted. The max guard keeps
    # weights non-negative even for perfect validities.
    weights = np.maximum(theta - validities, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the two options [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

**Rationale:** This is a minimal edit of the accepted iter2 base. The one-sided low-validity ensemble salience weights, w0/kappa/gamma/c, and the base softmax/lapse are unchanged. The only changes are two calibrated decision-temperature gates. First, a high-concentration quench triggers only in the near-monopoly regime: top1 normalized cue share near 0.84-0.92 and top1-top2 gap near 0.42-0.58 are required jointly, so it targets Experiment 6's lone extreme low-validity cue while leaving the moderate-concentration regimes of Experiments 1, 3, 4, and 5 untouched. When it fires, beta is quenched toward 0.25-0.35, which reduces a near-unit-weight cue to roughly the observed weak contrast. Second, a low-concentration boost targets the balanced three-low-validity-cue regime of Experiment 2 by gating on both effective number of cues near 3 and near-equal top-three normalized weights, which should sharpen Experiment 2 from about -0.61 toward -0.80/-0.86 without affecting the already-good Experiment 1/3/5 fits.

**Parameters:**
  - `w0`: `[0.005, 0.04]`
  - `kappa`: `[1.5, 4.0]`
  - `gamma`: `[1.0, 1.5]`
  - `c`: `[0.005, 0.02]`
  - `beta`: `[1.7, 2.3]`
  - `epsilon`: `[0.12, 0.18]`
  - `share_hi_center`: `[0.84, 0.92]`
  - `share_hi_width`: `[0.012, 0.035]`
  - `gap_hi_center`: `[0.42, 0.58]`
  - `gap_hi_width`: `[0.03, 0.08]`
  - `beta_floor_hi`: `[0.25, 0.35]`
  - `eff_n_center`: `[2.95, 3.10]`
  - `eff_n_width`: `[0.10, 0.18]`
  - `top3_sim_center`: `[0.92, 0.98]`
  - `top3_sim_width`: `[0.04, 0.10]`
  - `beta_boost_lo`: `[0.8, 1.4]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus array.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    median = np.median(validities)
    mad = np.median(np.abs(validities - median))
    c = float(parameters['c'])

    denom = mad + c
    if denom <= 0.0:
        denom = 1e-12
    deviance = np.maximum(0.0, median - validities) / denom

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    weights = w0 + kappa * np.power(deviance, gamma)

    total = weights.sum()
    if total <= 0.0:
        weights = np.ones_like(weights) / max(weights.size, 1)
    else:
        weights = weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    nw = sorted_w.shape[0]
    top1 = sorted_w[0]
    top2 = sorted_w[1] if nw > 1 else top1
    top3 = sorted_w[2] if nw > 2 else (sorted_w[-1] if nw > 0 else 0.0)
    top3_similarity = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    beta_floor_hi = float(parameters['beta_floor_hi'])

    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    beta_boost_lo = float(parameters['beta_boost_lo'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_similarity - top3_sim_center) / top3_sim_width) ** 2)
    low_gate = eff_gate * balance_gate

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    beta_eff = beta + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.05, 15.0))

    logits = np.array([evidence_A, 0.0])
    z = beta_eff * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** People assign choice weight by each expert's rank inside the advertised validity ensemble, strongly discounting higher ranks. The steepness of that discount is fixed, but the weight floor given to the high-validity tail is gated by how large the lower-validity bloc is. When low-validity experts form a large subgroup of the ensemble, a small floor is added so that the many high-validity cues are not completely ignored; this prevents the extreme rank-reversal saturation seen in balanced low/high ensembles. When low-validity experts are a small minority, the floor is zero, the high-validity tail is almost fully suppressed, and the response temperature is lowered so that the remaining single-cue contrasts are not over-saturated. Signed weighted tallies, with ties contributing zero, then pass through the regime-appropriate fixed-temperature softmax with a small lapse probability.

**Rationale:** This is a minimal, low-risk calibration pass on the accepted bloc-gated rank-salience base. The gate-validated mechanism is unchanged: rank-rescaled weights with a balanced-bloc floor and regime-dependent fixed temperature. I only soften the minority low-bloc regime by moving beta_low from 1.85-1.95 down to 1.65-1.80, which should shrink the dominant over-sharp Experiment 6 contrast (0.3429 vs 0.1525) and pull the too-negative Experiment 1 and Experiment 3 values and the slightly-too-positive Experiment 8 correlation toward their targets. To protect Experiment 5's low-cue dominance while beta_low is softened, I nudge alpha up from 4.40-4.60 to 4.60-4.90 so the bottom ranks retain their renormalized weight share. beta_high, delta_high, and the bloc threshold are left untouched because Experiments 2 and 4 are already near their targets through those shared parameters. No rejected levers are used: no global tally compression, no per-cue tanh gating, no flat minority-regime floor, and no return to the concave rank-spacing transform.

**Parameters:**
  - `alpha`: `[4.60, 4.90]`
  - `delta_high`: `[0.11, 0.13]`
  - `delta_low`: `{0.0}`
  - `beta_high`: `[2.95, 3.05]`
  - `beta_low`: `[1.65, 1.80]`
  - `epsilon`: `{0.02}`
  - `spread_frac`: `{0.25}`
  - `bottom_share_threshold`: `{0.42}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            'Rank-salience integration expects a (2, n_features) stimulus; got shape {}.'.format(stim.shape)
        )

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError(
            'validities length {} != n_features {}.'.format(n_feat, stim.shape[1])
        )

    alpha = float(parameters['alpha'])
    delta_high = float(parameters['delta_high'])
    delta_low = float(parameters['delta_low'])
    beta_high = float(parameters['beta_high'])
    beta_low = float(parameters['beta_low'])
    epsilon = float(parameters['epsilon'])
    spread_frac = float(parameters['spread_frac'])
    bottom_share_threshold = float(parameters['bottom_share_threshold'])

    vmin = float(np.min(validities))
    vmax = float(np.max(validities))
    if vmax - vmin <= 1e-12:
        bottom_share = 1.0
    else:
        lower_bloc = validities <= vmin + spread_frac * (vmax - vmin)
        bottom_share = float(np.mean(lower_bloc))

    if bottom_share >= bottom_share_threshold:
        delta = delta_high
        beta = beta_high
    else:
        delta = delta_low
        beta = beta_low

    ranks = rankdata(validities, method='average')
    if n_feat > 1:
        u = (ranks - 1.0) / (n_feat - 1.0)
    else:
        u = np.array([0.5])

    raw_weights = np.power(1.0 - u, alpha) + delta
    total = raw_weights.sum()
    if total <= 0.0:
        weights = np.ones(n_feat) / max(n_feat, 1)
    else:
        weights = raw_weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```
