# Round 6 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_8` — SURVIVED ✓

**Description:** People anchor each advertised expert validity to the current ensemble, so cues below the ensemble median earn decision weight through a median/MAD-normalized deviance score that is multiplicatively amplified by cue distinctiveness. A homogeneous low-validity bloc shares weight evenly, while a single extreme low-validity cue becomes highly salient and dominates the signed evidence. A small, strictly bounded rank-based baseline keeps high-validity cues from being ignored, with strength that increases with high-bloc size and decreases when low cues are highly distinctive. Choice stochasticity is governed by a conditional entropy/conflict gate: homogeneous multi-cue low blocs receive a high temperature drive floor, whereas conflict damping is strengthened selectively when the high-validity bloc outnumbers multiple low cues or when a low cue is solitary and highly distinctive. This preserves strong homogeneous low-bloc choices and extreme-outlier following without allowing high-cue sweeps or high-distinctness conflicts to over-saturate.

**Rationale:** This is a minimal edit of the accepted distinctiveness-gated low-cue dominance model. The key change addresses the prior uniform softening: distinctiveness is no longer allowed to halve the stochasticity drive for homogeneous multi-cue low blocs. When at least three below-median cues are nearly identical, the drive factor is floored near 0.9 instead of falling to 0.5, restoring the strong homogeneous low-bloc behavior needed in Experiment 2 while leaving the two-cue high-bloc cases in Experiment 5 on the original path. Conflict damping is made condition-specific rather than global: the base damping term is lowered, but additive boosts now apply when the high-validity bloc outnumbers multiple low cues, when low-cue distinctiveness is large, or when a low cue is solitary. This keeps high-cue sweeps and solitary-outlier conflicts from re-overshooting in Experiments 5, 6, 11, and 12 while allowing the Experiment 2 boost. The high-validity rank baseline is left broadly intact, and several parameter ranges are widened to restore realistic between-subject variance.

**Parameters:**
  - `c`: `[0.007, 0.016]`
  - `gamma`: `[1.15, 1.55]`
  - `w0`: `[0.05, 0.13]`
  - `kappa`: `[2.0, 3.2]`
  - `lam`: `{0.6}`
  - `distinct_scale`: `[1.2, 2.6]`
  - `eta`: `{1.0}`
  - `b_scale`: `[0.25, 0.55]`
  - `b_pow`: `{1.0}`
  - `b_base`: `[0.01, 0.09]`
  - `b_rank`: `[0.08, 0.22]`
  - `b_alpha`: `[0.9, 1.3]`
  - `b_cap`: `[0.12, 0.33]`
  - `d_gate`: `[7.0, 16.0]`
  - `H_thresh`: `[0.75, 0.90]`
  - `s_H`: `[0.04, 0.12]`
  - `D_half`: `[0.8, 2.2]`
  - `hom_count_thresh`: `{3}`
  - `hom_dmax_thresh`: `[0.25, 0.60]`
  - `hom_floor`: `[0.82, 0.95]`
  - `e_half`: `[0.08, 0.35]`
  - `beta_base`: `[0.30, 0.60]`
  - `beta_range`: `[3.2, 5.6]`
  - `conflict_damp`: `[0.26, 0.50]`
  - `c_asym`: `[0.18, 0.42]`
  - `c_distinct`: `[0.06, 0.14]`
  - `c_solitary`: `[0.03, 0.10]`
  - `beta_min`: `[0.08, 0.28]`
  - `beta_max`: `[4.2, 5.8]`
  - `epsilon`: `[0.08, 0.22]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Distinctive low-cue dominance expects a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    c = float(parameters['c'])
    median = float(np.median(validities))
    abs_dev = np.abs(validities - median)
    mad = float(np.median(abs_dev))
    scale = mad + c
    if scale <= 1e-12:
        scale = 1e-12

    dev = np.maximum(0.0, median - validities) / scale
    below = validities < median - 1e-12

    distinct = np.zeros(n_feat, dtype=float)
    low_idx = np.where(below)[0]
    if low_idx.size == 1:
        distinct[low_idx[0]] = dev[low_idx[0]]
    elif low_idx.size > 1:
        for i in low_idx:
            others = [j for j in low_idx if j != i]
            if others:
                dgap = min(abs(float(validities[j]) - float(validities[i])) for j in others)
                distinct[i] = dgap / scale
            else:
                distinct[i] = dev[i]

    gamma = float(parameters['gamma'])
    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    lam = float(parameters['lam'])
    distinct_scale = float(parameters['distinct_scale'])
    eta = float(parameters['eta'])

    dev_bounded = np.minimum(dev, 100.0)
    dist_score = np.tanh(distinct / distinct_scale)

    low_raw = np.zeros(n_feat, dtype=float)
    low_raw[below] = (
        w0
        + kappa * np.power(dev_bounded[below], gamma)
        * (1.0 + lam * np.power(dist_score[below], eta))
    )

    high_mask = ~below
    high_raw = np.zeros(n_feat, dtype=float)
    if high_mask.any():
        if below.any():
            dmax_base = float(np.max(distinct))
        else:
            dmax_base = 0.0

        d_gate = float(parameters['d_gate'])
        baseline_factor = 1.0 / (1.0 + (dmax_base / d_gate) ** 2)

        vh = validities[high_mask]
        nh = int(vh.size)
        high_frac = nh / float(n_feat)
        ranks = rankdata(vh, method='average')
        if nh > 1:
            u = (ranks - 1.0) / (nh - 1.0)
        else:
            u = np.array([1.0], dtype=float)

        b_scale = float(parameters['b_scale'])
        b_pow = float(parameters['b_pow'])
        b_base = float(parameters['b_base'])
        b_rank = float(parameters['b_rank'])
        b_alpha = float(parameters['b_alpha'])
        b_cap = float(parameters['b_cap'])

        raw_h = (
            b_scale
            * np.power(high_frac, b_pow)
            * baseline_factor
            * (b_base + b_rank * np.power(u, b_alpha))
        )
        raw_h = np.minimum(raw_h, b_cap)
        high_raw[high_mask] = raw_h

    raw = low_raw + high_raw
    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    diff = stim[0] - stim[1]
    evidence = float(np.dot(weights, diff))

    low_w = weights * below.astype(float)
    high_w = weights * high_mask.astype(float)
    low_evidence = float(np.dot(low_w, diff))
    high_evidence = float(np.dot(high_w, diff))

    if low_evidence * high_evidence < 0.0:
        conflict = min(abs(low_evidence), abs(high_evidence)) / (abs(evidence) + 1e-12)
        conflict = float(np.clip(conflict, 0.0, 1.0))
    else:
        conflict = 0.0

    en = -np.sum(weights * np.log(weights + 1e-12))
    entropy = float(en / np.log(max(n_feat, 2)))

    H_thresh = float(parameters['H_thresh'])
    s_H = float(parameters['s_H'])
    concentration = 1.0 / (1.0 + np.exp((entropy - H_thresh) / s_H))

    if below.any():
        dmax = float(np.max(distinct))
    else:
        dmax = 0.0

    n_low = int(low_idx.size)
    D_half = float(parameters['D_half'])
    base_distinct_factor = 0.5 + 0.5 * np.tanh(dmax / D_half)

    hom_count_thresh = float(parameters['hom_count_thresh'])
    hom_dmax_thresh = float(parameters['hom_dmax_thresh'])
    hom_floor = float(parameters['hom_floor'])
    homogeneous_low_bloc = (n_low >= hom_count_thresh) and (dmax < hom_dmax_thresh)
    if homogeneous_low_bloc:
        distinct_factor = hom_floor + (1.0 - hom_floor) * np.tanh(dmax / D_half)
    else:
        distinct_factor = base_distinct_factor

    low_share = abs(low_evidence) / (abs(low_evidence) + abs(high_evidence) + 1e-12)

    e_half = float(parameters['e_half'])
    spread = abs(evidence) / (abs(evidence) + e_half)

    beta_base = float(parameters['beta_base'])
    beta_range = float(parameters['beta_range'])
    conflict_damp = float(parameters['conflict_damp'])
    c_asym = float(parameters['c_asym'])
    c_distinct = float(parameters['c_distinct'])
    c_solitary = float(parameters['c_solitary'])
    beta_min = float(parameters['beta_min'])
    beta_max = float(parameters['beta_max'])

    high_frac_for_gate = float(np.mean(high_mask))
    low_frac = n_low / float(n_feat)
    asym = float(np.clip(high_frac_for_gate - low_frac, 0.0, 1.0))
    distinct_conf = float(np.tanh(dmax / D_half))
    solitary = 1.0 if n_low == 1 else 0.0
    asym_term = asym * (1.0 if n_low > 1 else 0.0)

    conflict_damp_eff = (
        conflict_damp
        + c_asym * asym_term
        + c_distinct * distinct_conf
        + c_solitary * solitary
    )
    conflict_damp_eff = float(np.clip(conflict_damp_eff, 0.0, 0.95))

    drive = concentration * low_share * distinct_factor * spread
    beta = beta_base + beta_range * drive * (1.0 - conflict_damp_eff * conflict)
    beta = float(np.clip(beta, beta_min, beta_max))

    epsilon = float(parameters['epsilon'])
    logits = np.array([evidence, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    e = np.exp(z)
    core = e / np.sum(e)

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts, dtype=float) / float(n_opts)
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


### slot 2 — `pi_5` — KILLED ✗

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

### `pi_9` → slot 2 (via `new_theory`)

**Description:** Deviance-anchored, state-gated high-cue bonus with diagnostic low-cue attenuation (DASH). People anchor each advertised expert validity to the current ensemble median. Below-median cues earn weight from a median/MAD-normalized deviance score, but when two or more below-median cues are diagnostic on a trial their weights are attenuated by a count-dependent factor, preventing co-equal low-validity blocs from dominating choice. A solitary diagnostic below-median cue instead receives an amplified weight, capturing strong lone-cue following. Above-median cues receive only a tiny at-median baseline plus a rank-ordered bonus that is gated by current signed low-cue evidence and low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

**Rationale:** This is a minimal-diff edit of the accepted DASH base. Following the most recent critique, it replaces the flat low-cue weighting with a diagnostic count-dependent attenuation: when two or more below-median cues have nonzero signed differences on the current trial, their weights are multiplied by 1/(1 + phi*(n_active-1)). This directly targets the multi-active low-bloc overdominance in Experiments 3, 4, 5, and 6 while leaving single-active low-cue trials unchanged. The solitary amplifier is strengthened by raising lambda to [2.2, 2.8] and lowering sigma to [0.20, 0.30], and it is now triggered only when exactly one active below-median cue exists, which should push Experiment 12 above the accepted base. High-cue influence is reduced only modestly by lowering the constant at-median/high baseline and slightly reducing the rho upper bound, aimed at the high-tally sweeps in Experiments 14 and 11 without the excessive high-cue suppression that previously regressed Experiments 2, 10, and 13. The fixed-temperature softmax and zero epsilon floor are preserved, maintaining the zero-lapse estimate for Experiment 13.

**Parameters:**
  - `c`: `[0.008, 0.013]`
  - `w0`: `[0.006, 0.015]`
  - `kappa`: `[1.8, 2.4]`
  - `gamma`: `[1.2, 1.5]`
  - `lambda`: `[2.2, 2.8]`
  - `sigma`: `[0.20, 0.30]`
  - `phi`: `[0.25, 0.50]`
  - `w_hi0`: `[0.0008, 0.0015]`
  - `rho`: `[0.018, 0.028]`
  - `alpha`: `[2.5, 3.5]`
  - `a`: `[0.22, 0.38]`
  - `K`: `[0.35, 0.55]`
  - `beta`: `[1.8, 2.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('DASH expects a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    c = float(parameters['c'])
    median = float(np.median(validities))
    abs_dev = np.abs(validities - median)
    mad = float(np.median(abs_dev))
    scale = mad + c
    if scale <= 1e-12:
        scale = 1e-12

    dev = np.maximum(0.0, median - validities) / scale
    below = validities < median - 1e-12
    above = validities > median + 1e-12
    at_median = ~below & ~above

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lambda'])
    sigma = float(parameters['sigma'])
    w_hi0 = float(parameters['w_hi0'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))
    if n_active >= 2:
        phi = float(parameters['phi'])
        att = 1.0 / (1.0 + phi * (n_active - 1))
        raw[below] = raw[below] * att
    elif n_active == 1:
        i = int(np.flatnonzero(active_below)[0])
        amp = 1.0 + lam * np.tanh(float(dev[i]) / sigma)
        raw[i] = raw[i] * amp

    raw[at_median] = w_hi0

    if np.any(above):
        above_idx = np.flatnonzero(above)
        vh = validities[above_idx]
        ranks = rankdata(vh, method='average')
        if vh.size > 1:
            u = (ranks - 1.0) / (vh.size - 1.0)
        else:
            u = np.ones(1, dtype=float)
        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        bonus_pre = rho * np.power(np.clip(u, 0.0, 1.0), alpha)
        raw_pre_high = w_hi0 + bonus_pre
    else:
        raw_pre_high = np.empty(0, dtype=float)

    raw_pre = raw.copy()
    if np.any(above):
        raw_pre[above] = raw_pre_high

    low_mask = below
    high_mask = ~below
    E_low = float(np.dot(raw_pre[low_mask], diff[low_mask])) if np.any(low_mask) else 0.0
    E_high_pre = float(np.dot(raw_pre[high_mask], diff[high_mask])) if np.any(high_mask) else 0.0

    if E_low * E_high_pre < 0.0:
        conflict = 2.0 * min(abs(E_low), abs(E_high_pre)) / (abs(E_low) + abs(E_high_pre) + 1e-12)
    else:
        conflict = 0.0

    a = float(parameters['a'])
    K = float(parameters['K'])
    low_gate = abs(E_low) / (abs(E_low) + a)
    conflict_gate = 1.0 / (1.0 + (conflict / K) ** 2)
    gate = low_gate * conflict_gate

    if np.any(above):
        above_idx = np.flatnonzero(above)
        vh = validities[above_idx]
        ranks = rankdata(vh, method='average')
        if vh.size > 1:
            u = (ranks - 1.0) / (vh.size - 1.0)
        else:
            u = np.ones(1, dtype=float)
        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        bonus = rho * np.power(np.clip(u, 0.0, 1.0), alpha) * gate
        raw[above] = w_hi0 + bonus

    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters['beta'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    e = np.exp(z)
    probs = e / np.sum(e)
    return probs
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```
