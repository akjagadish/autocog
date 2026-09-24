# Round 7 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

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


### slot 2 — `pi_9` — SURVIVED ✓

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


## Replacement

### `pi_10` → slot 1 (via `new_theory`)

**Description:** Conflict-gated validity-gap compensation with strict homogeneous-bloc licensing and stronger opposition-sensitive lone amplification (GVC-strict). People anchor each advertised expert validity to the ensemble median. Below-median cues earn decision weight through a median/MAD-normalized deviance score. A single diagnostically deviant low cue is strongly amplified, with amplification increasing in the number of opposed at-or-above-median cues. When multiple below-median cues are diagnostic, a homogeneous low-deviant group is attenuated by cue count, but an extreme focal low cue is exempted from that group attenuation so it can still dominate the signed evidence. At-or-above-median cues receive no default weight; their validity-gap bonus is licensed only by a very homogeneous, sufficiently deviant bloc of three or more simultaneously diagnostic below-median cues, and is sharply damped by low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

**Rationale:** Minimal in-family repair of the running-best GVC-focal candidate. The critic's main diagnosis was that the validity-gap bonus was still too readily licensed and insufficiently suppressed under low-versus-high conflict. Here K is moved from [0.70, 1.00] to [0.30, 0.50], sharply damping the high-cue bonus in conflict, which should fix the Exp2 direction and recover Exp15 without a global beta increase. The homogeneous low-bloc license is made stricter by reducing b_boost to [0.02, 0.08], lowering D_boost to [0.6, 1.2], narrowing H_spread to [0.005, 0.02], and raising the homogeneity exponent from 8 to 12, so Exp6-like near-median blocs no longer reopen the gate while genuinely deviant homogeneous blocs in Exp4/13 can still do so. To preserve lone-low following under stronger conflict damping, lone amplification is strengthened: lambda [4.5, 5.5], nu [0.15, 0.25], and opp_cap fixed at 5. Beta, phi, and rho are deliberately left unchanged to avoid re-breaking Exp3/4/5. The fixed-temperature softmax, no lapse floor, and hard-zero high-cue default are retained.

**Parameters:**
  - `c`: `[0.008, 0.012]`
  - `w0`: `[0.007, 0.012]`
  - `kappa`: `[1.9, 2.3]`
  - `gamma`: `[1.25, 1.45]`
  - `lambda`: `[4.5, 5.5]`
  - `sigma`: `[0.10, 0.18]`
  - `phi`: `[0.55, 0.75]`
  - `nu`: `[0.15, 0.25]`
  - `opp_cap`: `{5}`
  - `focal_ratio`: `[3.0, 4.0]`
  - `gate_base`: `{0.0}`
  - `lone_floor`: `[0.00, 0.03]`
  - `D_half`: `[0.35, 0.55]`
  - `q_dev`: `{2}`
  - `b_boost`: `[0.02, 0.08]`
  - `D_boost`: `[0.6, 1.2]`
  - `H_spread`: `[0.005, 0.02]`
  - `K`: `[0.30, 0.50]`
  - `w_hi0`: `{0.0}`
  - `rho`: `[0.08, 0.14]`
  - `alpha`: `[1.1, 1.8]`
  - `beta`: `[1.4, 1.8]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('GVC expects a (2, n_features) stimulus.')

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
    at_or_above = ~below
    gap = np.maximum(0.0, validities - median) / scale

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lambda'])
    sigma = float(parameters['sigma'])
    phi = float(parameters['phi'])
    nu = float(parameters['nu'])
    opp_cap = float(parameters['opp_cap'])
    focal_ratio = float(parameters['focal_ratio'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))

    def lone_amplification(idx):
        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        att = 1.0 / (1.0 + phi * (n_active - 1))
        active_devs = dev[active_below]
        i_max = int(np.argmax(active_devs))
        max_dev = float(active_devs[i_max])
        med_dev = float(np.median(active_devs))
        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:
            focal = int(np.flatnonzero(active_below)[i_max])
            others = below.copy()
            others[focal] = False
            raw[others] = raw[others] * att
            raw[focal] = raw[focal] * lone_amplification(focal)
        else:
            raw[below] = raw[below] * att

    gate_base = float(parameters['gate_base'])
    lone_floor = float(parameters['lone_floor'])
    D_half = float(parameters['D_half'])
    q_dev = float(parameters['q_dev'])
    b_boost = float(parameters['b_boost'])
    D_boost = float(parameters['D_boost'])
    H_spread = float(parameters['H_spread'])
    K = float(parameters['K'])

    gate = gate_base
    if n_active == 1:
        gate *= lone_floor

    if n_active > 0:
        active_devs = dev[active_below]
        dmax = float(np.max(active_devs))
    else:
        dmax = 0.0

    if dmax > 1e-12:
        gate *= 1.0 / (1.0 + (dmax / D_half) ** q_dev)

    if n_active >= 3:
        low_vals = validities[active_below]
        spread = float(np.std(low_vals))
        spread_norm = spread / scale
        hom = 1.0 / (1.0 + (spread_norm / H_spread) ** 12.0)
        mean_dev = float(np.mean(dev[active_below]))
        act = float(np.tanh(mean_dev / D_boost))
        gate += b_boost * hom * act

    w_hi0 = float(parameters['w_hi0'])
    rho = float(parameters['rho'])
    alpha = float(parameters['alpha'])

    gap_bounded = np.minimum(gap, 100.0)
    hi_pre = np.zeros(n_feat, dtype=float)
    if np.any(at_or_above):
        hi_pre[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha)

    E_low = float(np.dot(raw, diff))
    E_high_pre = float(np.dot(hi_pre, diff))

    if E_low * E_high_pre < 0.0:
        conflict = 2.0 * min(abs(E_low), abs(E_high_pre)) / (abs(E_low) + abs(E_high_pre) + 1e-12)
        gate *= 1.0 / (1.0 + (conflict / K) ** 2.0)

    gate = float(np.maximum(gate, 0.0))

    raw_high = np.zeros(n_feat, dtype=float)
    if np.any(at_or_above):
        raw_high[at_or_above] = w_hi0 + rho * np.power(gap_bounded[at_or_above], alpha) * gate

    raw_all = raw + raw_high
    total = float(np.sum(raw_all))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw_all / total

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
