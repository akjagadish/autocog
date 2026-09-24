# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

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


### slot 2 — `pi_9` — KILLED ✗

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

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Median anchoring of advertised validities gives below-median cues MAD-normalized deviance weight. Lone diagnostic low cues receive strong amplification, while multiple diagnostic low cues receive count-based attenuation that is deviance-selective: an extreme below-median singleton is partially exempt from attenuation without demoting inactive low cues. Above-median cues contribute through a sign-flexible rank-ordered consensus term gated by the signed evidence carried by actually diagnostic low cues. Balanced low evidence permits a negative consensus weight, opposing low evidence gates the high-cue term toward zero, and weak low evidence permits only a small capped positive rank bonus. Large comparable multi-low coalitions receive a linear soft attenuation of their total active low evidence to prevent aligned low blocs from over-saturating choice. Choice is a fixed-temperature softmax with a tiny lapse floor.

**Rationale:** This is a minimal in-family edit of the accepted iter-5 base, directly implementing the most recent critique. The multi-active branch no longer applies one uniform attenuation to all below-median cues including inactive ones. Instead, active below-median cues receive a smoothed deviance-relative attenuation: a cue whose deviance is extreme relative to the other active low cues receives a partial attenuation exemption, while comparable low-cue coalitions keep the iter-5 base attenuation. Only active low cues are attenuated, so inactive low-cue mass and the denominator are not artificially collapsed. The low-evidence gates now use the mass and signed evidence of actually diagnostic active low cues rather than total low-cue mass, which sharpens opposition gating and preserves exact no-lapse patterns such as Experiment 13. A linear soft cap on the final active low-evidence contribution is applied only for large non-outlier multi-low coalitions, addressing the Experiment 6 overshoot without re-introducing outlier amplification or demoting inactive cues. All other accepted mechanisms and parameter ranges are unchanged.

**Parameters:**
  - `c`: `[0.008, 0.012]`
  - `w0`: `[0.007, 0.012]`
  - `kappa`: `[1.9, 2.3]`
  - `gamma`: `[1.25, 1.45]`
  - `lone_gain`: `[5.5, 6.5]`
  - `lone_sigma`: `[0.10, 0.18]`
  - `phi`: `[1.2, 1.5]`
  - `nu`: `[0.15, 0.25]`
  - `opp_cap`: `{5}`
  - `out_gain`: `[0.50, 0.80]`
  - `out_thresh`: `[2.2, 2.8]`
  - `out_slope`: `[3.5, 5.5]`
  - `cap_out_gate`: `{0.45}`
  - `n_cap_min`: `{4}`
  - `low_multi_scale`: `[0.60, 0.80]`
  - `rho_hi`: `[0.025, 0.035]`
  - `alpha_hi`: `[3.0, 4.5]`
  - `h_cap`: `[0.06, 0.12]`
  - `h_pos`: `[0.15, 0.25]`
  - `h_neg`: `[1.30, 1.55]`
  - `min_low_mass`: `{0.10}`
  - `low_bal_eps`: `{0.02}`
  - `weak_thresh`: `[0.12, 0.20]`
  - `n_min_hi`: `{2}`
  - `cons_thresh`: `[0.82, 0.88]`
  - `cons_s`: `{0.04}`
  - `beta`: `[1.85, 2.20]`
  - `epsilon`: `{0.001}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('DSA-SFCC expects a (2, n_features) stimulus.')

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
    at_or_above = ~below
    gap = np.maximum(0.0, validities - median) / scale

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lone_gain = float(parameters['lone_gain'])
    lone_sigma = float(parameters['lone_sigma'])
    phi = float(parameters['phi'])
    nu = float(parameters['nu'])
    opp_cap = float(parameters['opp_cap'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)
    pre_att_raw = raw.copy()

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))
    use_low_cap = False

    def lone_amplification(idx):
        base_amp = 1.0 + lone_gain * np.tanh(float(dev[idx]) / lone_sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        base_att = 1.0 / (1.0 + phi * (n_active - 1))
        d_act = dev[active_below]
        d_ref = float(np.median(d_act)) + 1e-12
        out_slope = float(parameters['out_slope'])
        out_thresh = float(parameters['out_thresh'])
        out_gain = float(parameters['out_gain'])

        ratio = np.clip(d_act / d_ref, 0.0, 20.0)
        outlier = 1.0 / (1.0 + np.exp(-out_slope * (ratio - out_thresh)))
        att_factors = base_att + (1.0 - base_att) * out_gain * outlier
        raw[active_below] = raw[active_below] * att_factors

        outlier_max = float(np.max(outlier))
        cap_out_gate = float(parameters['cap_out_gate'])
        n_cap_min = int(float(parameters['n_cap_min']))
        use_low_cap = (outlier_max < cap_out_gate) and (n_active >= n_cap_min)

    high_base = np.zeros(n_feat, dtype=float)
    above_idx = np.flatnonzero(above)
    if above_idx.size > 0:
        hi_gap = gap[above]
        order = np.argsort(hi_gap)
        m = above_idx.size
        ranks = np.empty(m, dtype=float)
        ranks[order] = np.arange(1, m + 1, dtype=float)
        if m > 1:
            u = (ranks - 1.0) / (m - 1.0)
        else:
            u = np.ones(m, dtype=float)
        rho_hi = float(parameters['rho_hi'])
        alpha_hi = float(parameters['alpha_hi'])
        bonus = rho_hi * np.power(np.clip(u, 0.0, 1.0), alpha_hi)
        high_base[above] = bonus
        total_hi = float(np.sum(high_base))
        h_cap = float(parameters['h_cap'])
        if total_hi > h_cap and total_hi > 1e-12:
            high_base[above] *= h_cap / total_hi

    total_low = float(np.sum(raw))
    E_low = float(np.dot(raw, diff))
    total_hi = float(np.sum(high_base))
    E_high = float(np.dot(high_base, diff))

    active_low_mass = float(np.sum(raw[active_below]))
    low_strength = abs(E_low) / (active_low_mass + 1e-12) if active_low_mass > 1e-12 else 0.0
    low_bal_eps = float(parameters['low_bal_eps'])
    min_low_mass = float(parameters['min_low_mass'])
    pre_att_active_mass = float(np.sum(pre_att_raw[active_below]))
    has_low_evidence = pre_att_active_mass > min_low_mass
    low_bal = 1.0 if (has_low_evidence and low_strength <= low_bal_eps) else 0.0
    weak_thresh = float(parameters['weak_thresh'])
    low_weak = 1.0 / (1.0 + (low_strength / weak_thresh) ** 2)

    active_high = above & (np.abs(diff) > 1e-12)
    n_ha = int(np.count_nonzero(active_high))
    consensus = 0.0
    if n_ha > 0:
        w_act = high_base[active_high]
        total_act = float(np.sum(w_act))
        if total_act > 1e-12:
            E_act = float(np.dot(w_act, diff[active_high]))
            align_share = 0.5 + 0.5 * abs(E_act) / total_act
            n_min_hi = float(parameters['n_min_hi'])
            size_factor = float(np.tanh(n_ha / max(n_min_hi, 1e-12)))
            consensus = align_share * size_factor

    cons_thresh = float(parameters['cons_thresh'])
    cons_s = float(parameters['cons_s'])
    zz = (cons_thresh - consensus) / cons_s
    zz = float(np.clip(zz, -60.0, 60.0))
    strong = 1.0 / (1.0 + float(np.exp(zz)))

    strong_neg = low_bal * strong
    h_pos = float(parameters['h_pos'])
    h_neg = float(parameters['h_neg'])
    g = h_pos * low_weak * (1.0 - strong_neg) - h_neg * strong_neg
    g = float(np.clip(g, -h_neg, h_pos))

    E_low_evidence = E_low
    if use_low_cap:
        low_multi_scale = float(parameters['low_multi_scale'])
        E_low_evidence = low_multi_scale * E_low

    denom = total_low + total_hi
    if denom <= 1e-12:
        evidence = 0.0
    else:
        evidence = (E_low_evidence + g * E_high) / denom

    beta = float(parameters['beta'])
    logits = np.array([evidence, 0.0], dtype=float)
    z = beta * (logits - float(np.max(logits)))
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    core = e / np.sum(e)

    epsilon = float(parameters['epsilon'])
    probs = (1.0 - epsilon) * core + epsilon * np.ones(2, dtype=float) / 2.0
    probs = probs / np.sum(probs)
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
