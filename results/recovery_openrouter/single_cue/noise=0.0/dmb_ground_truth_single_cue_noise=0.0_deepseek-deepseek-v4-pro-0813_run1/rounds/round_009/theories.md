# Round 9 — Theories

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


### slot 2 — `pi_11` — KILLED ✗

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


## Replacement

### `pi_12` → slot 2 (via `new_theory`)

**Description:** People anchor advertised expert validities to the ensemble median and give below-median cues MAD-normalized deviance weight, with strong lone-cue amplification and count/deviance-based attenuation of multiple diagnostic low cues. Above-median cues receive no default rank bonus; instead they enter through a continuous validity-gap gate that depends on low-bloc size, low-evidence balance, deviance, and of an absolute prominence statistic for the high-bloc: the largest above-median gap must be both large in raw validity units and separated from the next-largest above-median gap by a margin. When that prominence condition fails, the high-cue gate is damped toward zero and control returns to the low-cue branch. When no below-median cues are diagnostic, a small continuous anti-top correction is added to the weak equal-weight fallback, with an intermediate effective temperature, so a single genuinely standout top cue is followed against rather than integrated positively. Choice is a fixed-temperature softmax with no lapse floor, and heterogeneous multi-low-cue trials receive weaker effective decision strength based on signed within-bloc disagreement.

**Rationale:** Minimal edit of the accepted iter-8 base. (1) The high-cue prominence term is now margin-gated in addition to being absolute-gap-gated: the largest above-median validity gap must be separated from the next-largest gap, which suppresses the anti-high branch in Exp19's diffuse/modest high-gap profile while retaining Exp17's distinguishable top cue and Exp20's standout top cue. The existing share-power concentration is kept as a multiplier rather than replaced by a binary share precondition, so active-low experiments such as Exp4/Exp5/Exp12/Exp17 are not coupled to an above-median share threshold. (2) The n_active == 0 fallback now adds a continuous prominence-weighted anti-top correction, mean(diff) + conc_act * anti_top_w * (-diff[top_cue]), with an intermediate effective inverse temperature that interpolates from the weak fallback beta to anti_beta. This fixes Exp20's anti-top direction without a discrete evidence replacement and without contaminating non-standout no-active-low trials. (3) beta_multi_scale is restored to the accepted iter-8 range and the multi-low-cue temperature now also depends on signed within-bloc disagreement, so homogeneous low blocs in Exp5/Exp12/Exp19 keep full decision strength while heterogeneous multi-low-cue trials are softened. The shared low-cue machinery, no-lapse fixed-temperature softmax, continuous gate family, and all other accepted iter-8 components are unchanged.

**Parameters:**
  - `c`: `[0.008, 0.012]`
  - `w0`: `[0.007, 0.012]`
  - `kappa`: `[2.30, 2.80]`
  - `gamma`: `[1.25, 1.45]`
  - `lambda`: `[8.0, 11.0]`
  - `sigma`: `[0.14, 0.22]`
  - `phi`: `[0.38, 0.55]`
  - `nu`: `[0.15, 0.25]`
  - `opp_cap`: `{5}`
  - `focal_ratio`: `[3.0, 4.0]`
  - `rho`: `[0.08, 0.14]`
  - `alpha`: `[1.1, 1.8]`
  - `h_pos`: `[0.005, 0.02]`
  - `h_neg`: `[0.12, 0.20]`
  - `neg_bal_floor`: `[0.25, 0.45]`
  - `neg_dev_floor`: `[0.25, 0.45]`
  - `n_small`: `[0.8, 1.2]`
  - `s_small`: `[0.10, 0.25]`
  - `n_large`: `[7.5, 9.2]`
  - `s_large`: `[1.5, 2.5]`
  - `K_bal`: `[0.40, 0.65]`
  - `H_spread`: `[0.003, 0.012]`
  - `D_half`: `[0.40, 0.65]`
  - `beta`: `[1.8, 2.3]`
  - `fallback_beta_scale`: `[0.05, 0.12]`
  - `one_cue_gate_scale`: `[0.15, 0.35]`
  - `conc_boost_gain`: `[2.5, 4.5]`
  - `conc_power`: `{2}`
  - `branch_boost`: `[0.0, 0.08]`
  - `G_min`: `[0.08, 0.12]`
  - `G_s`: `[0.01, 0.025]`
  - `neg_count_floor_low`: `[0.03, 0.10]`
  - `neg_count_floor_high`: `[0.30, 0.45]`
  - `strong_bal_floor`: `[0.55, 0.80]`
  - `strong_dev_floor`: `[0.55, 0.80]`
  - `beta_multi_scale`: `[0.60, 0.90]`
  - `margin_min`: `[0.012, 0.020]`
  - `margin_s`: `[0.003, 0.008]`
  - `anti_top_w`: `[0.85, 1.15]`
  - `anti_beta`: `[1.8, 2.6]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('GVC-prominence expects a (2, n_features) stimulus.')

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
    gap = np.maximum(0.0, validities - median) / scale
    below = validities < median - 1e-12
    above = validities > median + 1e-12
    at_or_above = ~below

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))

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

    def lone_amplification(idx):
        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        base_att = 1.0 / (1.0 + phi * (n_active - 1))
        active_devs = dev[active_below]
        i_max = int(np.argmax(active_devs))
        max_dev = float(active_devs[i_max])
        med_dev = float(np.median(active_devs))
        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:
            focal = int(np.flatnonzero(active_below)[i_max])
            others = below.copy()
            others[focal] = False
            raw[others] = raw[others] * base_att
            raw[focal] = raw[focal] * lone_amplification(focal)
        else:
            raw[below] = raw[below] * base_att

    # Precompute above-median prominence before the n_active branch so the
    # no-active-low fallback can use a continuous anti-top correction.
    G_min = float(parameters['G_min'])
    G_s = float(parameters['G_s'])
    margin_min = float(parameters['margin_min'])
    margin_s = float(parameters['margin_s'])
    conc_power = float(parameters['conc_power'])
    conc_boost_gain = float(parameters['conc_boost_gain'])

    max_gap_share = 0.0
    max_gap_raw = 0.0
    margin_gap_raw = 0.0
    top_idx = -1

    if np.any(at_or_above):
        gap_above = gap[at_or_above]
        sum_gap = float(np.sum(gap_above))
        if sum_gap > 1e-12:
            max_gap_share = float(np.max(gap_above)) / sum_gap
        raw_gap_above = np.maximum(0.0, validities[at_or_above] - median)
        max_gap_raw = float(np.max(raw_gap_above))

    if np.any(above):
        above_idx = np.flatnonzero(above)
        i_top_above = int(np.argmax(validities[above_idx]))
        top_idx = int(above_idx[i_top_above])
        raw_gaps_above = np.sort(np.maximum(0.0, validities[above_idx] - median))[::-1]
        if raw_gaps_above.size > 1:
            margin_gap_raw = float(raw_gaps_above[0] - raw_gaps_above[1])
        else:
            margin_gap_raw = float(raw_gaps_above[0])

    z_stand = float(np.clip((max_gap_raw - G_min) / G_s, -50.0, 50.0))
    standout = 1.0 / (1.0 + float(np.exp(-z_stand)))

    z_margin = float(np.clip((margin_gap_raw - margin_min) / margin_s, -50.0, 50.0))
    margin_act = 1.0 / (1.0 + float(np.exp(-z_margin)))
    margin_act = float(margin_act ** conc_power)

    share_pow = float(max_gap_share ** conc_power)
    conc_act = share_pow * standout * margin_act

    beta = float(parameters['beta'])
    fallback_beta_scale = float(parameters['fallback_beta_scale'])
    beta_multi_scale = float(parameters['beta_multi_scale'])
    anti_top_w = float(parameters['anti_top_w'])
    anti_beta = float(parameters['anti_beta'])

    if n_active == 0:
        fallback_evidence = float(np.mean(diff))
        base_beta_fb = beta * fallback_beta_scale
        evidence = fallback_evidence
        beta_eff = base_beta_fb
        if top_idx >= 0:
            top_diff = float(diff[top_idx])
            evidence = fallback_evidence + conc_act * anti_top_w * (-top_diff)
            beta_eff = base_beta_fb + conc_act * (anti_beta - base_beta_fb)
    else:
        beta_eff = beta
        mass_active = float(np.sum(raw[active_below]))
        E_low = float(np.dot(raw, diff))
        imbalance = 0.0
        if mass_active > 1e-12:
            imbalance = min(1.0, abs(E_low) / mass_active)

        K_bal = float(parameters['K_bal'])
        low_balance = 1.0 / (1.0 + (imbalance / K_bal) ** 2.0)

        H_spread = float(parameters['H_spread'])
        if n_active >= 2:
            spread_norm = float(np.std(validities[active_below])) / scale
            homog = 1.0 / (1.0 + (spread_norm / H_spread) ** 4.0)
            signs_act = np.sign(diff[active_below])
            sign_agree = float(abs(np.mean(signs_act)))
        else:
            homog = 1.0
            sign_agree = 1.0

        if n_active >= 2:
            beta_eff = beta * (beta_multi_scale + (1.0 - beta_multi_scale) * homog * sign_agree)

        D_half = float(parameters['D_half'])
        dmax = float(np.max(dev[active_below]))
        dev_damp = 1.0 / (1.0 + (dmax / D_half) ** 2.0)

        h_pos = float(parameters['h_pos'])
        h_neg = float(parameters['h_neg'])
        neg_bal_floor = float(parameters['neg_bal_floor'])
        neg_dev_floor = float(parameters['neg_dev_floor'])
        n_small = float(parameters['n_small'])
        s_small = float(parameters['s_small'])
        n_large = float(parameters['n_large'])
        s_large = float(parameters['s_large'])

        z_small = (float(n_active) - n_small) / s_small
        small_win = float(np.exp(-z_small * z_small))

        z_large = (float(n_active) - n_large) / s_large
        z_large = float(np.clip(z_large, -50.0, 50.0))
        large_win = 1.0 / (1.0 + float(np.exp(-z_large)))

        conc_boost = 1.0 + conc_boost_gain * conc_act
        conc_mult = 1.0 + conc_boost_gain * conc_act

        neg_count_floor_low = float(parameters['neg_count_floor_low'])
        neg_count_floor_high = float(parameters['neg_count_floor_high'])
        neg_count_floor = neg_count_floor_low + (neg_count_floor_high - neg_count_floor_low) * standout
        neg_count_factor = neg_count_floor + (1.0 - neg_count_floor) * large_win

        strong_bal_floor = float(parameters['strong_bal_floor'])
        strong_dev_floor = float(parameters['strong_dev_floor'])
        eff_bal_floor = neg_bal_floor + (strong_bal_floor - neg_bal_floor) * standout
        eff_dev_floor = neg_dev_floor + (strong_dev_floor - neg_dev_floor) * standout
        neg_damp_bal = eff_bal_floor + (1.0 - eff_bal_floor) * low_balance
        neg_damp_dev = eff_dev_floor + (1.0 - eff_dev_floor) * dev_damp

        gate = (h_pos * small_win * homog * low_balance * dev_damp
                - h_neg * homog * conc_act * conc_mult * neg_count_factor
                * neg_damp_bal * neg_damp_dev)
        if n_active == 1:
            gate = gate * float(parameters['one_cue_gate_scale'])
        lower_bound = -h_neg * conc_boost
        gate = float(np.clip(gate, lower_bound, h_pos))

        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        branch_boost = float(parameters['branch_boost'])
        hi_boost = 1.0
        if n_active >= 2:
            hi_boost = 1.0 + branch_boost * small_win * homog * low_balance

        hi_pre = np.zeros(n_feat, dtype=float)
        gap_bounded = np.minimum(gap, 100.0)
        hi_pre[at_or_above] = rho * hi_boost * np.power(gap_bounded[at_or_above], alpha)

        raw_high = gate * hi_pre
        raw_all = raw + raw_high
        if n_active == 1:
            norm = float(np.sum(raw[below]))
        else:
            norm = float(np.sum(raw[below])) + float(np.sum(np.abs(raw_high[at_or_above])))
        if norm <= 1e-12:
            evidence = float(np.mean(diff))
        else:
            evidence = float(np.dot(raw_all, diff) / norm)

    logits = np.array([evidence, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    z = np.clip(z, -60.0, 60.0)
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
