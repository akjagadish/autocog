# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** People evaluate each advertised expert validity relative to the ensemble median, so that cues below the median acquire context-sensitive salience through a median-normalized deviance score. A rank-based baseline keeps higher-validity cues from being completely ignored, but only when the genuine low-validity outlier bloc is small; as the number of below-median outlier cues grows, this high-validity baseline is smoothly suppressed. Cue weights are capped by a smooth, low-bloc-size-dependent maximal share, so that no single cue can monopolize the decision while still allowing redistribution toward high-validity cues when several low-validity cues compete. Choice gain is set by smooth functions of low-outlier bloc size, validity gap, and effective number of competing cues.

**Rationale:** This is exactly the minimal cap correction requested in the latest feedback. The accepted iter-2 model used a fixed cap_share of 0.34, which sat at the critical value 1/3: with three low-validity cues raw low-cue weights remain just below the cap and no mass is redistributed to high-validity cues, while with two low outliers each cue clips and surplus mass artificially reconstructs high-validity dependence. I replaced the fixed cap with a smooth Gaussian function of low_bloc_frac, centered near the two-outlier fraction and clipped to [0.20, 0.50]. This preserves the near-0.34 singleton regime for Exp 10, raises the cap near the two-outlier fraction for Exp 8, and lowers the cap below 1/3 when the low-validity bloc is larger so surplus is forced onto high-validity cues in Exps 2 and 3. No other machinery, gates, or parameters were changed.

**Parameters:**
  - `c`: `[0.008, 0.014]`
  - `low_dev_threshold`: `{0.70}`
  - `rank_gate_center`: `{0.19}`
  - `rank_gate_width`: `{0.07}`
  - `w0`: `[0.005, 0.015]`
  - `kappa`: `[1.8, 2.2]`
  - `gamma`: `[1.3, 1.5]`
  - `b_scale`: `[0.30, 0.50]`
  - `b_base`: `[0.15, 0.25]`
  - `b_alpha`: `[1.3, 1.7]`
  - `gap_center`: `{0.38}`
  - `gap_width`: `{0.10}`
  - `cap_base`: `{0.26}`
  - `cap_peak`: `{0.44}`
  - `cap_center`: `{0.22}`
  - `cap_width`: `{0.10}`
  - `share_hi_center`: `{0.85}`
  - `share_hi_width`: `{0.02}`
  - `gap_hi_center`: `{0.40}`
  - `gap_hi_width`: `{0.06}`
  - `eff_n_center`: `{3.0}`
  - `eff_n_width`: `{0.14}`
  - `top3_sim_center`: `{0.95}`
  - `top3_sim_width`: `{0.07}`
  - `low_bloc_center`: `{0.48}`
  - `low_bloc_width`: `{0.10}`
  - `beta_base`: `[2.4, 2.8]`
  - `beta_floor_hi`: `[0.35, 0.50]`
  - `beta_boost_lo`: `[0.35, 0.60]`
  - `epsilon`: `[0.04, 0.06]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Bounded contextual salience expects a (2, n_features) stimulus.')

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

    deviance = np.maximum(0.0, median - validities) / scale

    low_dev_threshold = float(parameters['low_dev_threshold'])
    low_bloc_frac = float(np.mean(deviance > low_dev_threshold))

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    dev_weight = w0 + kappa * np.power(deviance, gamma)

    ranks = rankdata(validities, method='average')
    if n_feat > 1:
        u = (ranks - 1.0) / (n_feat - 1.0)
    else:
        u = np.array([0.5])

    b_scale = float(parameters['b_scale'])
    b_base = float(parameters['b_base'])
    b_alpha = float(parameters['b_alpha'])
    gap_center = float(parameters['gap_center'])
    gap_width = float(parameters['gap_width'])

    gap_norm = (float(np.max(validities)) - float(np.min(validities))) / (float(np.median(validities)) + c)
    gap_activ = 1.0 / (1.0 + np.exp(-float(np.clip((gap_norm - gap_center) / gap_width, -500.0, 500.0))))

    rank_gate_center = float(parameters['rank_gate_center'])
    rank_gate_width = float(parameters['rank_gate_width'])
    rank_gate = 1.0 / (1.0 + np.exp(float(np.clip((rank_gate_center - low_bloc_frac) / rank_gate_width, -500.0, 500.0))))
    rank_base = b_scale * (b_base + gap_activ) * np.power(u, b_alpha) * rank_gate

    raw = dev_weight + rank_base
    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    cap_base = float(parameters['cap_base'])
    cap_peak = float(parameters['cap_peak'])
    cap_center = float(parameters['cap_center'])
    cap_width = float(parameters['cap_width'])
    cap_share = cap_base + (cap_peak - cap_base) * np.exp(-0.5 * ((low_bloc_frac - cap_center) / cap_width) ** 2)
    cap_share = float(np.clip(cap_share, 0.20, 0.50))

    for _ in range(100):
        max_w = float(np.max(weights)) if weights.size else 0.0
        if max_w <= cap_share + 1e-12:
            break
        weights = np.minimum(weights, cap_share)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
            break
        weights = weights / total_w

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    top1 = float(sorted_w[0]) if sorted_w.size > 0 else 1.0
    top2 = float(sorted_w[1]) if sorted_w.size > 1 else top1
    top3 = float(sorted_w[2]) if sorted_w.size > 2 else (float(sorted_w[-1]) if sorted_w.size > 0 else 0.0)
    top3_sim = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights * weights)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    weight_gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * weight_gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    low_bloc_center = float(parameters['low_bloc_center'])
    low_bloc_width = float(parameters['low_bloc_width'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_sim - top3_sim_center) / top3_sim_width) ** 2)
    low_size_gate = np.exp(-0.5 * ((low_bloc_frac - low_bloc_center) / low_bloc_width) ** 2)
    low_gate = eff_gate * balance_gate * low_size_gate

    beta_base = float(parameters['beta_base'])
    beta_floor_hi = float(parameters['beta_floor_hi'])
    beta_boost_lo = float(parameters['beta_boost_lo'])
    beta_eff = beta_base + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.10, 12.0))

    epsilon = float(parameters['epsilon'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    e = np.exp(z)
    core = e / e.sum()

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

### `pi_8` → slot 1 (via `new_theory`)

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
