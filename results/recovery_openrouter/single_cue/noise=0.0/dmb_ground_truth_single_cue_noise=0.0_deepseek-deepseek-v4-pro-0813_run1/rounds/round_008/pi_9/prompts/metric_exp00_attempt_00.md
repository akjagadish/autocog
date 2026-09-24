# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=26):** [0.7999, 0.7999, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** Twenty-six experts are used. Experts 0 and 1 are at 0.7999, just below the 0.8000 ensemble median; experts 2 through 17 sit exactly at the median; experts 18 through 25 are at 0.9000, giving a real upper spread. Because the median and high experts are heavily represented, the MAD is 0 and the deviance scale is driven by the model c parameter, so the two low experts receive only a modest deviance weight rather than a near-monopoly. Every trial makes exactly one of the two low experts diagnostic while the other low expert is tied, so the diagnostic low-cue count is always 1. The critical manipulation sweeps k = 0, 2, 5, 8, 12, 16, 20, 24 median-or-above experts opposing the diagnostic low cue. In GVC-strict, the solo-low gate starts at zero, so all 24 median and 0.9000 experts receive exactly zero decision weight; the only weights are the amplified diagnostic low cue and the tied low cue. Its lone amplification includes the opposition multiplier that rises with k and caps at 5, so GVC-strict p(A) rises slightly from k = 0 to k = 5 and then stays flat. DASH instead gives the diagnostic low cue only a constant lone amplification, while the 24 high experts retain at least the w_hi0 baseline and the 0.9000 experts also receive a partly gated rank bonus. As k opposing high experts increases, DASH high-cue evidence becomes increasingly negative, so DASH p(A) falls across k and crosses below 0.50 around k = 16 to 20, reaching about 0.40 at k = 24. This creates an opposite-slope dissociation: GVC-strict rises and plateaus while DASH decreases and crosses chance. The two mirrored low-cue blocks control response bias, and each of the 16 unique pairs is repeated K = 6 times for 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Deviance-anchored, state-gated high-cue bonus with diagnostic low-cue attenuation (DASH). People anchor each advertised expert validity to the current ensemble median. Below-median cues earn weight from a median/MAD-normalized deviance score, but when two or more below-median cues are diagnostic on a trial their weights are attenuated by a count-dependent factor, preventing co-equal low-validity blocs from dominating choice. A solitary diagnostic below-median cue instead receives an amplified weight, capturing strong lone-cue following. Above-median cues receive only a tiny at-median baseline plus a rank-ordered bonus that is gated by current signed low-cue evidence and low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

**Parameters:**
- c: [0.008, 0.013]
- w0: [0.006, 0.015]
- kappa: [1.8, 2.4]
- gamma: [1.2, 1.5]
- lambda: [2.2, 2.8]
- sigma: [0.20, 0.30]
- phi: [0.25, 0.50]
- w_hi0: [0.0008, 0.0015]
- rho: [0.018, 0.028]
- alpha: [2.5, 3.5]
- a: [0.22, 0.38]
- K: [0.35, 0.55]
- beta: [1.8, 2.5]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## COMPETING THEORY
**Description:** Conflict-gated validity-gap compensation with strict homogeneous-bloc licensing and stronger opposition-sensitive lone amplification (GVC-strict). People anchor each advertised expert validity to the ensemble median. Below-median cues earn decision weight through a median/MAD-normalized deviance score. A single diagnostically deviant low cue is strongly amplified, with amplification increasing in the number of opposed at-or-above-median cues. When multiple below-median cues are diagnostic, a homogeneous low-deviant group is attenuated by cue count, but an extreme focal low cue is exempted from that group attenuation so it can still dominate the signed evidence. At-or-above-median cues receive no default weight; their validity-gap bonus is licensed only by a very homogeneous, sufficiently deviant bloc of three or more simultaneously diagnostic below-median cues, and is sharply damped by low-versus-high conflict. Choice is a fixed-temperature softmax with no lapse floor.

**Parameters:**
- c: [0.008, 0.012]
- w0: [0.007, 0.012]
- kappa: [1.9, 2.3]
- gamma: [1.25, 1.45]
- lambda: [4.5, 5.5]
- sigma: [0.10, 0.18]
- phi: [0.55, 0.75]
- nu: [0.15, 0.25]
- opp_cap: {5}
- focal_ratio: [3.0, 4.0]
- gate_base: {0.0}
- lone_floor: [0.00, 0.03]
- D_half: [0.35, 0.55]
- q_dev: {2}
- b_boost: [0.02, 0.08]
- D_boost: [0.6, 1.2]
- H_spread: [0.005, 0.02]
- K: [0.30, 0.50]
- w_hi0: {0.0}
- rho: [0.08, 0.14]
- alpha: [1.1, 1.8]
- beta: [1.4, 1.8]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    a = np.array(data['option_a_ratings'].tolist(), dtype=float)\n    b = np.array(data['option_b_ratings'].tolist(), dtype=float)\n\n    low_side = (b[:, 0] > a[:, 0]).astype(float)\n    resp = data['response'].to_numpy(dtype=float)\n    agree_low = (resp == low_side).astype(float)\n\n    k = (a[:, 2:] != b[:, 2:]).sum(axis=1).astype(float)\n\n    frame = data[['subject_id']].copy()\n    frame['agree_low'] = agree_low\n    frame['k'] = k\n\n    slopes = []\n    for sid, sub in frame.groupby('subject_id'):\n        x = sub['k'].to_numpy(dtype=float)\n        y = sub['agree_low'].to_numpy(dtype=float)\n        xm = x.mean()\n        ym = y.mean()\n        denom = np.sum((x - xm) ** 2)\n        if denom <= 1e-12:\n            slopes.append(0.0)\n        else:\n            slopes.append(float(np.sum((x - xm) * (y - ym)) / denom))\n\n    return float(np.mean(slopes)) if slopes else float('nan')",
  "rationale": "Each trial places exactly one below-median expert (feature index 0) on one side while the other below-median expert is tied, and k = 0, 2, 5, 8, 12, 16, 20, 24 median-or-above experts oppose it. The metric converts each choice into low-cue agreement, coded 1 when the subject chose the side carrying the diagnostic low cue and 0 otherwise, and estimates the within-subject ordinary least squares slope of that agreement against k. DASH predicts a reliably negative slope because accumulating median and high-cue evidence increasingly overcomes the lone low cue, crossing chance at high k. GVC-strict predicts a zero or slightly positive slope because high cues are gated off in the solo-low case and the lone low cue's amplification rises with opposition up to its cap. Averaging per-subject slopes over all 96 trials provides a stable scalar with low between-subject variance and separates the two accounts."
}
```

## Usage

```json
{
  "input_tokens": 8859,
  "output_tokens": 26597,
  "reasoning_tokens": 26038,
  "provider": "Baidu"
}
```
