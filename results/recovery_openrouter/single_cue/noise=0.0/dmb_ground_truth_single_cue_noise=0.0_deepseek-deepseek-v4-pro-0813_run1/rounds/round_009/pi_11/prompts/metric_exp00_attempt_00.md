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
**Validities (n_features=12):** [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.85, 0.92, 0.99]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** This design uses a no-below-median ensemble to force a fallback dissociation between GVC-strict and DSA-SFCC. Eight experts are advertised at 0.60; because they form 8/12 of the ensemble, the median validity is 0.60 and no expert lies below the median. Four experts are above the median (0.80, 0.85, 0.92, 0.99). Under GVC-strict, with no diagnostic below-median cue, the high-cue gate remains at its zero base, all raw weights are zero, and the model's fallback assigns equal weight to all 12 experts. GVC-strict therefore reduces to tallying: p(A) follows the signed tally margin across all experts. Under DSA-SFCC, the seven/eight median experts receive zero decision weight, while the above-median experts receive the rank-ordered consensus bonus. With no low evidence, low_bal=0 and low_weak=1, so the sign-flexible high gate is g = h_pos > 0, and DSA-SFCC follows the above-median high block, especially its top expert (0.99), which carries about 75-86% of the high-block weight. The eight selected pairs always put the eight median experts in opposition to the top high expert. In Block A the median experts favor A and the top high expert favors B for k=0..3 low-to-high sweeps: GVC-strict predicts p(A) > 0.5 (tally margins +4, +6, +8, +10), while DSA-SFCC predicts p(A) < 0.5. In Block B the median experts favor B and the top high expert favors A for k=1..4 high-to-low sweeps: GVC-strict predicts p(A) < 0.5 (margins -10, -8, -6, -4), while DSA-SFCC predicts p(A) > 0.5. Thus every included pair gives opposite directional predictions, and the k sweep also separates the smooth tally-margin psychometric function from the top-high-expert-driven function.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Median anchoring of advertised validities gives below-median cues MAD-normalized deviance weight. Lone diagnostic low cues receive strong amplification, while multiple diagnostic low cues receive count-based attenuation that is deviance-selective: an extreme below-median singleton is partially exempt from attenuation without demoting inactive low cues. Above-median cues contribute through a sign-flexible rank-ordered consensus term gated by the signed evidence carried by actually diagnostic low cues. Balanced low evidence permits a negative consensus weight, opposing low evidence gates the high-cue term toward zero, and weak low evidence permits only a small capped positive rank bonus. Large comparable multi-low coalitions receive a linear soft attenuation of their total active low evidence to prevent aligned low blocs from over-saturating choice. Choice is a fixed-temperature softmax with a tiny lapse floor.

**Parameters:**
- c: [0.008, 0.012]
- w0: [0.007, 0.012]
- kappa: [1.9, 2.3]
- gamma: [1.25, 1.45]
- lone_gain: [5.5, 6.5]
- lone_sigma: [0.10, 0.18]
- phi: [1.2, 1.5]
- nu: [0.15, 0.25]
- opp_cap: {5}
- out_gain: [0.50, 0.80]
- out_thresh: [2.2, 2.8]
- out_slope: [3.5, 5.5]
- cap_out_gate: {0.45}
- n_cap_min: {4}
- low_multi_scale: [0.60, 0.80]
- rho_hi: [0.025, 0.035]
- alpha_hi: [3.0, 4.5]
- h_cap: [0.06, 0.12]
- h_pos: [0.15, 0.25]
- h_neg: [1.30, 1.55]
- min_low_mass: {0.10}
- low_bal_eps: {0.02}
- weak_thresh: [0.12, 0.20]
- n_min_hi: {2}
- cons_thresh: [0.82, 0.88]
- cons_s: {0.04}
- beta: [1.85, 2.20]
- epsilon: {0.001}
- validities: validities

**`predict source code`:**
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    if data.empty:\n        return 0.0\n    top_a = data['option_a_ratings'].apply(lambda x: int(x[-1]))\n    top_b = data['option_b_ratings'].apply(lambda x: int(x[-1]))\n    top_favors_a = (top_a > top_b).astype(int)\n    chose_a = (data['response'] == 0).astype(int)\n    congruent = (top_favors_a == chose_a).astype(int)\n    return float(2.0 * congruent.mean() - 1.0)\n",
  "rationale": "The last feature is the 0.99-validity expert. In every designed pair this top expert opposes the eight 0.60 median experts. Define a trial score as +1 when the subject chooses the option favored by the top expert and -1 otherwise. The metric is the mean of this score, which equals (proportion choosing A when the top expert favors A) minus (proportion choosing A when the top expert favors B). GVC-strict falls back to equal weighting of all experts when no below-median cue exists, so choices follow the eight median experts and this top-expert congruence score is strongly negative. DSA-SFCC gives the median experts zero weight and gives the highest validity expert an outsize rank-ordered high-block weight, so the score is positive. The mean of 96 bounded +/-1 indicators per subject also keeps between-subject variance small, making the contrast reliable."
}
```

## Usage

```json
{
  "input_tokens": 7787,
  "output_tokens": 14801,
  "reasoning_tokens": 14457,
  "provider": "Baidu"
}
```
