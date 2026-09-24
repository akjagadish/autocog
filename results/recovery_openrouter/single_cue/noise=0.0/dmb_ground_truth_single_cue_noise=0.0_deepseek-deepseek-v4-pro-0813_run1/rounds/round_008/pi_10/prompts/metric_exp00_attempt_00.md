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
**Validities (n_features=15):** [0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.8, 0.8, 0.8, 0.8, 0.86, 0.88, 0.9, 0.95, 0.99]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 11: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 12: A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** This design isolates a high-cue bonus dissociation between GVC-strict and DASH. Six identical below-median experts at 0.7985 are diagnostic on every trial and are split 3 for A and 3 for B, so signed low-cue evidence is exactly zero. Four median experts at 0.800 are tied on every trial. Five above-median experts (0.860, 0.880, 0.900, 0.950, 0.990) are swept from all favoring B to all favoring A (k = 0..5). In GVC-strict, the homogeneous bloc of six simultaneously diagnostic below-median cues licenses a validity-gap bonus for the above-median cues, and conflict damping is absent because low-cue evidence is zero. The signed high-cue evidence therefore moves GVC-strict p(A) monotonically from clearly below 0.5 at k = 0 to clearly above 0.5 at k = 5, crossing near k = 3/4. In DASH, the zero low-cue evidence makes its low_gate exactly zero, so the rank bonus on above-median cues is fully gated off; only the tiny w_hi0 baseline remains, producing an essentially flat p(A) near 0.50 at every k. The 12 pairs include the six high-sweep pairs and their A/B mirrors to double observations and control response bias; each unique pair is repeated K = 8 times for roughly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "import numpy as np\n\nVALIDITIES = np.array([0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.8, 0.8, 0.8, 0.8, 0.86, 0.88, 0.9, 0.95, 0.99])\nMEDIAN = float(np.median(VALIDITIES))\nHIGH_MASK = VALIDITIES > MEDIAN + 1e-12\nHIGH_WEIGHTS = VALIDITIES - MEDIAN\n\ndef metric(data):\n    a = data['option_a_ratings'].tolist()\n    b = data['option_b_ratings'].tolist()\n    r = data['response'].to_numpy()\n    match = []\n    for ai, bi, ri in zip(a, b, r):\n        ai_arr = np.asarray(ai, dtype=float)\n        bi_arr = np.asarray(bi, dtype=float)\n        ev = float(np.dot(HIGH_WEIGHTS[HIGH_MASK], ai_arr[HIGH_MASK] - bi_arr[HIGH_MASK]))\n        if ev > 1e-12:\n            rec = 0\n        elif ev < -1e-12:\n            rec = 1\n        else:\n            continue\n        match.append(1.0 if ri == rec else 0.0)\n    if not match:\n        return float('nan')\n    return float(np.mean(match))",
  "rationale": "This metric measures how often the subject chooses the option supported by the five above-median-validity experts, weighted by how far each expert's advertised accuracy exceeds the median. In GVC-strict, these high-validity cues receive a licensed validity-gap bonus and drive choices across the high-sweep, so adherence is clearly above chance. In DASH, the signed low-cue evidence is exactly zero because the six below-median cues are balanced, so its low gate is zero and the high-cue bonus is fully gated off; only a tiny baseline weight remains, keeping adherence near 0.50. Because every trial contributes a match or mismatch and each subject completes 96 trials, the per-subject estimate is stable."
}
```

## Usage

```json
{
  "input_tokens": 7248,
  "output_tokens": 21738,
  "reasoning_tokens": 21216,
  "provider": "Baidu"
}
```
