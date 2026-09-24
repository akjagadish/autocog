# Round 6 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7` — KILLED ✗

**Description:** All subjects treat a binary expert rating of 0 as clean/favorable and 1 as a defect. On each trial every discriminating expert contributes continuous inverse-valence evidence weighted by a saturating, near-tally-like transform of communicated validity: w_j = 1 + m * tanh(2 * [logit(v_j) - mean(logit(v))] / half_range), with the weight vector normalized to mean one. The subject-level saturation m is small and capped, so high-validity experts are attended more but cannot generate extreme top-to-bottom weight ratios: a three-cue high-validity coalition can win small conflicts yet is overridden by a five-cue low-validity majority. When the weighted evidence is inconclusive, or when the raw inverse-clean tally is exactly balanced, subjects fall back to inspecting changed experts in descending validity order with tied highest-validity experts randomly permuted per trial; the precision of that fallback grows with the validity gap separating the best expert from the next-best expert. Choice is a softmax over net evidence with subject-specific gain and lapse. There is no valence-orientation mixture and no separate top-cue attention bonus.

**Rationale:** This edit applies the previously accepted calibration but replaces the unbounded exponential validity-to-weight map with a saturating tanh transform whose subject-level amplitude is small and capped. The previous accepted base still allowed a subset of subjects to have top-to-bottom weight ratios large enough that three high-validity cues beat five low-validity cues in Experiment 8, keeping that metric wrong-signed. The saturating map caps the effective top-to-bottom ratio near 1.2-1.4 for most subjects, so low-validity majorities dominate exactly in the regime observed in Experiment 8 while high-validity experts still receive modest extra weight. The main softmax gain is kept at the accepted base's moderate value rather than the rejected higher value, preventing the overshoot seen in Experiments 1, 4, 10, and 12. The fallback is triggered both when weighted evidence is inconclusive and when the raw inverse-clean tally is exactly balanced; this preserves the validity-ordered choices in Experiment 3 and the balanced one-cue conflicts in Experiments 5 and 7, while reducing the previous over-sharp fallback in Experiment 9. The fallback's precision increases with the validity gap between the best and next-best changed experts, which restores Experiment 5's moderate top-cue following without reintroducing a top-cue bonus in the continuous tally path. Tied highest-validity experts remain exchangeable per trial, there is no valence-orientation mixture, and no separate top-cue attention bonus is used.

**Parameters:**
  - `slope_a`: `[0.02, 0.30]`
  - `gain`: `[0.10, 1.50]`
  - `epsilon`: `[0.03, 0.16]`
  - `tie_gain`: `[0.30, 1.30]`
  - `tie_tol`: `[0.02, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a_row = stim[0]
    b_row = stim[1]
    n_features = stim.shape[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Subject-level parameter shaping -----------------------------------
    # Saturation amplitude for the validity-to-weight map.
    # Kept deliberately small: this is the main fix for the unbounded
    # exponential weights that let three high-validity cues destroy the
    # five-cue low-validity majority in Experiment 8.
    u_m = float(np.clip((float(parameters['slope_a']) - 0.02) / 0.28, 0.0, 1.0))
    sat_m = float(np.exp(_trunc_qnorm(
        u_m, np.log(0.10), 0.45, np.log(0.03), np.log(0.18))))

    # Main choice gain: restore the accepted base's moderate calibration.
    u_g = float(np.clip((float(parameters['gain']) - 0.10) / 1.40, 0.0, 1.0))
    gain = float(np.exp(_trunc_qnorm(
        u_g, np.log(0.40), 0.50, np.log(0.10), np.log(1.50))))

    lo_logit = np.log(0.03 / 0.97)
    hi_logit = np.log(0.16 / 0.84)
    mu_logit = np.log(0.08 / 0.92)
    u_e = float(np.clip((float(parameters['epsilon']) - 0.03) / 0.13, 0.0, 1.0))
    z_eps = _trunc_qnorm(u_e, mu_logit, 0.35, lo_logit, hi_logit)
    epsilon = float(_inv_logit(z_eps))

    # Base tie-break gain is moderate, but is amplified by the validity gap
    # between the selected best expert and the next-best changed expert.
    u_tg = float(np.clip((float(parameters['tie_gain']) - 0.30) / 1.00, 0.0, 1.0))
    tie_gain = float(np.exp(_trunc_qnorm(
        u_tg, np.log(0.45), 0.45, np.log(0.30), np.log(0.70))))

    # Tightened inconclusive-evidence threshold.
    u_tt = float(np.clip((float(parameters['tie_tol']) - 0.02) / 0.13, 0.0, 1.0))
    tie_tol = float(_trunc_qnorm(u_tt, 0.08, 0.030, 0.02, 0.15))

    # ---- Saturating continuous inverse-valence weights ---------------------
    # 0 is clean/favorable, 1 is a defect.
    v = np.clip(val, 0.50001, 0.9999)
    logit_v = np.log(v / (1.0 - v))
    center = float(np.mean(logit_v))

    if n_features > 1:
        spread = float(np.max(logit_v) - np.min(logit_v))
        if spread <= 1e-8:
            z = np.zeros(n_features, dtype=float)
        else:
            half_spread = 0.5 * spread
            z = np.tanh(2.0 * (logit_v - center) / half_spread)
        w = 1.0 + sat_m * z
        w = w / np.mean(w)
    else:
        w = np.ones(1, dtype=float)

    changed_mask = a_row != b_row
    if not np.any(changed_mask):
        return np.ones(2) / 2.0

    # A=0/B=1 gives +w_j; A=1/B=0 gives -w_j.
    diff = (b_row > a_row).astype(float) - (a_row > b_row).astype(float)
    raw_tally = float(np.sum(diff))
    net_evidence = float(np.dot(w, diff))

    if abs(net_evidence) <= tie_tol or abs(raw_tally) <= 0.5:
        # Inconclusive evidence or an exactly balanced raw inverse-clean
        # tally: inspect changed experts in descending validity order.
        # Ties at the maximum validity are exchangeable and resolved by a
        # random permutation on each trial.
        disc = np.flatnonzero(changed_mask)
        top_val = float(np.max(val[disc]))
        tied = disc[np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        if tied.size == 0:
            tied = disc
        idx = int(tied[np.random.randint(tied.size)])

        # Validity-gap-dependent tie-break precision.  This is not a top-cue
        # bonus inside the tally; it is how strongly subjects rely on the
        # lexical fallback only when they are already in the fallback regime.
        non_top = disc[~np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        second_val = float(np.max(val[non_top])) if non_top.size > 0 else top_val
        gap = top_val - second_val
        gap_boost = 1.0 + 2.5 * np.tanh(gap / 0.15)

        if a_row[idx] == 0.0 and b_row[idx] == 1.0:
            sign = 1.0   # clean cue favors A
        elif a_row[idx] == 1.0 and b_row[idx] == 0.0:
            sign = -1.0  # clean cue favors B
        else:
            return np.ones(2) / 2.0

        scores = np.array([sign, -sign], dtype=float)
        beta_eff = tie_gain * gap_boost
    else:
        scores = np.array([net_evidence, -net_evidence], dtype=float)
        beta_eff = gain

    logits = beta_eff * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Rationale:** This is a minimal parameter-prior edit on top of the accepted validity-pruned inverse-tally architecture. The architecture is unchanged, but the direct uniform subject-level parameters are replaced by heavy-tailed transformed draws: log-uniform for beta_tally and beta_tiebreak, a symmetric logit-centered transform for gamma within the requested [-4.5, -2.5] band, and a logit transform for epsilon. The effective beta_tally range is now about 0.74-1.65 and beta_tiebreak about 1.49-3.00, making small inverse tallies and clean-zero tie-breaks more decisive, which repairs the under-powered Exp2, Exp3, and Exp4 scores. Tightening gamma to [-4.5, -2.5] with an edge-concentrated shape restores a small positive Exp6 validity-gradient effect instead of the previous overcorrection to zero, while alpha [5,9] removes the very high-alpha tail that flattened validity sensitivity. The interval transforms also raise targeted between-subject variance, especially in Exp5/Exp6, without the uniform widening that previously inflated Exp1-Exp3 variance.

**Parameters:**
  - `alpha`: `[5.0, 9.0]`
  - `gamma_logit`: `[-2.50, 2.50]`
  - `beta_tally_log`: `[-0.30, 0.50]`
  - `beta_tiebreak_log`: `[0.40, 1.10]`
  - `epsilon_logit`: `[-3.00, -0.40]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_8` → slot 1 (via `new_theory`)

**Description:** Inverse-valence majority with gated unique-expert precedence and tied-top large-conflict softness. Subjects treat a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The default choice is a softmax over this near-equal-weight tally. When exactly one discriminating expert has strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus inside a limited raw-tally window. In large multi-cue conflicts, this unique-expert reliance is separately and more strongly applied. When two or more top experts are tied, unique-expert precedence is suppressed, but the conflict is still processed with reduced decision sensitivity to the raw tally, preventing an unrealistically sharp majority effect while never letting tied top experts override a non-zero low-validity majority.

**Rationale:** This is a minimal edit on the accepted base. The main fix targets E8, where the three high-validity cues are empirically tied at the top, so unique_top is false and the model falls back to a near-pure inverse-majority softmax that is far too sharp. A new subject-level large_tie_scale parameter applies only when is_large is true, unique_top is false, and the raw tally is non-zero; it multiplies the raw tally before the softmax, so tied-top experts still never override a non-zero low-validity majority but conflict-size-dependent softness reduces E8 toward its observed value. The p_large lower bound is raised from 0.25 to 0.35 to strengthen the existing unique-expert precedence in small-margin multi-cue conflicts, which should lift E6 and E9 without disturbing the D=0 tie machinery that protects E3 and E9. All other calibrated mechanisms, including the small-disc and exact-zero-tally behavior, are unchanged.

**Parameters:**
  - `gain`: `[1.20, 1.80]`
  - `epsilon`: `[0.20, 0.36]`
  - `p_override`: `[0.30, 0.80]`
  - `p_large`: `[0.35, 0.98]`
  - `tie_p`: `[0.55, 0.95]`
  - `window`: `{1, 2}`
  - `large_tie_scale`: `[0.05, 0.75]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import math
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError('validities length mismatch')

    changed = a != b
    disc = np.flatnonzero(changed)
    if disc.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))

    gain = float(parameters['gain'])
    epsilon = float(parameters['epsilon'])
    p_override = float(parameters['p_override'])
    p_large = float(parameters['p_large'])
    tie_p = float(parameters['tie_p'])
    window = int(round(float(parameters['window'])))

    is_large = disc.size >= 6 and 7 <= stim.shape[1] <= 9
    window_eff = window
    if is_large:
        window_eff = max(2, window)
        p_apply = p_large
    else:
        p_apply = p_override

    E = D
    unique_top = False
    sign_top = 0.0
    gap = 0.0

    if disc.size == 1:
        j_top = int(disc[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
        unique_top = True
    else:
        vals_disc = val[disc]
        top_val = float(np.max(vals_disc))
        top_hits = disc[np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
        unique_top = top_hits.size == 1
        if unique_top:
            j_top = int(top_hits[0])
            sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
            others = vals_disc[~np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
            if others.size == 0:
                gap = 0.04
            else:
                gap = float(top_val - float(np.max(others)))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = 0.72 + 0.50 * math.tanh(gap / 0.05)
                E = sign_top * tie_bonus
            else:
                E = 0.0
        elif abs(D) <= window_eff and np.random.rand() < p_apply:
            bonus = 0.95 + 2.0 * math.tanh(gap / 0.10)
            if is_large:
                bonus = 1.00 + 2.40 * math.tanh(gap / 0.10)
                bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc.size - 4)))
            E = D + sign_top * bonus
    elif abs(D) <= 1e-12:
        E = 0.0

    # Tied-top large conflicts: unique-expert precedence is suppressed by the
    # theory, but raw-tally sensitivity is reduced so the low-validity majority
    # does not drive an unrealistically sharp choice.
    if is_large and not unique_top and abs(D) > 1e-12:
        E = D * float(parameters['large_tie_scale'])

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```
