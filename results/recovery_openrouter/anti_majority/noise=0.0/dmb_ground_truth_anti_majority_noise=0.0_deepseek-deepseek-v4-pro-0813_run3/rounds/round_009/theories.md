# Round 9 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_10` — SURVIVED ✓

**Description:** Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive amplified validity weighting. When highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction, but the component is capped at |D|=1 and sharply suppressed for three-cue top groups facing five opposing lower-validity cues, so lower-validity majorities can override the tied top group in large conflicts. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts and is further damped in six-feature full conflicts with |D|>=2. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.

**Rationale:** This is a minimal-diff edit of the running-best candidate. It re-applies the iter-7 pieces that were explicitly credited with helping: df_t lowered to 1.6 for heavier tails; the large unique-top validity-weight trigger extended from disc_size>=6 to disc_size>=4 for n_features>=7; an 8-feature D=0 precedence boost of 1.03; and the m=2 tied-coalition multiplier raised to 2.70. It implements the newly prescribed midpoint for m=3 tied coalitions: the bonus is active for lower_count 3 and 4 (D=0 and |D|=1) but sharply attenuated at lower_count=5 with a steep |D| gate, and the global tied-top d_factor is capped at tanh(min(|D|,1.0)) so the bonus never grows with conflict. These changes should move Experiment 8 from negative toward positive and preserve Experiment 18 near its real trend. The unchosen iter-6/iter-7 extremes are avoided. Sparse six-feature unique-top precedence receives a 1.45 boost to recover Experiment 5, while six-feature full conflicts with |D|>=2 get stronger damping (0.40 vs 0.50) to cool Experiment 4. The lam_large_scale median is raised from 8 to 12 with wider bounds to push Experiment 6 across zero. All edits remain inside the prescribed inverse-valence clean-cue plus softmax family.

**Parameters:**
  - `gain_raw`: `[0.02, 0.98]`
  - `epsilon_raw`: `[0.02, 0.98]`
  - `lambda_raw`: `[0.02, 0.98]`
  - `prec_p_raw`: `[0.02, 0.98]`
  - `prec_large_p_raw`: `[0.02, 0.98]`
  - `tie_p_raw`: `[0.02, 0.98]`
  - `prec_strength_raw`: `[0.02, 0.98]`
  - `opp_decay_raw`: `[0.02, 0.98]`
  - `prec_disc_raw`: `[0.02, 0.98]`
  - `tie_strength_raw`: `[0.02, 0.98]`
  - `tie_gap_amp_raw`: `[0.02, 0.98]`
  - `tie_conflict_amp_raw`: `[0.02, 0.98]`
  - `tie_d0_raw`: `[0.02, 0.98]`
  - `lam_large_scale_raw`: `[0.02, 0.98]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import math
import numpy as np
from scipy.stats import t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    df_t = 1.6

    gain = float(np.exp(_tqt(u_raw(parameters['gain_raw']),
        math.log(1.45), 0.60, math.log(0.45), math.log(4.00), df_t)))
    epsilon = float(_ilogit(_tqt(u_raw(parameters['epsilon_raw']),
        _logit(0.26), 0.70, _logit(0.03), _logit(0.65), df_t)))
    lam = float(np.exp(_tqt(u_raw(parameters['lambda_raw']),
        math.log(0.22), 0.75, math.log(0.002), math.log(0.60), df_t)))
    p_prec = float(_ilogit(_tqt(u_raw(parameters['prec_p_raw']),
        _logit(0.50), 0.80, _logit(0.04), _logit(0.99), df_t)))
    p_large = float(_ilogit(_tqt(u_raw(parameters['prec_large_p_raw']),
        _logit(0.65), 0.75, _logit(0.08), _logit(0.995), df_t)))
    tie_p = float(_ilogit(_tqt(u_raw(parameters['tie_p_raw']),
        _logit(0.82), 0.70, _logit(0.20), _logit(0.99), df_t)))
    prec_str = float(np.exp(_tqt(u_raw(parameters['prec_strength_raw']),
        math.log(0.75), 1.00, math.log(0.10), math.log(2.50), df_t)))
    opp_decay = float(np.exp(_tqt(u_raw(parameters['opp_decay_raw']),
        math.log(0.90), 0.80, math.log(0.08), math.log(3.50), df_t)))
    prec_disc = float(np.exp(_tqt(u_raw(parameters['prec_disc_raw']),
        math.log(0.55), 0.70, math.log(0.08), math.log(2.80), df_t)))
    tie_str = float(np.exp(_tqt(u_raw(parameters['tie_strength_raw']),
        math.log(0.55), 1.60, math.log(0.03), math.log(12.00), df_t)))
    tie_gap_amp = float(np.exp(_tqt(u_raw(parameters['tie_gap_amp_raw']),
        math.log(0.80), 0.80, math.log(0.20), math.log(2.50), df_t)))
    tie_conflict_att = float(np.exp(_tqt(u_raw(parameters['tie_conflict_amp_raw']),
        math.log(0.85), 0.90, math.log(0.05), math.log(4.00), df_t)))
    tie_d0 = float(_ilogit(_tqt(u_raw(parameters['tie_d0_raw']),
        _logit(0.78), 0.70, _logit(0.20), _logit(0.95), df_t)))
    lam_large_scale = float(np.exp(_tqt(u_raw(parameters['lam_large_scale_raw']),
        math.log(12.0), 1.80, math.log(0.3), math.log(35.0), df_t)))

    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 4 and n_features >= 7
    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9
    lam_eff = lam
    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    unique_top = False
    top_tied = False
    top_mask = None
    top_idx = np.array([], dtype=int)
    sign_top = 0.0
    gap = 0.04
    top_clean = 0.0

    if disc_size == 1:
        unique_top = True
        j_top = int(disc_idx[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
    else:
        top_val = float(np.max(vals_disc))
        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
        top_idx = disc_idx[top_mask]
        if top_idx.size == 1:
            unique_top = True
            top_pos = int(np.flatnonzero(top_mask)[0])
            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04
        else:
            top_tied = True
            top_clean = float(np.sum(diff[disc_idx[top_mask]]))
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04

    if is_large_weight and unique_top:
        lam_eff = lam * lam_large_scale
        w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
        E = float(np.sum(w_disc * diff[disc_idx]))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))
                if n_features == 8:
                    tie_bonus *= 1.03
                E = sign_top * tie_bonus
        else:
            if top_mask is not None:
                opp = float(np.sum(diff[disc_idx[~top_mask]] * sign_top < 0.0))
            else:
                opp = 0.0
            dense_conflict = (disc_size == n_features) or (n_features >= 7 and disc_size >= n_features - 1)
            if dense_conflict:
                size_penalty = 1.0 / (1.0 + prec_disc * max(0.0, float(disc_size - 3)))
            else:
                size_penalty = 1.0
            decay = size_penalty / (1.0 + opp_decay * opp)
            p_apply = p_large if is_large_scale else p_prec
            if np.random.rand() < p_apply:
                prec_str_eff = prec_str
                if n_features == 6:
                    if disc_size == n_features and abs(D) >= 2.0:
                        prec_str_eff = prec_str * 0.40
                    elif disc_size >= n_features - 1:
                        prec_str_eff = prec_str * 0.78
                if n_features == 6 and disc_size <= 4:
                    prec_str_eff *= 1.45
                if is_large_scale:
                    gap_factor = 1.00 + 2.40 * math.tanh(gap / 0.10)
                    bonus = prec_str_eff * gap_factor * decay
                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))
                else:
                    gap_factor = 0.95 + 2.00 * math.tanh(gap / 0.10)
                    bonus = prec_str_eff * gap_factor * decay
                E = E + sign_top * bonus
    elif top_tied:
        m = int(top_idx.size)
        lower_count = int(disc_size - m)

        if m >= 4:
            w_disc_tied = 1.0 + lam * 0.85 * (vals_disc - mean_v)
            E = float(np.sum(w_disc_tied * diff[disc_idx]))
            if m >= 5 and lower_count > m:
                E = D

        if abs(top_clean) > 1e-9:
            coalition_ev = m * math.tanh(top_clean / float(max(1, m)))
            gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.12)

            if m >= 5:
                over = float(max(0.0, lower_count - m + 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * (over + 2.0 * over * over))
            elif m == 4:
                over = float(max(0.0, lower_count - m + 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)
            else:
                over = float(max(0.0, lower_count - m - 1.0))
                conflict_factor = 1.0 / (1.0 + tie_conflict_att * over)

            d_factor = tie_d0 + (1.0 - tie_d0) * math.tanh(min(abs(D), 1.0))

            tie_str_eff = tie_str
            if m == 2:
                tie_str_eff *= 2.70
            elif m == 3:
                if lower_count <= 4:
                    if lower_count == 3:
                        tie_str_eff *= 2.70
                    else:
                        tie_str_eff *= 2.20
                else:
                    steep_gate = 1.0 / (1.0 + math.exp((abs(D) - 1.5) / 0.2))
                    tie_str_eff *= steep_gate
            elif m >= 4:
                tie_str_eff *= 0.85

            if m >= 5 and lower_count > m and abs(D) <= 1.5:
                tie_str_eff *= 0.55

            E = E + tie_str_eff * coalition_ev * gap_factor * conflict_factor * d_factor

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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


### slot 2 — `pi_9` — KILLED ✗

**Description:** Subjects treat a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so each discriminating expert contributes an inverse-valence direction. The final evidence is a damped validity-weighted tally: each discriminating cue is weighted by 1 + lambda*(v_j - mean(v)), with lambda modest and further reduced in large multi-cue conflicts, so raw clean-cue majority dominates while high-validity cues receive a moderate extra weight. When exactly one discriminating expert has strictly highest communicated validity, a probabilistic, validity-gap-scaled precedence bonus is added to that expert's direction. When top experts are tied and the weighted evidence is near zero, subjects follow the sign of the tied top group's clean-cue advantage with moderate sensitivity instead of guessing. Choice is a softmax over the resulting evidence with subject-specific gain and lapse. Subject-level gain, lambda, precedence probability, large-conflict raw-tally softening, tie parameters, and epsilon are drawn from Student-t heavy-tailed/logit-Student-t distributions with medians anchored near the accepted base, providing the between-subject variance observed in the data without shifting the aggregate pattern too far.

**Rationale:** This is a targeted, in-family calibration edit of the accepted iter-1 base, following the most recent critique. It takes a smaller step than the rejected iter-2 candidate: lambda is only partially increased, with a Student-t median near 0.24 rather than iter-2's stronger value, so Exp15/Exp2/Exp13 improve without over-driving Exp6/Exp9. Gain and epsilon are kept exactly at the iter-1 values, avoiding the global overshoot pattern seen in Exp3/Exp5/Exp7/Exp9/Exp12. The tied-top clean-cue rule is moved to the midpoint suggested by the critic: tie_strength median 0.60, tanh denominator 2.8, and tie_tol median 0.60, which should raise Exp16 from 0.174 toward the real 0.249 without reaching the rejected candidate's 0.403. large_d_scale is nudged from 0.45 to 0.50 to help the large tied-top conflict (Exp8) while staying conservative. Finally, all subject-level parameters are switched from truncated-normal/logit-normal quantile transforms to truncated Student-t quantile transforms with df=4, keeping the same central medians but producing genuinely heavier tails. This targets the large between-subject variance gaps in Exp6/Exp8/Exp14/Exp16 while anchoring aggregate point estimates near the accepted base.

**Parameters:**
  - `gain_raw`: `[0.02, 0.98]`
  - `epsilon_raw`: `[0.02, 0.98]`
  - `lambda_raw`: `[0.02, 0.98]`
  - `prec_p_raw`: `[0.02, 0.98]`
  - `prec_large_p_raw`: `[0.02, 0.98]`
  - `tie_p_raw`: `[0.02, 0.98]`
  - `prec_strength_raw`: `[0.02, 0.98]`
  - `tie_strength_raw`: `[0.02, 0.98]`
  - `tie_tol_raw`: `[0.02, 0.98]`
  - `lam_large_scale_raw`: `[0.02, 0.98]`
  - `large_d_scale_raw`: `[0.02, 0.98]`
  - `window`: `{1, 2}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import math
import numpy as np
from scipy.stats import norm, t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Heavy-tailed subject-level transforms -------------------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.45), 0.32, math.log(0.60), math.log(2.80), 4)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.26), 0.35, _logit(0.07), _logit(0.45), 4)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.55, math.log(0.005), math.log(0.40), 4)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.55), 0.50, _logit(0.12), _logit(0.96), 4)))

    p_large = float(_ilogit(_tqt(
        u_raw(parameters['prec_large_p_raw']),
        _logit(0.72), 0.45, _logit(0.20), _logit(0.99), 4)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.75), 0.40, _logit(0.25), _logit(0.98), 4)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.28, math.log(0.40), math.log(1.70), 4)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(0.60), 0.40, math.log(0.15), math.log(1.30), 4)))

    tie_tol = float(np.exp(_tqt(
        u_raw(parameters['tie_tol_raw']),
        math.log(0.60), 0.30, math.log(0.20), math.log(1.30), 4)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.45), 0.40, _logit(0.10), _logit(0.85), 4)))

    large_d_scale = float(np.exp(_tqt(
        u_raw(parameters['large_d_scale_raw']),
        math.log(0.50), 0.40, math.log(0.15), math.log(0.90), 4)))

    window = int(round(float(parameters['window'])))

    # ---- Inverse-valence + damped validity weighting -------------------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    # pi_8-style large-conflict regime: n_features 7-9 with many discriminators
    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9
    # broader large-conflict flag used only to shrink lambda further
    is_large_weight = disc_size >= 6 and n_features >= 7

    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)
    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---- Unique top-expert precedence (pi_8-like) ---------------------------
    unique_top = False
    top_tied = False
    top_tied_idx = np.array([], dtype=int)
    top_clean = 0.0
    sign_top = 0.0
    gap = 0.04

    if disc_size == 1:
        unique_top = True
        sign_top = 1.0 if diff[disc_idx[0]] > 0.0 else -1.0
        gap = 0.04
    else:
        top_val = float(np.max(vals_disc))
        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
        top_idx = disc_idx[top_mask]
        if top_idx.size == 1:
            unique_top = True
            top_pos = int(np.flatnonzero(top_mask)[0])
            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            else:
                gap = 0.04
        else:
            top_tied = True
            top_tied_idx = top_idx
            top_clean = float(np.sum(diff[top_idx]))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))
                E = sign_top * tie_bonus
        else:
            p_apply = p_large if is_large_scale else p_prec
            if abs(D) <= window and np.random.rand() < p_apply:
                bonus = prec_str * (0.95 + 2.0 * math.tanh(gap / 0.10))
                if is_large_scale:
                    bonus = prec_str * (1.00 + 2.40 * math.tanh(gap / 0.10))
                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))
                E = E + sign_top * bonus
    elif top_tied:
        # Raw-majority-dominant softening in the known large-conflict regime.
        if is_large_scale and abs(D) > 1e-12:
            E = D * large_d_scale + (E - D)

        # Tied-top group clean-cue advantage when the tally is near zero.
        if abs(E) <= tie_tol and abs(top_clean) > 1e-9:
            top_effect = tie_str * math.tanh(top_clean / 2.8)
            E = E + top_effect

    # ---- Softmax choice ------------------------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Subjects code a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, with lambda modest but only mildly reduced in large conflicts so high-validity cues still move moderately against raw majorities. A continuously active, smoothly attenuated top-coalition bonus follows tied highest-validity expert groups: it grows approximately linearly with top-group size for small coalitions (m=2-3) but is sharply and smoothly penalized for larger coalitions (m=4+), while opposition from lower-validity cues is damped in a way that scales with top-group size. Unique highest-validity expert precedence is probabilistic and survives large conflicts, but is attenuated when the unique top cue merely agrees with the raw majority. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain and lapse, preserving between-subject heterogeneity.

**Rationale:** This edit keeps the accepted inverse-valence plus damped validity-weighting frame and applies the critic's in-family fixes directly. The tied-top coalition is re-weighted from sqrt(m) to m*tanh(c/scale), which strongly raises m=2-3 coalition following (E5, E8, E18, E20) while a new smooth large-coalition penalty and top-size-scaled opposition damping stop m=4-5 groups from overwhelming lower-cue majorities (E13, E14, E16, E12). Large-conflict validity damping is relaxed by moving lam_large_scale's median from 0.45 to 0.85, directly targeting the E6 sign error. Unique-top precedence is made more available in large conflicts by lowering conflict and net-tally damping, but a new aligned-with-majority attenuation term lowers overuse when the top cue merely agrees with the raw tally (E12). Finally, subject-level transforms use heavier-tailed Student-t/logit-Student-t draws with wider sigma and lower df for the top-coalition, precedence, and lambda parameters, restoring the larger between-subject variance observed in E5, E6, E8, E19, and E20.

**Parameters:**
  - `gain_raw`: `[0.02, 0.98]`
  - `epsilon_raw`: `[0.02, 0.98]`
  - `lambda_raw`: `[0.02, 0.98]`
  - `lam_large_scale_raw`: `[0.02, 0.98]`
  - `prec_p_raw`: `[0.02, 0.98]`
  - `tie_p_raw`: `[0.02, 0.98]`
  - `prec_strength_raw`: `[0.02, 0.98]`
  - `prec_gap_amp_raw`: `[0.02, 0.98]`
  - `prec_opp_decay_raw`: `[0.02, 0.98]`
  - `prec_d_decay_raw`: `[0.02, 0.98]`
  - `prec_conf_att_raw`: `[0.02, 0.98]`
  - `align_scale_raw`: `[0.02, 0.98]`
  - `align_d_att_raw`: `[0.02, 0.98]`
  - `tie_strength_raw`: `[0.02, 0.98]`
  - `tie_coal_scale_raw`: `[0.02, 0.98]`
  - `tie_size_amp_raw`: `[0.02, 0.98]`
  - `tie_large_decay_raw`: `[0.02, 0.98]`
  - `tie_gap_amp_raw`: `[0.02, 0.98]`
  - `tie_opp_decay_raw`: `[0.02, 0.98]`
  - `tie_opp_size_raw`: `[0.02, 0.98]`
  - `tie_d_decay_raw`: `[0.02, 0.98]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import math
import numpy as np
from scipy.stats import t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---------- Heavy-tailed subject-level parameters -------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    df_ht = 1.8

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.35), 0.55, math.log(0.45), math.log(3.60), df_ht)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.20), 0.65, _logit(0.03), _logit(0.55), df_ht)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.75, math.log(0.005), math.log(0.50), df_ht)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.85), 0.45, _logit(0.10), _logit(0.98), df_ht)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.75), 0.90, _logit(0.06), _logit(0.98), df_ht)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.60), 0.75, _logit(0.15), _logit(0.99), df_ht)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.90, math.log(0.15), math.log(2.80), df_ht)))

    prec_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['prec_gap_amp_raw']),
        math.log(0.90), 0.50, math.log(0.10), math.log(2.20), df_ht)))

    prec_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_opp_decay_raw']),
        math.log(0.50), 0.50, math.log(0.03), math.log(1.20), df_ht)))

    prec_d_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_d_decay_raw']),
        math.log(0.10), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    prec_conf_att = float(np.exp(_tqt(
        u_raw(parameters['prec_conf_att_raw']),
        math.log(0.06), 0.55, math.log(0.01), math.log(0.60), df_ht)))

    align_scale = float(_ilogit(_tqt(
        u_raw(parameters['align_scale_raw']),
        _logit(0.40), 0.60, _logit(0.05), _logit(0.90), df_ht)))

    align_d_att = float(np.exp(_tqt(
        u_raw(parameters['align_d_att_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.60), df_ht)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(1.10), 1.00, math.log(0.10), math.log(7.00), df_ht)))

    tie_coal_scale = float(np.exp(_tqt(
        u_raw(parameters['tie_coal_scale_raw']),
        math.log(2.80), 0.30, math.log(1.20), math.log(5.00), df_ht)))

    tie_size_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_size_amp_raw']),
        math.log(0.50), 0.45, math.log(0.05), math.log(1.50), df_ht)))

    tie_large_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_large_decay_raw']),
        math.log(3.00), 0.45, math.log(0.80), math.log(9.00), df_ht)))

    tie_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_gap_amp_raw']),
        math.log(0.70), 0.55, math.log(0.05), math.log(2.00), df_ht)))

    tie_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_decay_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.50), df_ht)))

    tie_opp_size = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_size_raw']),
        math.log(0.80), 0.50, math.log(0.10), math.log(1.80), df_ht)))

    tie_d_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_d_decay_raw']),
        math.log(0.12), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    # ---------- Inverse-valence + damped validity weighting ------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 6 and n_features >= 7
    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)

    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---------- Top-group / unique-top structure -----------------------
    top_val = float(np.max(vals_disc))
    top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
    top_idx = disc_idx[top_mask]

    unique_top = False
    top_tied = False
    sign_top = 0.0
    top_clean = 0.0
    gap = 0.04

    if top_idx.size == 1:
        unique_top = True
        top_pos = int(np.flatnonzero(top_mask)[0])
        sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
        others = vals_disc[~top_mask]
        if others.size > 0:
            gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04
        else:
            gap = 0.25
    else:
        top_tied = True
        top_clean = float(np.sum(diff[top_idx]))
        others = vals_disc[~top_mask]
        if others.size > 0:
            gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04

    lower_positions = disc_idx[~top_mask]

    # ---------- Unique-top precedence ----------------------------------
    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                E = sign_top * prec_str * gap_factor
        else:
            p_apply = p_prec / (1.0 + prec_conf_att * max(0.0, float(disc_size - 2)))
            if np.random.rand() < p_apply:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                opp = 0.0
                if lower_positions.size > 0:
                    opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))
                opp_factor = 1.0 / (1.0 + prec_opp_decay * (opp ** 0.7))
                d_factor = math.exp(-prec_d_decay * min(abs(D), 8.0))

                bonus = prec_str * gap_factor * opp_factor * d_factor

                # Smooth attenuation when the unique top cue merely repeats
                # the direction of the raw majority.
                if sign_top * D > 0.0 and disc_size >= 3:
                    align_factor = align_scale / (1.0 + align_d_att * abs(D))
                    bonus *= align_factor

                E = E + sign_top * bonus

    # ---------- Smooth continuous tied-top coalition bonus -------------
    elif top_tied and abs(top_clean) > 1e-9:
        sign_top = 1.0 if top_clean > 0.0 else -1.0
        m = int(top_idx.size)
        c = abs(top_clean)

        opp = 0.0
        if lower_positions.size > 0:
            opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))

        # Linear in m for small coalitions, with a smooth penalty for large
        # coalitions that would otherwise overwhelm lower-cue majorities.
        coalition = float(m) * math.tanh(c / tie_coal_scale)
        size_factor = 1.0 + tie_size_amp * math.tanh((float(m) - 2.0) / 2.0)
        large_att = 1.0 / (1.0 + tie_large_decay * max(0.0, float(m - 3.0)))

        gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.08)
        opp_mult = 1.0 + tie_opp_size * max(0.0, float(m - 3.0))
        opp_factor = 1.0 / (1.0 + tie_opp_decay * opp_mult * (opp ** 0.7))
        d_factor = math.exp(-tie_d_decay * min(abs(D), 8.0))

        bonus = tie_str * coalition * size_factor * large_att * gap_factor * opp_factor * d_factor
        bonus = min(bonus, 8.0)
        E = E + sign_top * bonus

    # ---------- Softmax choice -----------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    out = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)) or out.sum() <= 0.0:
        return np.array([0.5, 0.5])
    out = out / out.sum()
    return out
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```
