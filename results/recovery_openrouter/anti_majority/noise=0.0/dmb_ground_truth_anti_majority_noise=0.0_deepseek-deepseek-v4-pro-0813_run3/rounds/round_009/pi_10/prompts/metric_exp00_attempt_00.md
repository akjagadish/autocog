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
**Validities (n_features=9):** [0.99, 0.9, 0.88, 0.86, 0.55, 0.55, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Rationale:** This design isolates the large-conflict validity-weight amplification in the advocated theory against the damped weighted tally and D-gated precedence of the competing theory. Nine features: one unique top expert (0.99), three high experts (0.90/0.88/0.86), and five low experts (0.55). All nine features discriminate on every trial, so the unique top expert is never tied and the raw clean-cue tally is controlled by the high coalition versus the low majority. In the top-A block the top expert and two or three high experts favor A while all five low experts favor B: the all-high-aligned trial has raw tally D=-1, and the three dissent trials (dissenting high validity 0.90, 0.88, or 0.86) have D=-3. In the advocated theory, the large-conflict regime amplifies validity weighting by about lam_large_scale, giving the top/high coalition weights near 1.4-1.7 and the 0.55 cues weights near 0.58; even at D=-3 the weighted evidence for the top direction stays positive (approximately +0.19 to +0.40), so the model chooses the top direction on all top-A trials. It also predicts a small graded effect of which high expert dissents. The competing theory instead shrinks lambda in large conflicts, leaving high-versus-low weights near 1.0: at D=-3 its weighted evidence is approximately -2.86 for all three dissent trials, and because its unique-top precedence is only applied when |D|<=window (window is 1 or 2), those trials flip to the low-validity majority. At D=-1 the competing model does apply probabilistic top precedence, producing an intermediate p(A) around 0.35-0.65, while the advocated model is near ceiling for the top direction. The four mirror trials reverse the top direction. The critical signatures are (i) an opposite directional prediction on the D=+/-3 dissent trials and (ii) a flat-versus-graded effect of dissenting-high validity: the advocated model moves with which high expert dissents, while the competing model stays flat because its D=-3 evidence is outside the precedence window and its validities carry little weight. Eight unique pairs repeated K=12 give about 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive amplified validity weighting. When highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction, but the component is capped at |D|=1 and sharply suppressed for three-cue top groups facing five opposing lower-validity cues, so lower-validity majorities can override the tied top group in large conflicts. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts and is further damped in six-feature full conflicts with |D|>=2. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.

**Parameters:**
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- prec_large_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- opp_decay_raw: [0.02, 0.98]
- prec_disc_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_gap_amp_raw: [0.02, 0.98]
- tie_conflict_amp_raw: [0.02, 0.98]
- tie_d0_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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


## COMPETING THEORY
**Description:** Subjects treat a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so each discriminating expert contributes an inverse-valence direction. The final evidence is a damped validity-weighted tally: each discriminating cue is weighted by 1 + lambda*(v_j - mean(v)), with lambda modest and further reduced in large multi-cue conflicts, so raw clean-cue majority dominates while high-validity cues receive a moderate extra weight. When exactly one discriminating expert has strictly highest communicated validity, a probabilistic, validity-gap-scaled precedence bonus is added to that expert's direction. When top experts are tied and the weighted evidence is near zero, subjects follow the sign of the tied top group's clean-cue advantage with moderate sensitivity instead of guessing. Choice is a softmax over the resulting evidence with subject-specific gain and lapse. Subject-level gain, lambda, precedence probability, large-conflict raw-tally softening, tie parameters, and epsilon are drawn from Student-t heavy-tailed/logit-Student-t distributions with medians anchored near the accepted base, providing the between-subject variance observed in the data without shifting the aggregate pattern too far.

**Parameters:**
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- prec_large_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_tol_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- large_d_scale_raw: [0.02, 0.98]
- window: {1, 2}
- validities: validities

**`predict source code`:**
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

**`policy source code`:**
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
  "metric_source": "import numpy as np\ndef metric(data):\n    a = [row for row in data['option_a_ratings']]\n    b = [row for row in data['option_b_ratings']]\n    resp = data['response'].to_numpy()\n    d_vals = []\n    top_choices = []\n    for ar, br, r in zip(a, b, resp):\n        d = sum([1 if (y > x) else -1 for x, y in zip(ar, br) if x != y])\n        d_vals.append(d)\n        top_choices.append(((ar[0] == 0) and (br[0] == 1) and (r == 0)) or\n                           ((ar[0] == 1) and (br[0] == 0) and (r == 1)))\n    d_vals = np.array(d_vals)\n    top_choices = np.array(top_choices, dtype=float)\n    mask = np.abs(d_vals) == 3\n    if mask.sum() == 0:\n        return 0.5\n    return float(np.mean(top_choices[mask]))",
  "rationale": "This metric isolates the six large-conflict dissent trial types, where the raw tally D has magnitude 3. In these trials one of the three high-validity experts dissents from the unique top expert at feature 0 while all five low-validity experts oppose. The metric returns the proportion of those trials in which the subject chose the option favored by the unique top expert. The advocated theory predicts amplified validity weighting keeps the top/coalition direction positive on these trials, producing a high top-consistency rate. The competing theory predicts the damped weighted tally is strongly negative on these trials and its top-precedence bonus is disabled because abs(D)=3 exceeds the precedence window, so the low-validity majority is chosen and the same rate is low. The D=+/-1 trials are excluded because both theories can show partial top-following there, sharpening the contrast while retaining 72 stable trials per subject."
}
```

## Usage

```json
{
  "input_tokens": 8858,
  "output_tokens": 19703,
  "reasoning_tokens": 19238,
  "provider": "Baidu"
}
```
