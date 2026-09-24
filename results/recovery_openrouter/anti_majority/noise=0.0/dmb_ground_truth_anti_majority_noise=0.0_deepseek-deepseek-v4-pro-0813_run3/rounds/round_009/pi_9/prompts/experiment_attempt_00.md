# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
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


## COMPETING THEORY
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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Six features are used. The highest-validity expert (0.90) is always the first cue TTB tests. I hold that cue's discriminating direction constant across blocks and vary the lower five cues so Tallying's net evidence changes while TTB sees the same first discriminating cue. In the top-A block the top cue is 1 vs 0 and lower-cue net wins take values +5, +3, +1, -1, -3, -5; in the top-B block the top cue is reversed. TTB predicts identical p(A) for all six top-A pairs and identical p(A) for all six top-B pairs because the first discriminating cue is unchanged; Tallying predicts a graded logistic change because its tally advantage is 1 + lower_net for top-A pairs and lower_net - 1 for top-B pairs. At lower_net = -1 in the top-A block and lower_net = +1 in the top-B block, Tallying is exactly tied while TTB commits to A or B. At lower_net = -3/-5 in the top-A block and +3/+5 in the top-B block, the two theories make opposite directional predictions. The critical signature is flat TTB choice probabilities across the six lower-net levels versus a monotone logistic gradient for Tallying. Twelve unique pairs repeated K = 8 gives 96 trials.

[1] Twelve unique pairs form a 4 × 3 factorial. Six features have validities [0.92, 0.84, 0.76, 0.68, 0.60, 0.55], so TTB's cue cascade is feature 0 → 1 → 2 → 3 → 4 → 5. The first factor is the direction of the first discriminating cue: feature 0 favors A; feature 0 favors B; feature 0 ties and feature 1 favors A; feature 0 ties and feature 1 favors B. The second factor is the tally difference (+2, 0, -2), implemented by the remaining lower features. Tallying ignores which features discriminate, so it predicts three choice-probability levels that depend only on tally difference and are identical across the four first-cue conditions. TTB ignores all lower features once a discriminating cue is found, so it predicts only two choice-probability levels (high for first-cue A, low for first-cue B) and no effect of tally difference. This yields opposite directional predictions on conflict trials: top A with tally -2 (pair 3), top B with tally +2 (pair 4), top tie/second A with tally -2 (pair 9), and top tie/second B with tally +2 (pair 10). It is the complement of the already-explored design: instead of holding the top cue constant and varying lower net, it holds tally difference constant across cue-order conditions and varies the first discriminating cue. With 12 unique pairs repeated K=8, total trials = 96.

[2] Take The Best stops at the first discriminating cue in validity order and ignores all lower cues. Inverse tallying aggregates all cues with reversed valence (0 = favorable) and chooses the option with more zeros. This design holds the inverse-tallying score at exactly zero for every trial: each option has exactly one 1 and one 0 in the two discriminating positions, and all other positions tie. Thus inverse tallying predicts p(A)=0.5 for every pair, regardless of its beta and epsilon. TTB, however, consults cues in validity order and stops at the first discriminating cue. By shifting the position of that first discriminating cue from feature 0 to feature 5 and varying its direction (A=1/B=0 vs A=0/B=1), TTB predicts p(A) to be high when the first cue favors A and low when it favors B, regardless of the lower balancing cue. This yields a strong quantitative dissociation: a flat 0.5 psychometric function across all 12 pairs for inverse tallying versus a step function of first-cue direction for TTB. Unlike the already-explored designs, which held the top cue constant while varying lower-cue net (experiment 0) or varied first-cue direction and tally difference with standard Tallying (experiment 1), this design uses the inverse-tallying tie prediction on every trial and varies the first-cue position across all six possible positions.

[3] Six features with validities [0.90, 0.80, 0.70, 0.60, 0.55, 0.50] fix TTB's cue order as feature 0 through 5. Every trial has all six features discriminating, so TTB always stops at feature 0; its direction is manipulated across trials. The remaining five features are arranged so that the inverse-tally score s_A = #(A=0,B=1) - #(A=1,B=0) takes values +2, 0, or -2. This creates a 2 (feature-0 direction: A=1/B=0 vs A=0/B=1) x 3 (inverse tally score) factorial with two replications per cell. Inverse tallying ignores which feature discriminates and uses only s_A, so it predicts p(A) ordered by s (+2 > 0 > -2) and no effect of feature-0 direction. TTB stops at feature 0 and ignores all lower features, so it predicts p(A) high when feature 0 favors A and low when feature 0 favors B, with no effect of s. The conflict cells (feature-0 A with s=-2; feature-0 B with s=+2) yield opposite directional predictions, and the s=0 cells yield p(A)=0.5 for inverse tallying while TTB commits to the top cue. This double dissociation quantitatively separates the two theories.

[4] This design dissociates the advocated inverse weighted-tally plus validity-ordered tie-break theory from plain inverse tallying. Six validities are spread as [0.95, 0.85, 0.75, 0.65, 0.55, 0.50], so the advocated weak validity weighting (1.9-2.1) makes high-validity cues roughly 2.5 times as influential as low-validity cues. Six tie trials hold the raw inverse tally at exactly 0 (one A-clean and one B-clean discriminating feature). Competing inverse tallying must predict p(A)=0.5 for all six because the tallies are equal, whereas the advocated tie-break predicts p(A) near 0.71 when the first discriminating cue in descending validity order is A=0/B=1 and near 0.29 when it is A=1/B=0; the first discriminator is varied across features 0, 1, and 2. Six conflict trials hold the raw inverse tally at +1 or -1 but place the high-validity wins on the opposite side, so the advocated validity-weighted score reverses sign. For raw +1 trials the weighted scores are about -0.62, -0.42, and -0.22, predicting p(A) around 0.38-0.46, while competing inverse tallying predicts high p(A) around 0.65-0.85. For raw -1 trials the mirror arrangement gives advocated p(A) around 0.54-0.62, while competing inverse tallying predicts low p(A) around 0.15-0.35. Twelve unique pairs repeated K=8 gives 96 trials.

[5] Eight features with symmetric validities around mean 0.75. At validity_weight=2.0 the competing weights are [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6]. The 12 unique pairs form a 2x6 factorial: raw inverse tally d=#(A=0,B=1)-#(A=1,B=0) is +1 or -1, while the validity-weighted score s is set to -0.9,-0.6,-0.3,0,+0.3,+0.6 for d=+1 and the mirror for d=-1. No trial has d=0, so the competing tie-break is never engaged; this isolates the validity-weighting component and avoids repeating design [4]. Plain inverse tallying ignores validities: all d=+1 trials predict p(A) about 0.64-0.85 and all d=-1 trials predict p(A) about 0.15-0.36, flat across s. The competing weighted tally predicts p(A) increasing with s: about 0.33,0.38,0.44,0.50,0.56,0.62,0.67 for s=-0.9,-0.6,-0.3,0,+0.3,+0.6,+0.9. Therefore conflict trials (d=+1 with s<0; d=-1 with s>0) give opposite directional predictions, and s=0 trials oppose p=0.5 to high/low. K=8 repetitions gives 96 trials.

[6] Eight features with validities symmetric around 0.75. For the advocated theory this gives validity weights [1.4,1.3,1.2,1.1,0.9,0.8,0.7,0.6] when the raw tally is nonzero, but every trial here has raw inverse tally exactly zero, so the advocated theory always uses its validity-ordered tie-break. Feature 0 (0.95) discriminates on every trial; top-A trials have feature 0 A=0/B=1 and top-B trials have A=1/B=0, so advocated predicts p(A) approximately 0.715 for all six top-A trials and approximately 0.285 for all six top-B trials, flat across lower-cue composition. The competing inclusion theory does not tie-break on raw zero: cue inclusion is probabilistic, so the included tally is usually nonzero. Lower features are arranged to have net -1 (top A) or +1 (top B) but with increasing numbers of opposite-side lower cues across three levels. With typical inclusion parameters (e.g., alpha=7, gamma=-4.0), competing expected p(A) moves from about 0.81 (level 1) to 0.72 (level 2) to 0.54 (level 3) for top A, and mirror 0.19/0.28/0.46 for top B, crossing the advocated flat predictions. This yields a quantitative interaction: advocated predicts no effect of lower-cue composition, while competing predicts a monotone gradient.

[7] Eight features with three perfect experts (1.0) and five moderate experts (0.70). The three high-validity cues are always unanimous for one product; the five low-validity cues oppose them in 3, 4, or 5 positions (remaining low cues tied). This yields six unique pairs: 2 high-cue direction × 3 low-cue count. Competing theory uses all cues with validity weights 1.375 (high) and 0.775 (low); for high-B/low-A pairs the weighted score is negative (n=4: -1.025, n=5: -0.25), predicting p(A)≈0.31/0.45, and the n=3 raw tie is broken by the high cue, p(A)≈0.29. The mirror high-A/low-B pairs predict p(A)≈0.72/0.69/0.55. The advocated inclusion theory includes high cues with p≈0.97 and low cues with p≈0.80 and tallies included cues equally. It predicts p(A)≈0.18 (n=3), 0.46 (n=4), 0.71 (n=5) for high-B/low-A, and p(A)≈0.83, 0.54, 0.29 for high-A/low-B. Thus the theories make opposite directional predictions in the n=4/n=5 high-B/low-A cells and the n=5 high-A/low-B cell, and show different slopes across low-cue count. K=16 repetitions of the six unique pairs gives 96 trials.

[8] This design holds the raw inverse tally at exactly 0 on every unique pair, so the advocated theory is always in its validity-ordered tie-break mode and chooses using only the first discriminating feature. The top two experts (validities 0.95 and 0.90) are tied in all pairs, so the first discriminator is placed at validity 0.65 or 0.64 after those ties. The six pairs form a first-direction by lower-cue-mass design. Advocated theory predicts a flat step of approximately p(A)=0.71 when the first discriminating cue is A=0/B=1 and approximately p(A)=0.29 when it is A=1/B=0, with no effect of later cues. The competing stochastic-inclusion theory does not have this deterministic raw-tie mode: it includes later cues probabilistically, and in the k=3 pairs the three mid-validity opposite cues (0.64, 0.63, 0.62) plus two 0.50 same-side cues give an expected included tally of about -0.40. Under typical parameters alpha=7, gamma=-3.5, beta_tally around 1.1, beta_tiebreak around 2, p(A) is approximately 0.47 for the first-cue-A k=3 pair and approximately 0.53 for its mirror, opposite in direction to the advocated tie-break. The k=2 pairs provide a weaker opposite lower-cue mass and therefore an intermediate quantitative gradient. This separates a deterministic validity-ordered raw-tie-break from stochastic all-cue inclusion while avoiding the previously explored top-cue tie-break designs.

[9] All six pairs have a raw inverse tally of +2 or -2, so the competing theory is forced to use its mean-centered validity-weighted tally rather than a raw tie-break. The validities are deliberately skewed: features 0-5 are high (1.00 to 0.95), but features 3-5 are tied on every trial and act as high-validity anchors that raise the validity mean to about 0.795. The five low-validity features 6-10 (0.62, 0.60, 0.58, 0.56, 0.54) actually do the discriminating work. At median advocated parameters alpha=7 and gamma=-3.5, inclusion probabilities are about 0.97 for the high cues and 0.70, 0.67, 0.64, 0.60, 0.57 for the low cues. In the competing theory the tied high anchors lower the mean-centered weights of the low cues to about 0.72 to 0.59, while high-cue weights are about 1.33. For the raw +2 pairs, a low-validity majority of 3, 4, or 5 A-clean cues is opposed by 1, 2, or 3 high-validity B-clean cues. The competing weighted score goes from +0.73 through -0.08 to -0.66, so p(A) drops from about 0.64 to 0.49 to 0.37. The advocated expected included tally stays positive at about 1.03, 0.54, and 0.27, so advised p(A) stays at about 0.84, 0.73, and 0.62. The raw -2 mirrors reverse the signs and create a crossover: at the 4- and 5-low-cue levels the two theories make opposite directional predictions. The key manipulation is the tied high anchors, which change the competing mean-centered weights without changing the advocated inclusion tally. Six unique pairs repeated K=16 gives 96 trials.

[10] Nine unique pairs are built from eight binary cues. Every cue discriminates on every trial, so there are no degenerate display ties. The pairs are indexed by k = 0 through 8: the k highest-validity cues are A=0/B=1 and the remaining 8-k cues are A=1/B=0. This makes the inverse-clean tally (0=clean) take the values -8, -6, -4, -2, 0, +2, +4, +6, +8. The pure inverse-valence competing theory predicts a steep monotone trace, near p(A)=0 at k=0 and near p(A)=1 at k=8. The advocated theory has the same inverse tally path for about 86 percent of subjects, but about 14 percent of subjects carry positive valence and treat 1 as clean; on every discriminating cue those subjects choose the opposite product. Thus the advocated prediction is approximately P_adv(k) = 0.14 + 0.72 * P_inv(k), a linear contraction of the inverse trace toward 0.5. At k=0 the advocated model is pulled up to roughly 0.14 to 0.18 instead of near zero; at k=8 it is capped near 0.82 to 0.86 instead of near one; and the overall deflection across k is only about 72 percent as steep. The validities are spread but kept moderately high so inclusion probabilities stay high and the large raw tallies dominate stochastic exclusion, making the orientation-mixture compression the main quantitative dissociation. The k=4 pair also carries an exact raw tie, but its advocated tie-break prediction still lies closer to 0.5 than the pure inverse tie-break prediction. The nine unique pairs are repeated K = max(1, 96 // 9) = 10 times per subject for about 90 trials.

[11] This eight-pair factorial isolates the near-zero top-cue attention bonus in the competing theory while controlling the validity-ordered tie-break. Five informative experts all have validity 0.99, so their inclusion probabilities are nearly identical and high; a sixth low-validity expert (0.55) is tied on every trial, giving the required validity spread without adding discriminative information. Feature 0 is treated as the highest-validity discriminator through stable sorting. Each inverse tally margin is defined as D = number of informative features with A=0/B=1 minus number with A=1/B=0. There are two top-cue directions, with feature 0 favoring A versus B, crossed with four raw inverse-tally margins D = +3, +1, -1, -3. For D = +3 and -3 the competing near-zero bonus is not active because the included tally is not within one cue of zero, and exact-zero tie-breaks are negligible at 0.99 validity. Thus both theories predict essentially no top-direction effect at D = +/-3, but the competing orientation mixture compresses overall choice probabilities toward 0.5. For D = +1 and -1 the competing theory adds its top-cue attention bonus, so the top cue is more decisive only at these near-zero margins; the advocated theory has no such bonus, and only rare inclusion-omission zero tallies create a small tie-break spillover. Approximate median-parameter predictions are: D=+3, advocated p(A) about 0.92 for both top-cue directions, competing p(A) about 0.83 for both; D=+1, advocated p(A) about 0.84 when feature 0 favors A versus 0.79 when feature 0 favors B, whereas competing p(A) shows a larger top-direction gap of about 0.82 versus 0.73. The signature is therefore the top-direction by margin-size interaction: the competing model predicts an enlarged top-cue effect only at D = +/-1, while the advocated model predicts only a small tie-break spillover there and no such effect at D = +/-3. Eight unique pairs are repeated K=12 times for 96 trials.

[12] The fourteen validities form two groups: five 0.96 experts and nine 0.50 experts. All 14 features discriminate on every unique pair, so the raw inverse-clean tally D = (number of A=0/B=1 positions) - (number of A=1/B=0 positions) is +2 for pairs 1-6 and -2 for the swapped mirrors 7-12. Within the high group, x = high-clean advantage for A over B takes the values -5, -3, -1, +1, +3, +5; the low group is arranged to give y = D - x. This holds the raw tally constant while moving the high-validity cues from all favoring B to all favoring A. The advocated theory uses saturating logit-tanh weights that are normalized and bounded; with this validity mix the high/low weight ratio is only about 1.2 at the median of sat_m, so in the x=-5 pair the seven low-validity A-clean cues outweigh the five high-validity B-clean cues and p(A) is about 0.65. The competing theory includes the 0.96 cues with probability about 0.96 and the 0.50 cues with probability about 0.50, then tallies included cues equally, giving expected included evidence about 7*0.50 - 5*0.96 = -1.31 and p(A) about 0.12 for that same pair. The next x condition and its mirror also disagree in direction, and the two models show different crossover locations across x: the competing model crosses to A around x=-1, while the advocated model remains A-favoring for all D=+2 pairs. The D=-2 mirrors reverse the signs and provide the complementary quantitative dissociation. Twelve unique pairs repeated K=8 gives 96 trials.

[13] This design uses 11 features: five high-validity experts at 0.95 and six low-validity experts at 0.85. Every unique pair uses all 11 features as discriminating cues, so the six pairs form a 2-direction by 3-coalition-size factorial. In the critical majority-size cell the raw inverse-clean tally is +1 (six low-validity A-clean cues versus five high-validity B-clean cues). The advocated probabilistic-inclusion tally predicts a clear A advantage because the low cues are included with high probability and outnumber the five high cues. The competing continuous-weight theory, however, uses saturating logit-tanh validity weights that give the five 0.95 cues only about 1.10 weight and the six 0.85 cues about 0.91 weight, so 6*0.91 - 5*1.10 is approximately -0.06 at the median saturation. That near-zero weighted evidence sends the competing model into its inconclusive-evidence fallback, where the five tied top-validity cues all favor B, producing p(A) around 0.40. The advocated model stays above 0.60 for A. The mirror pair reverses this, giving p(A) around 0.60 for the competing model and below 0.40 for the advocated model. The two adjacent coalition-size pairs move the competing weighted evidence upward to about +0.33 and +0.71 for the positive-raw-tally direction, so the competing model's p(A) crosses from below 0.5 to about 0.53 and 0.57 while the advocated model remains roughly flat near 0.62. This yields a quantitative crossover interaction, not just a main effect. The six unique pairs are repeated K = 16 times for 96 trials.

[14] Eight features are used; the three highest-validity experts are tied at 0.95 and the five remaining experts are at 0.50. Every trial uses all eight features as discriminating cues, so the advocated theory is forced into its large tied-top conflict regime: there are eight discriminating cues in an eight-feature stimulus and no unique highest-validity expert, so the raw inverse-clean tally D is multiplied by large_tie_scale. Thus within each fixed-D block the advocated theory predicts one flat choice probability: all D=+2 pairs have the same p(A) and all D=-2 pairs have the mirrored p(A), with no effect of how the high-validity versus low-validity cues are arranged. The competing theory includes each cue with a validity-dependent probability, about 0.96 for the 0.95 cues and about 0.50 for the 0.50 cues at median parameters, and then tallies included cues equally. Its expected included tally is s_hat = p_high * h + p_low * l, where h is the high-group clean-cue advantage and l is the low-group clean-cue advantage. Each block fixes D = h + l at +2 or -2 while h sweeps from -3 to +3. For the D=+2 block the advocated prediction is flat near p(A)=0.69, whereas the competing expected p(A) rises from about 0.33 for h=-3,l=+5, to 0.73, 0.89, and 0.92. The first pair therefore gives opposite directional predictions. The D=-2 block is the mirror: advocated stays flat near p(A)=0.31, while competing falls from about 0.67 for h=+3,l=-5, to 0.27, 0.11, and 0.08, again giving an opposite direction on the first pair. The critical signature is a flat composition effect coupled only with a D sign effect for the advocated theory, versus a large composition gradient and crossover for the competing theory. Eight unique pairs repeated K=12 gives 96 trials.

[15] Eight experts: four tied high-validity experts (0.98) and four lower-validity experts (0.56 to 0.53). Every expert discriminates on every unique pair. The four unique pairs hold the raw inverse-clean tally at exactly zero: the high-validity group's clean-cue advantage for A over B (h) takes -4, -2, +2, and +4, while the low-validity group's advantage is set to -h. Thus D = 0 on every trial. In the competing theory, zero raw evidence with a tied top group of four equal 0.98 validities cannot trigger the unique-expert precedence or tie-break path, and the large tied-top scaled branch only applies to nonzero D. Therefore the competing theory collapses to E = 0 and p(A) = 0.5 for all four pairs regardless of its parameters, giving a flat psychometric function over h. The advocated theory uses validity-graded probabilistic inclusion, so the 0.98 experts are included with probability near 0.97 while the 0.53-0.56 experts are included with probability roughly 0.56 to 0.60 at median parameters. The expected included evidence for A is therefore approximately the sign of (p_high - p_low) times h, so p(A) is predicted to be low for h = -4 and -2 and high for h = +2 and +4, increasing monotonically: around 0.12, 0.21, 0.78, and 0.88 at median parameters. This yields a strong quantitative dissociation: a flat 0.5 line for the competing theory versus a monotonic h-gradient with sign reversal for the advocated theory.

[16] This design isolates the tied-top large-conflict divergence between the two theories. Nine experts are split into four tied highest-validity experts (0.95) and five low-validity experts (0.50). Every unique pair is coded by D, the raw inverse clean-cue tally, and h, the clean-cue advantage within the tied top group. In the D = +1 block, h sweeps -4, -2, 0, +2, +4 while the low group is set to D - h; the D = -1 block is the mirror. Because the four top experts are tied and there are 9 features with at least 8 discriminating cues, the advocated theory enters its large tied-top conflict regime: it suppresses tied-top precedence and only scales D by large_tie_scale, so all five D = +1 pairs must produce exactly the same p(A), all D = -1 pairs produce the mirrored p(A), and the D = 0 pairs must produce p(A) = 0.5. The competing theory instead uses a damped validity-weighted tally plus a tied-top clean-cue fallback when the weighted evidence is near zero. Its p(A) therefore varies with h within a fixed D. In the critical D = +1, h = -4 pair the four top cues all favor B while five low cues favor A; the competing fallback flips choice below 0.5, roughly p(A) around 0.41, whereas the advocated theory stays above 0.5. The D = -1, h = +4 mirror gives the opposite reversal. The two D = 0 pairs provide a second lever: advocated is forced to 0.5, while competing follows the tied top clean-cue advantage, roughly 0.26 versus 0.74. This produces both directional reversals and a flat-versus-gradient interaction. The 12 unique pairs repeated 8 times give about 96 trials.

[17] This design isolates the tied-top near-zero fallback and damped validity weighting in the advocated theory against the raw large-conflict tally of the competing theory. Seven features are used: the three highest-validity experts are exactly tied at 0.85, and the four lowest are exactly tied at 0.50, so every trial has a tied top group and cannot trigger unique-expert precedence. Every pair makes all seven features discriminate. The raw inverse-clean tally is D = sum diff, where diff = +1 when A=0/B=1 and diff = -1 when A=1/B=0. In the D = +1 block, the top-group clean-cue advantage h takes -3, -1, +1, and +3, with the bottom-group advantage fixed to l = 1 - h. The D = -1 block is the mirror image. The competing theory suppresses the tied top group and, in this large conflict, uses only E = D * large_tie_scale for every pair in a block, so its predicted p(A) is flat across h within each D block. The advocated theory first applies damped validity weights, then softens the raw D in the large tied-top regime while keeping the validity-weighted residual, and finally adds a tied-top clean-cue fallback because the softened evidence is near zero. This produces a monotone h gradient and a crossover: for D = +1 the advocated p(A) is approximately 0.47, 0.56, 0.68, and 0.74 as h goes from -3 to +3, whereas the competing p(A) is flat near 0.58 to 0.63. For D = -1 the advocated p(A) is approximately 0.53, 0.44, 0.32, and 0.26, again crossing the flat competing prediction. The critical signature is therefore an h by D interaction in the advocated theory but only a D main effect in the competing theory.

[18] This design isolates the large-conflict validity-weight amplification in the advocated theory against the damped weighted tally and D-gated precedence of the competing theory. Nine features: one unique top expert (0.99), three high experts (0.90/0.88/0.86), and five low experts (0.55). All nine features discriminate on every trial, so the unique top expert is never tied and the raw clean-cue tally is controlled by the high coalition versus the low majority. In the top-A block the top expert and two or three high experts favor A while all five low experts favor B: the all-high-aligned trial has raw tally D=-1, and the three dissent trials (dissenting high validity 0.90, 0.88, or 0.86) have D=-3. In the advocated theory, the large-conflict regime amplifies validity weighting by about lam_large_scale, giving the top/high coalition weights near 1.4-1.7 and the 0.55 cues weights near 0.58; even at D=-3 the weighted evidence for the top direction stays positive (approximately +0.19 to +0.40), so the model chooses the top direction on all top-A trials. It also predicts a small graded effect of which high expert dissents. The competing theory instead shrinks lambda in large conflicts, leaving high-versus-low weights near 1.0: at D=-3 its weighted evidence is approximately -2.86 for all three dissent trials, and because its unique-top precedence is only applied when |D|<=window (window is 1 or 2), those trials flip to the low-validity majority. At D=-1 the competing model does apply probabilistic top precedence, producing an intermediate p(A) around 0.35-0.65, while the advocated model is near ceiling for the top direction. The four mirror trials reverse the top direction. The critical signatures are (i) an opposite directional prediction on the D=+/-3 dissent trials and (ii) a flat-versus-graded effect of dissenting-high validity: the advocated model moves with which high expert dissents, while the competing model stays flat because its D=-3 evidence is outside the precedence window and its validities carry little weight. Eight unique pairs repeated K=12 give about 96 trials.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Six features keep the design out of both theories' large-conflict regimes (which require 7-9 features), so only the core weighted-tally and tied-top components are tested. The two highest experts are exactly tied at 0.95; the other four take 0.58, 0.55, 0.52, and 0.50. Six unique pairs cross D in {-2, 0, +2} (raw inverse-clean tally, where 0 is favorable) with h in {-2, +2} (clean-cue advantage inside the tied top pair), while the lower four cues are set to D - h. For the advocated theory, damped validity weighting leaves the weighted tally near 1.5 to 2.2 in the D = +2/-2 rows, so the tied-top fallback, which requires abs(E) <= tie_tol, stays off; p(A) should be flat across h inside each non-zero D row. For the competing theory, the tied-top component is always active, with an m = 2 multiplier of 2.70, and contributes about +2.1 or -2.1, reversing direction on the critical D = +2/h = -2 and D = -2/h = +2 pairs. Approximate median predictions are 0.81 versus 0.41 for D = +2/h = -2, and 0.19 versus 0.59 for its mirror, while the D = 0 row gives 0.36 versus 0.16 and 0.65 versus 0.84. The six unique pairs are repeated K = 16 times for 96 trials.",
  "validities": [
    0.95,
    0.95,
    0.85,
    0.83,
    0.81,
    0.8
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 14801,
  "output_tokens": 22883,
  "reasoning_tokens": 22238,
  "provider": "Baidu"
}
```
