# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_10" and "pi_9") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_10" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_9" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_10
Subjects code a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, but large conflicts with a unique top expert receive amplified validity weighting. When highest-validity experts are tied, their clean-cue advantage is a continuous, always-active evidence component that scales with top-group size, validity gap, conflict size, and clean-cue direction, but the component is capped at |D|=1 and sharply suppressed for three-cue top groups facing five opposing lower-validity cues, so lower-validity majorities can override the tied top group in large conflicts. Unique top-expert precedence is strong in small and sparse conflicts but decays in dense conflicts and is further damped in six-feature full conflicts with |D|>=2. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain, lapse, validity-weight sensitivity, precedence probability, and top-group sensitivity.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_9
Subjects treat a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so each discriminating expert contributes an inverse-valence direction. The final evidence is a damped validity-weighted tally: each discriminating cue is weighted by 1 + lambda*(v_j - mean(v)), with lambda modest and further reduced in large multi-cue conflicts, so raw clean-cue majority dominates while high-validity cues receive a moderate extra weight. When exactly one discriminating expert has strictly highest communicated validity, a probabilistic, validity-gap-scaled precedence bonus is added to that expert's direction. When top experts are tied and the weighted evidence is near zero, subjects follow the sign of the tied top group's clean-cue advantage with moderate sensitivity instead of guessing. Choice is a softmax over the resulting evidence with subject-specific gain and lapse. Subject-level gain, lambda, precedence probability, large-conflict raw-tally softening, tie parameters, and epsilon are drawn from Student-t heavy-tailed/logit-Student-t distributions with medians anchored near the accepted base, providing the between-subject variance observed in the data without shifting the aggregate pattern too far.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_10)

### DESIGN
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



### METRIC
Rationale:
This metric isolates the six large-conflict dissent trial types, where the raw tally D has magnitude 3. In these trials one of the three high-validity experts dissents from the unique top expert at feature 0 while all five low-validity experts oppose. The metric returns the proportion of those trials in which the subject chose the option favored by the unique top expert. The advocated theory predicts amplified validity weighting keeps the top/coalition direction positive on these trials, producing a high top-consistency rate. The competing theory predicts the damped weighted tally is strongly negative on these trials and its top-precedence bonus is disabled because abs(D)=3 exceeds the precedence window, so the low-validity majority is chosen and the same rate is low. The D=+/-1 trials are excluded because both theories can show partial top-following there, sharpening the contrast while retaining 72 stable trials per subject.

Source:
import numpy as np
def metric(data):
    a = [row for row in data['option_a_ratings']]
    b = [row for row in data['option_b_ratings']]
    resp = data['response'].to_numpy()
    d_vals = []
    top_choices = []
    for ar, br, r in zip(a, b, resp):
        d = sum([1 if (y > x) else -1 for x, y in zip(ar, br) if x != y])
        d_vals.append(d)
        top_choices.append(((ar[0] == 0) and (br[0] == 1) and (r == 0)) or
                           ((ar[0] == 1) and (br[0] == 0) and (r == 1)))
    d_vals = np.array(d_vals)
    top_choices = np.array(top_choices, dtype=float)
    mask = np.abs(d_vals) == 3
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(top_choices[mask]))

### RESULTS
- Predicted under pi_10 (simulated): 0.3828 (var=0.0886)
- Predicted under pi_9 (simulated): 0.1447 (var=0.0036)
- Observed on real data: 0.3422 (var=0.0518)

## EXPERIMENT 2 (proposed by pi_9)

### DESIGN
**Validities (n_features=6):** [0.95, 0.95, 0.85, 0.83, 0.81, 0.8]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** Six features keep the design out of both theories' large-conflict regimes (which require 7-9 features), so only the core weighted-tally and tied-top components are tested. The two highest experts are exactly tied at 0.95; the other four take 0.58, 0.55, 0.52, and 0.50. Six unique pairs cross D in {-2, 0, +2} (raw inverse-clean tally, where 0 is favorable) with h in {-2, +2} (clean-cue advantage inside the tied top pair), while the lower four cues are set to D - h. For the advocated theory, damped validity weighting leaves the weighted tally near 1.5 to 2.2 in the D = +2/-2 rows, so the tied-top fallback, which requires abs(E) <= tie_tol, stays off; p(A) should be flat across h inside each non-zero D row. For the competing theory, the tied-top component is always active, with an m = 2 multiplier of 2.70, and contributes about +2.1 or -2.1, reversing direction on the critical D = +2/h = -2 and D = -2/h = +2 pairs. Approximate median predictions are 0.81 versus 0.41 for D = +2/h = -2, and 0.19 versus 0.59 for its mirror, while the D = 0 row gives 0.36 versus 0.16 and 0.65 versus 0.84. The six unique pairs are repeated K = 16 times for 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the always-active tied-top-cue effect that separates the competing theory from the advocated theory. For each trial I compute the raw clean-cue tally D and the clean-cue advantage of the two tied top experts, top. I keep only the non-zero D trials (the four D = +2 and D = -2 cells), where the advocated weighted tally is far from zero and its tied-top fallback stays off. Within each D sign there is an h = -2 cell and an h = +2 cell. The metric is the per-subject average difference in P(choose A) between top = +2 and top = -2, averaged across D = +2 and D = -2. Under the advocated theory this difference is close to zero because only a tiny validity-weighting asymmetry separates the two h cells. Under the competing theory the tied-top component is always active and adds approximately 2 units of evidence in the top-pair direction, so the same difference is large and positive. Comparing h cells within the same D sign also cancels each subject's overall raw-tally bias, gain, and lapse, which keeps between-subject variance low.

Source:
def metric(data):
    import numpy as np
    import pandas as pd

    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    diff = (B > A).astype(float) - (A > B).astype(float)
    D = diff.sum(axis=1)
    top = diff[:, 0] + diff[:, 1]
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    aux = pd.DataFrame({
        'sid': data['subject_id'].to_numpy(),
        'D': D,
        'top': top,
        'chose_a': chose_a,
    })

    def subj_metric(g):
        pp = g.loc[(g['D'] > 0) & (g['top'] > 0), 'chose_a'].mean()
        pn = g.loc[(g['D'] > 0) & (g['top'] < 0), 'chose_a'].mean()
        np_ = g.loc[(g['D'] < 0) & (g['top'] > 0), 'chose_a'].mean()
        nn = g.loc[(g['D'] < 0) & (g['top'] < 0), 'chose_a'].mean()
        return 0.5 * ((pp - pn) + (np_ - nn))

    scores = aux.groupby('sid', sort=False).apply(subj_metric)
    return float(scores.mean())


### RESULTS
- Predicted under pi_10 (simulated): 0.3738 (var=0.1102)
- Predicted under pi_9 (simulated): 0.0181 (var=0.0090)
- Observed on real data: 0.0900 (var=0.0603)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    A = np.array(data['option_a_ratings'].tolist(), dtype=int)
    B = np.array(data['option_b_ratings'].tolist(), dtype=int)
    y = 1.0 - data['response'].to_numpy(dtype=float)

    top_a = A[:, 0] > B[:, 0]
    top_b = A[:, 0] < B[:, 0]

    lower_a_wins = (A[:, 1:] > B[:, 1:]).sum(axis=1)
    lower_b_wins = (B[:, 1:] > A[:, 1:]).sum(axis=1)
    lower_net = lower_a_wins - lower_b_wins

    high = lower_net >= 1
    low = lower_net <= -1

    def diff(top_mask):
        hi = y[top_mask & high]
        lo = y[top_mask & low]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(hi.mean() - lo.mean())

    return (diff(top_a) + diff(top_b)) / 2.0
```

**Observed (real) value:** -0.5117 (var=0.0161)
**Predicted under pi_10:** -0.5287 (var=0.0134)
**Predicted under pi_9:** -0.4625 (var=0.0083)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_rows = data['option_a_ratings'].tolist()
    b_rows = data['option_b_ratings'].tolist()
    tally = []
    for ra, rb in zip(a_rows, b_rows):
        a_wins = sum(1 for x, y in zip(ra, rb) if x > y)
        b_wins = sum(1 for x, y in zip(ra, rb) if y > x)
        tally.append(a_wins - b_wins)
    tally = np.asarray(tally, dtype=float)
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    pos_rate = choose_a[tally > 0].mean()
    neg_rate = choose_a[tally < 0].mean()
    return float(pos_rate - neg_rate)
```

**Observed (real) value:** -0.6438 (var=0.0430)
**Predicted under pi_10:** -0.5869 (var=0.0358)
**Predicted under pi_9:** -0.5531 (var=0.0300)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55])
    cue_order = np.argsort(-validities, kind='stable')

    a = np.array([list(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([list(r) for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)

    n = a.shape[0]
    if n == 0:
        return 0.0

    predicted = np.full(n, -1, dtype=int)
    for cue in cue_order:
        undecided = predicted == -1
        disc = np.logical_and(undecided, a[:, cue] != b[:, cue])
        if not np.any(disc):
            continue
        predicted[disc] = np.where(a[disc, cue] > b[disc, cue], 0, 1)

    decided = predicted != -1
    if not np.any(decided):
        return 0.0

    match = (resp[decided] == predicted[decided]).astype(float)
    return float(np.mean(2.0 * match - 1.0))
```

**Observed (real) value:** -0.4300 (var=0.0221)
**Predicted under pi_10:** -0.4000 (var=0.0432)
**Predicted under pi_9:** -0.4450 (var=0.0283)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    # inverse-tally advantage for A: #(A=0,B=1) - #(A=1,B=0)
    s = (b > a).sum(axis=1) - (a > b).sum(axis=1)
    mask_pos = s == 2
    mask_neg = s == -2
    pos = data.loc[mask_pos, 'response']
    neg = data.loc[mask_neg, 'response']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_pos = (pos == 0).mean()
    p_neg = (neg == 0).mean()
    return float(p_pos - p_neg)
```

**Observed (real) value:** 0.4925 (var=0.0658)
**Predicted under pi_10:** 0.6331 (var=0.0232)
**Predicted under pi_9:** 0.5325 (var=0.0266)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

PAIR_SIGNS = {
    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): 1.0,
    ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)): 1.0,
    ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)): -1.0,
    ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 0, 1, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0)): 1.0,
}

def metric(data: pd.DataFrame) -> float:
    contributions = []
    for _, row in data.iterrows():
        key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))
        sign = PAIR_SIGNS.get(key)
        if sign is None:
            continue
        y = 1.0 if row['response'] == 0 else -1.0
        contributions.append(y * sign)
    return float(np.mean(contributions)) if contributions else 0.0
```

**Observed (real) value:** 0.2658 (var=0.1584)
**Predicted under pi_10:** 0.0975 (var=0.0284)
**Predicted under pi_9:** 0.3075 (var=0.0162)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    validities = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55], dtype=float)
    weights = 1.0 + 2.0 * (validities - validities.mean())

    pos = []
    neg = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        clean_a = (b > a).astype(float)
        clean_b = (a > b).astype(float)
        diff = clean_a - clean_b
        d = float(np.sum(diff))
        s = float(np.dot(weights, diff))
        y = 1.0 - float(row['response'])
        if d < 0.0 and s > 0.0:
            pos.append(y)
        elif d > 0.0 and s < 0.0:
            neg.append(y)

    if not pos or not neg:
        return 0.0

    return float(np.mean(pos) - np.mean(neg))
```

**Observed (real) value:** 0.0867 (var=0.4244)
**Predicted under pi_10:** 0.1267 (var=0.1923)
**Predicted under pi_9:** 0.0483 (var=0.0368)

### Experiment 9
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    ld = data.apply(lambda row: sum([1 for x, y in zip(row['option_a_ratings'][1:], row['option_b_ratings'][1:]) if x != y]), axis=1).to_numpy(dtype=float)
    choose_a = 1.0 - data['response'].to_numpy(dtype=float)
    level = np.where(ld == 1, 1, np.where(ld == 3, 2, np.where(ld == 5, 3, 0)))
    topA = ((a0 == 0) & (b0 == 1))
    topB = ((a0 == 1) & (b0 == 0))
    m1 = choose_a[topA & (level == 1)].mean() if (topA & (level == 1)).any() else 0.0
    m2 = choose_a[topB & (level == 3)].mean() if (topB & (level == 3)).any() else 0.0
    m3 = choose_a[topA & (level == 3)].mean() if (topA & (level == 3)).any() else 0.0
    m4 = choose_a[topB & (level == 1)].mean() if (topB & (level == 1)).any() else 0.0
    return float(m1 + m2 - m3 - m4)
```

**Observed (real) value:** 0.0725 (var=0.0580)
**Predicted under pi_10:** 0.0213 (var=0.0926)
**Predicted under pi_9:** -0.0062 (var=0.0541)

### Experiment 10
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    counts = {}
    totals = {}

    for row in data.itertuples(index=False):
        a = [int(x) for x in row.option_a_ratings]
        b = [int(x) for x in row.option_b_ratings]

        # h = 0 means the three high-validity cues favour B under inverse valence;
        # h = 1 is the mirror where they favour A.
        h = 0 if sum(a[:3]) > sum(b[:3]) else 1

        # Number of opposing low-validity cues.
        n = sum(1 for i in range(3, len(a)) if a[i] != b[i])

        key = (h, n)
        totals[key] = totals.get(key, 0) + 1
        if int(row.response) == 0:
            counts[key] = counts.get(key, 0) + 1

    props = {}
    for h in (0, 1):
        for n in (3, 4, 5):
            tot = totals.get((h, n), 0)
            props[(h, n)] = (counts.get((h, n), 0) / tot) if tot > 0 else 0.5

    d3 = props[(0, 3)] - props[(1, 3)]
    d4 = props[(0, 4)] - props[(1, 4)]
    d5 = props[(0, 5)] - props[(1, 5)]

    return float(-d3 + d4 + 2.0 * d5)
```

**Observed (real) value:** 1.0100 (var=2.8518)
**Predicted under pi_10:** 1.3738 (var=0.2713)
**Predicted under pi_9:** 0.9838 (var=0.4229)

### Experiment 11
**Design**
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_VALIDITIES = np.array([0.95, 0.90, 0.65, 0.64, 0.63, 0.62, 0.60, 0.60])
_FIRST_CUE_ORDER = np.argsort(-_VALIDITIES, kind='stable')


def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=np.int64)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=np.int64)
    r = np.asarray(data['response'].tolist(), dtype=np.int64)

    n = len(data)
    first_col = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)

    for j in _FIRST_CUE_ORDER:
        differs = a[:, j] != b[:, j]
        newly = differs & ~assigned
        first_col[newly] = j
        assigned |= differs

    valid = first_col >= 0
    if not valid.any():
        return 0.0

    idx_valid = np.nonzero(valid)[0]
    cols_valid = first_col[valid]
    a_first = a[idx_valid, cols_valid]
    b_first = b[idx_valid, cols_valid]
    favors_a = (a_first == 0) & (b_first == 1)

    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)
    high_mask[idx_valid[favors_a]] = True
    low_mask[idx_valid[~favors_a]] = True

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0

    chose_a = r == 0
    return float(chose_a[high_mask].mean() - chose_a[low_mask].mean())

```

**Observed (real) value:** 0.3050 (var=0.0605)
**Predicted under pi_10:** 0.2175 (var=0.0401)
**Predicted under pi_9:** 0.3433 (var=0.0229)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    scores = []
    for sid, sub in data.groupby('subject_id'):
        low = []
        for a, b, resp in zip(sub['option_a_ratings'], sub['option_b_ratings'], sub['response']):
            s = 0
            for x, y in zip(a, b):
                if x == 0 and y == 1:
                    s += 1
                elif x == 1 and y == 0:
                    s -= 1
            if s > 0:
                low.append(1.0 if resp == 0 else 0.0)
            elif s < 0:
                low.append(1.0 if resp == 1 else 0.0)
        if not low:
            scores.append(0.0)
        else:
            q = float(np.mean(low))
            scores.append(q * q - q)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1667 (var=0.0053)
**Predicted under pi_10:** -0.1847 (var=0.0032)
**Predicted under pi_9:** -0.1827 (var=0.0021)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    k = np.array([sum(1 for x, y in zip(a, b) if x == 0 and y == 1) for a, b in zip(a_list, b_list)], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    subject_ids = data['subject_id'].to_numpy()
    informative = k != 4
    inv_pred = np.where(k < 4, 1.0, 0.0)
    inv_match = resp == inv_pred

    def subject_consistency(mask):
        valid = informative & mask
        if not valid.any():
            return 0.5
        return float(np.mean(inv_match[valid]))

    unique_ids = np.unique(subject_ids)
    if len(unique_ids) > 1:
        positive_count = []
        for sid in unique_ids:
            s = subject_consistency(subject_ids == sid)
            positive_count.append(1.0 if s < 0.5 else 0.0)
        return float(np.mean(positive_count))
    s = subject_consistency(np.ones(len(data), dtype=bool))
    return float(max(s, 1.0 - s))

```

**Observed (real) value:** 0.0000 (var=0.0045)
**Predicted under pi_10:** 0.0000 (var=0.0065)
**Predicted under pi_9:** 0.0000 (var=0.0028)

### Experiment 14
**Design**
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def score_one(g):
        a = np.array([list(x) for x in g['option_a_ratings']], dtype=int)
        b = np.array([list(x) for x in g['option_b_ratings']], dtype=int)
        resp = g['response'].to_numpy().astype(int)

        d = ((a == 0) & (b == 1)).sum(axis=1) - ((a == 1) & (b == 0)).sum(axis=1)
        choice_a = (resp == 0).astype(float)

        far = np.abs(d) == 3
        far_pos = far & (d == 3)
        far_neg = far & (d == -3)
        if np.any(far_pos) and np.any(far_neg):
            orient_diff = float(np.mean(choice_a[far_pos]) - np.mean(choice_a[far_neg]))
            orient = 1.0 if orient_diff >= 0.0 else -1.0
        else:
            corr = float(np.corrcoef(d.astype(float), choice_a)[0, 1])
            orient = 1.0 if np.isfinite(corr) and corr >= 0.0 else -1.0

        aligned_d = orient * d
        majority_opt = np.where(aligned_d > 0, 0, 1)

        raw_top_dir = np.where((a[:, 0] == 0) & (b[:, 0] == 1), 1.0, -1.0)
        top_opt = np.where(orient * raw_top_dir > 0.0, 0, 1)
        top_align = (top_opt == majority_opt)
        followed_top = (resp == top_opt).astype(float)

        near = np.abs(d) == 1
        far2 = np.abs(d) == 3

        def arcs(mask):
            vals = followed_top[mask]
            n = vals.shape[0]
            if n == 0:
                return 0.0
            p = (float(np.sum(vals)) + 0.5) / (float(n) + 1.0)
            p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            return float(np.arcsin(np.sqrt(p)))

        near_aligned = arcs(near & top_align)
        near_opposed = arcs(near & ~top_align)
        far_aligned = arcs(far2 & top_align)
        far_opposed = arcs(far2 & ~top_align)

        return (near_aligned - near_opposed) + (far_aligned - far_opposed)

    if data.empty:
        return 0.0

    scores = []
    for sid in data['subject_id'].unique():
        sub = data[data['subject_id'] == sid]
        scores.append(score_one(sub))
    return float(np.mean(scores))
```

**Observed (real) value:** 1.2973 (var=0.2536)
**Predicted under pi_10:** 1.4052 (var=0.1406)
**Predicted under pi_9:** 1.3103 (var=0.0593)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(row) for row in data['option_a_ratings']], dtype=float)
    b = np.array([list(row) for row in data['option_b_ratings']], dtype=float)

    clean_a = (a == 0) & (b == 1)
    clean_b = (a == 1) & (b == 0)

    D = clean_a.sum(axis=1) - clean_b.sum(axis=1)
    x = clean_a[:, :5].sum(axis=1) - clean_b[:, :5].sum(axis=1)

    chose_a = (data['response'].to_numpy(dtype=float) == 0)

    mask_pos = (D == 2) & (x == -3)
    mask_mirror = (D == -2) & (x == 3)

    if mask_pos.sum() == 0 or mask_mirror.sum() == 0:
        return 0.0

    p_pos = float(chose_a[mask_pos].mean())
    p_mirror = float(chose_a[mask_mirror].mean())
    return p_pos - p_mirror

```

**Observed (real) value:** 0.7300 (var=0.0558)
**Predicted under pi_10:** 0.6250 (var=0.1075)
**Predicted under pi_9:** 0.5800 (var=0.0580)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    crit_a1 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    crit_b1 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_a2 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_b2 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def pair_key(row):
        return (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    mask1 = (keys == (crit_a1, crit_b1)).to_numpy()
    mask2 = (keys == (crit_a2, crit_b2)).to_numpy()
    resp = data['response'].to_numpy()

    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0

    p_a1 = float((resp[mask1] == 0).mean())
    p_a2 = float((resp[mask2] == 0).mean())
    return p_a1 - p_a2

```

**Observed (real) value:** 0.3950 (var=0.3296)
**Predicted under pi_10:** 0.1500 (var=0.1973)
**Predicted under pi_9:** 0.3987 (var=0.0481)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(v) for v in data['option_a_ratings']], dtype=float)
    b = np.array([list(v) for v in data['option_b_ratings']], dtype=float)
    y = 1.0 - data['response'].to_numpy(dtype=float)
    sid = data['subject_id'].to_numpy()

    clean_a_high = ((a[:, :3] == 0) & (b[:, :3] == 1)).sum(axis=1)
    clean_b_high = ((a[:, :3] == 1) & (b[:, :3] == 0)).sum(axis=1)
    h = clean_a_high - clean_b_high

    clean_a_all = ((a == 0) & (b == 1)).sum(axis=1)
    clean_b_all = ((a == 1) & (b == 0)).sum(axis=1)
    D = clean_a_all - clean_b_all

    slopes = []
    for subj in np.unique(sid):
        m = sid == subj
        subj_slopes = []
        for D_val in (2.0, -2.0):
            dm = m & (D == D_val)
            if np.sum(dm) < 2:
                continue
            hh = h[dm].astype(float)
            yy = y[dm]
            hh_c = hh - hh.mean()
            yy_c = yy - yy.mean()
            denom = float(np.sum(hh_c * hh_c))
            if denom > 1e-12:
                subj_slopes.append(float(np.sum(hh_c * yy_c) / denom))
        slopes.append(float(np.mean(subj_slopes)) if subj_slopes else 0.0)

    if not slopes:
        return 0.0
    return float(np.mean(slopes))
```

**Observed (real) value:** 0.0608 (var=0.0035)
**Predicted under pi_10:** 0.0120 (var=0.0004)
**Predicted under pi_9:** 0.0088 (var=0.0006)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def top_group_sign(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = (b[:4] > a[:4]).astype(float) - (a[:4] > b[:4]).astype(float)
        h = float(np.sum(diff))
        if h > 0:
            return 1.0
        if h < 0:
            return -1.0
        return 0.0

    df = data.copy()
    df['sign_h'] = df.apply(top_group_sign, axis=1)
    df['chose_a'] = 1.0 - df['response'].astype(float)

    pos = df.loc[df['sign_h'] > 0, 'chose_a']
    neg = df.loc[df['sign_h'] < 0, 'chose_a']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(pos.mean() - neg.mean())
```

**Observed (real) value:** 0.2492 (var=0.0989)
**Predicted under pi_10:** 0.3746 (var=0.0795)
**Predicted under pi_9:** 0.2429 (var=0.0280)

### Experiment 19
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    if a.size == 0:
        return 0.0

    diff = (b > a).astype(int) - (a > b).astype(int)
    h = diff[:, :4].sum(axis=1).astype(float)
    d = diff.sum(axis=1).astype(int)

    resp = np.asarray(data['response'], dtype=float)
    y = 1.0 - resp

    h_c = h.copy()
    y_c = y.copy()
    for dval in np.unique(d):
        mask = d == dval
        h_c[mask] -= h_c[mask].mean()
        y_c[mask] -= y_c[mask].mean()

    denom = float(np.sum(h_c * h_c))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(h_c * y_c) / denom)

```

**Observed (real) value:** 0.0608 (var=0.0012)
**Predicted under pi_10:** 0.0545 (var=0.0013)
**Predicted under pi_9:** 0.0373 (var=0.0004)

### Experiment 20
**Design**
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def _trend_z_for_subject(sub):
    h_order = [-3.0, -1.0, 1.0, 3.0]
    score_vals = np.tanh(np.array(h_order) / 2.8)
    zs = []

    for d_target in (1.0, -1.0):
        h_list = []
        y_list = []
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = (b > a).astype(float) - (a > b).astype(float)
            D = float(diff.sum())
            if abs(D - d_target) > 1e-9:
                continue
            h = float(diff[0] + diff[1] + diff[2])
            y = 1.0 if float(row['response']) == 0.0 else 0.0
            h_list.append(h)
            y_list.append(y)

        if not h_list:
            zs.append(0.0)
            continue

        h_arr = np.array(h_list, dtype=float)
        y_arr = np.array(y_list, dtype=float)

        r = []
        n = []
        for hv in h_order:
            mask = np.isclose(h_arr, hv)
            r.append(float(y_arr[mask].sum()))
            n.append(float(mask.sum()))

        r = np.array(r, dtype=float)
        n = np.array(n, dtype=float)
        if n.sum() == 0.0:
            zs.append(0.0)
            continue

        pbar = r.sum() / n.sum()
        if pbar <= 0.0 or pbar >= 1.0:
            zs.append(0.0)
            continue

        numerator = np.sum(score_vals * (r - n * pbar))
        denominator = np.sqrt(pbar * (1.0 - pbar) * np.sum(n * score_vals * score_vals))
        zs.append(float(numerator / denominator) if denominator > 1e-12 else 0.0)

    return float(np.mean(zs))

def metric(data: pd.DataFrame) -> float:
    if data is None or data.empty:
        return 0.0
    subject_vals = [_trend_z_for_subject(sub) for _, sub in data.groupby('subject_id')]
    return float(np.mean(subject_vals)) if subject_vals else 0.0
```

**Observed (real) value:** 3.4448 (var=3.3537)
**Predicted under pi_10:** 3.9261 (var=1.6381)
**Predicted under pi_9:** 1.0841 (var=0.9280)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither theory is task-invariant across the full experiment set. pi_10 gets closer on the large-conflict and top-trend signatures it was built for (E1: 0.383 vs 0.342; E19: 0.055 vs 0.061; E20: 3.926 vs 3.445) and generally has a better variance profile, but its always-active tied-top component is far too strong in several non-large-conflict designs (E2: 0.374 vs 0.090; E6: 0.633 vs 0.493; E14: 1.405 vs 1.297; E18: 0.375 vs 0.249), and its special suppression rules cause misses such as E16 (0.150 vs 0.395). pi_9 is close on many tied-top and raw-majority experiments (E2, E6, E7, E10, E11, E14, E16, E18), but its hard gating is structurally wrong: unique-top precedence is unavailable at |D|=3, producing E1=0.145 versus observed 0.342, and the tied-top fallback only fires near E=0, producing the large E20 miss (1.084 vs 3.445). pi_9 also predicts implausibly small between-subject variance in several key experiments (E1 var 0.004 vs 0.052; E8 var 0.037 vs 0.424; E16 var 0.048 vs 0.330). Overall pi_10 is the better architectural starting point because it allows continuous top-coalition influence, but its magnitudes are miscalibrated; pi_9 is degenerate in its gating assumptions and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new theory in which subjects combine a damped validity-weighted inverse-valence tally with a continuously active, smoothly attenuated top-coalition bonus. Unlike pi_9, the top-coalition effect must not be gated by |D|<=window or abs(E)<=tie_tol; it should survive nonzero D at modest strength, so E1 top-following is partial (~0.30-0.40, not 0.145) and E20's h-trend is large (~3.4). Unlike pi_10, the tied-top component should be much weaker and saturating: no large m=2 multiplier of 2.7, no extreme m-based amplifications, and no sharp suppression for three-cue top groups facing five lower cues; it should increase gradually with top-group size and validity gap, decay smoothly with opposition and net tally, and use tanh-type saturation so E2 stays near 0.09 and E16 near 0.40. Unique-top precedence should be probabilistic and still available in large multi-cue conflicts but with conflict-dependent damping that keeps it partial rather than ceiling-level. Large-conflict validity weighting should follow pi_9's damped-lambda style rather than pi_10's boost, letting raw majority dominate when |D| is large while the small top-coalition bonus partially counters it. Finally, subject-level parameters should remain heavy-tailed but with enough variance in the top-coalition and precedence components to reproduce the observed between-subject variances (e.g., E1 ~0.05, E8 ~0.42, E16 ~0.33), avoiding pi_9's near-zero population variance."
}
```

## Usage

```json
{
  "input_tokens": 26548,
  "output_tokens": 15975,
  "reasoning_tokens": 15218,
  "provider": "NextBit"
}
```
