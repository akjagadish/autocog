# Round 7 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_8` — KILLED ✗

**Description:** Conflict-modulated continuous anti-validity with inverse-validity conflict counting and a tally-regularized response (CMCAV-iv). Subjects form a continuous validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j), mostly apply a negative validity-weighting sign, and respond through a fluid evidence-gated continuous transform S_i = -lambda_i * G(C) * D / (kappa_i + abs(D) + r_i) + bias_i + w_t * T. There is no fixed threshold, no dead zone, and no single fixed half-saturation constant. The conflict gain G(C) depends on both concentration of the dominant validity-weighted contribution and an inverse-validity-weighted count of disagreeing features, so conflict from many low-validity cues strongly suppresses responding while concentrated high-validity evidence remains effective. A small unidirectional tally term T = (1/n) * sum_j (A_j - B_j) with negative weight supports count-driven anti-validity gradients.

**Rationale:** This is a minimal, targeted edit of the previous CMCAV-c candidate within the same conflict-modulated continuous anti-validity family. First, the raw disagreement-count breadth penalty n_disc was replaced by an inverse-validity weighted count of disagreeing features, sum_j (1 - v_j)^eta. This stops high-validity disagreements from being suppressed like a generic feature count while strongly penalizing broad jumbled low-validity conflict, targeting the Experiment 12 over-response without emasculating count-driven designs such as Experiment 1. Second, the sign(D)-conditional tally weights were removed in favor of a single negative tally weight, so the tally term always opposes the count-based validity direction rather than sometimes canceling it; this directly repairs the Experiment 1 regression noted by the critic. Third, kappa was reduced to [0.02, 0.06] and g_range raised to [2.5, 3.2], strengthening small-|D| orientation contrast in Experiments 9 and 13 and otherwise weak concentration-driven contrasts, as advised. No fixed threshold, no fixed half-saturation constant, and no dead zone were introduced; the response remains continuous and probabilistic with a mostly negative subject-level validity-weighting sign.

**Parameters:**
  - `gamma`: `[1.6, 2.4]`
  - `lam`: `[1.3, 2.2]`
  - `beta`: `[1.5, 2.5]`
  - `bias`: `[-0.06, 0.06]`
  - `lapse`: `[0.03, 0.09]`
  - `kappa`: `[0.02, 0.06]`
  - `r`: `[0.00, 0.02]`
  - `p_positive`: `[0.02, 0.10]`
  - `sign_draw`: `[0.0, 1.0]`
  - `w_t`: `[-0.32, -0.14]`
  - `rho`: `[5.0, 8.0]`
  - `eta`: `[1.5, 2.0]`
  - `conc_exp`: `[1.4, 1.9]`
  - `nu`: `[0.5, 1.0]`
  - `xi`: `[0.0, 0.20]`
  - `g_min`: `[0.02, 0.06]`
  - `g_range`: `[2.5, 3.2]`
  - `c_half`: `[0.4, 0.7]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    """Conflict-modulated continuous anti-validity with inverse-validity
    conflict counting and a single negative tally term (CMCAV-iv).

    D = sum_j v_j^gamma * (A_j - B_j)
    S = sign * lam * G(C) * D / (kappa + abs(D) + r)
        + bias + w_t * (1/n) * sum_j (A_j - B_j)

    G(C) uses a concentration-style conflict gain whose conflict
    denominator is an inverse-validity weighted count of disagreeing
    features, not a raw disagreement count.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CMCAV-iv expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    lam = float(parameters["lam"])
    beta = float(parameters["beta"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    kappa = float(parameters["kappa"])
    r = float(parameters["r"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])

    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    conc_exp = float(parameters["conc_exp"])
    nu = float(parameters["nu"])
    xi = float(parameters["xi"])
    g_min = float(parameters["g_min"])
    g_range = float(parameters["g_range"])
    c_half = float(parameters["c_half"])
    w_t = float(parameters["w_t"])

    # Mostly negative validity-weighting sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted contributions.
    w = np.power(v, gamma)
    diff = a - b
    contrib = w * diff
    D = float(np.sum(contrib))

    mag = np.abs(contrib)
    M = float(np.sum(mag))
    disc_mask = mag > 1e-12

    # Concentration-style conflict gain.
    if M <= 1e-12:
        G = g_min + 0.5 * g_range
    else:
        dom_idx = int(np.argmax(mag))
        max_m = float(mag[dom_idx])
        v_dom = float(v[dom_idx])

        # Dominant share of total validity-weighted evidence.
        cohesion = max_m / M

        # Inverse-validity weighted disagreement count. High-validity
        # disagreeing cues are nearly free, while low-validity disagreeing
        # cues build conflict and suppress the gain.
        inv_validity = np.power(1.0 - v, eta)
        low_conflict = float(np.sum(inv_validity[disc_mask]))
        breadth = 1.0 + rho * low_conflict

        C = (cohesion ** conc_exp) * (v_dom ** nu)
        # Small residual coherence exponent so broad but internally
        # consistent evidence is not fully suppressed.
        C *= ((abs(D) / M) ** xi)
        C /= breadth
        C = float(np.clip(C, 0.0, 20.0))

        G = g_min + g_range * C / (c_half + C)

    G = float(np.clip(G, 0.0, 3.0))

    # Single negative tally term: p(A) is pushed against the count-based
    # validity direction regardless of D's sign.
    T = float(np.mean(diff))

    # Continuous anti-validity response: no fixed threshold and no one
    # fixed half-saturation constant.
    sat = D / (kappa + abs(D) + r)
    score = sign * lam * G * sat + bias + w_t * T
    z = beta * score

    # Numerically stable sigmoid.
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    p_a = (1.0 - lapse) * p_a_core + lapse / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

**Rationale:** The previous CNVW candidate used psi(D) = D / (1 + abs(D)), which was too close to linear over the validity-weighted difference values in these experiments. That preserved spurious within-sign graded contrasts, producing E6 = 0.131 versus the real near-zero 0.012 and underpowering the opposite-sign E4 effect. Minimal fix: keep the same no-tally/no-conflict family, but replace the transform with psi(D) = D / (tau + abs(D)) using a small subject-level half-saturation constant tau in [0.10, 0.25], matching the critic's prescribed range. This makes most non-negligible D values saturate near the sign bound, so sign dominates choice while magnitude effects are greatly compressed. That should collapse the E6 within-sign positive-D contrast toward zero, strengthen the opposite-sign E4 contrast toward the observed ~0.53, and push E1/E2 further in the negative direction. Per the feedback, beta and lambda ranges are initially kept unchanged; the sharpened transform alone does the necessary separation without needing higher gain.

**Parameters:**
  - `gamma`: `[1.5, 2.5]`
  - `tau`: `[0.10, 0.25]`
  - `lambda`: `[0.8, 2.0]`
  - `beta`: `[0.8, 2.0]`
  - `bias`: `[-0.25, 0.25]`
  - `epsilon`: `[0.03, 0.10]`
  - `p_positive`: `[0.03, 0.10]`
  - `sign_draw`: `[0.0, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Compressive negative validity-weighted integration with subject-level tau.

    D = sum_j v_j^gamma * (A_j - B_j)
    psi(D) = D / (tau + abs(D))
    S = sign * lambda * psi(D) + bias
    p(A) = lapse/2 + (1-lapse) * sigmoid(beta * S)

    Most subjects use sign = -1; a minority draw sign = +1.
    history is intentionally ignored because there is no feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CNVW-tau expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    lam = float(parameters["lambda"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    tau = float(parameters["tau"])

    # Latent sign mixture: most subjects use negative-valence weighting;
    # a minority trust positive-valence evidence and get a positive sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Subject-level strongly saturating bounded transform. The small
    # half-saturation constant tau forces most non-negligible validity-
    # weighted differences toward the sign bound, so sign and not
    # continuous magnitude drives the main choice contrast.
    psi = d / (tau + abs(d))

    score = sign * lam * psi + bias
    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy(probs)`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## Replacement

### `pi_9` → slot 1 (via `new_theory`)

**Description:** Thresholded validity-weighted anti-validity with a zero-tally dead-zone repair and fragmented-evidence dilution. Each subject forms D = sum_j v_j^{gamma_i} (A_j - B_j). When |D| is below a subject-level ambiguity threshold theta_i, the validity signal is treated as ambiguous. If the unweighted tally is nonzero, choice is driven by that tally through omega_i with a single strong weight, exactly as in the accepted base. If the tally is exactly zero, choice remains graded via a small signed validity-derived anti-validity term psi_i * sign_i * D / (kappa_i + |D|), but this term is now diluted multiplicatively by exp(-rho_disc_i * max(0, n_disc - 2)), where n_disc is the number of discriminating features. This dilution suppresses responding for zero-tally rows with many simultaneously disagreeing features while leaving zero-tally rows with only two discriminating features untouched. When |D| >= theta_i, choice follows the mostly negative anti-validity transform sign_i * lambda_i * D / (tau_i + |D|) + bias_i. Choice probability is p(A) = lapse_i/2 + (1-lapse_i) * sigmoid(beta_i * S_i).

**Rationale:** This edit follows the most recent critic feedback exactly. The accepted iter-2 base already solved E16, E13, E14, E9, E4 and E15 with its single strong nonzero-tally fallback and its zero-tally graded term. Every subsequent rejected edit failed because it modified the nonzero-tally arm and destroyed E16 and E13. Here I keep the base's tally arm and every other parameter untouched, and I add only a multiplicative dilution factor to the zero-tally branch: score = psi * sign * D / (kappa_dead + |D|) * exp(-rho_disc * max(0, n_disc - 2)), active only for |D| < theta and T = 0. The dilution is designed to fix the base's one dominant residual, E12, where many zero-tally rows contain four or eight discriminating features and the base was far too decisive (0.616 vs human 0.036). With rho_disc around 0.9-1.4, those high-n_disc zero-tally rows are pushed back toward chance, while n_disc=2 rows (E13 and E14) are completely unaffected because the exponent is zero. E16's rows all have nonzero T and therefore never enter the zero-tally branch, so its row-logit variance remains 3.597. No other parameter is changed, preserving the base's successes on E1, E2, E4, E9, E10, E13, E14 and E15. This is the minimal, targeted repair the gate should accept if E12 falls to roughly 0.03-0.06 without collateral damage.

**Parameters:**
  - `gamma`: `[1.0, 3.0]`
  - `theta`: `[0.15, 0.28]`
  - `tau`: `[0.70, 1.30]`
  - `lam`: `[1.60, 2.20]`
  - `beta`: `[2.10, 2.70]`
  - `bias`: `[-0.05, 0.05]`
  - `lapse`: `[0.01, 0.04]`
  - `p_positive`: `[0.03, 0.10]`
  - `sign_draw`: `[0.0, 1.0]`
  - `omega_mag`: `[0.45, 0.75]`
  - `p_omega_positive`: `[0.85, 0.95]`
  - `omega_sign_draw`: `[0.0, 1.0]`
  - `psi`: `[0.90, 1.40]`
  - `kappa_dead`: `[0.06, 0.16]`
  - `rho_disc`: `[0.9, 1.4]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    """Thresholded validity-weighted anti-validity with zero-tally dead-zone repair
    and fragmented-evidence dilution.

    D = sum_j v_j^gamma * (A_j - B_j)
    T = sum_j (A_j - B_j)
    n_disc = count of features where A_j != B_j

    If |D| < theta:
        if T == 0:
            dilution = exp(-rho_disc * max(0, n_disc - 2))
            score = psi * sign * D / (kappa_dead + |D|) * dilution
        else:
            score = omega * T
    Else:
        score = sign * lam * D / (tau + |D|) + bias

    p(A) = lapse/2 + (1 - lapse) * sigmoid(beta * score)

    history is intentionally ignored because the task provides no trial-by-trial
    correctness feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Thresholded anti-validity model expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    tau = float(parameters["tau"])
    lam = float(parameters["lam"])
    beta = float(parameters["beta"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    omega_mag = float(parameters["omega_mag"])
    p_omega_positive = float(parameters["p_omega_positive"])
    omega_sign_draw = float(parameters["omega_sign_draw"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    psi = float(parameters["psi"])
    kappa_dead = float(parameters["kappa_dead"])
    rho_disc = float(parameters["rho_disc"])

    # Mostly negative validity-weighting sign; a minority trust the
    # validity-weighted direction.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Subject-level tally weight, mostly positive but allowing a minority of
    # subjects to use the opposite tally sign.
    omega = omega_mag if omega_sign_draw < p_omega_positive else -omega_mag

    # Subject-level validity-weighted feature difference.
    w = np.power(v, gamma)
    diff = a - b
    D = float(np.dot(w, diff))

    # Unweighted tally difference used in the dead zone.
    T = float(np.sum(diff))

    # Number of discriminating features, used only in the zero-tally branch.
    n_disc = int(np.count_nonzero(diff))

    if abs(D) < theta:
        # Ambiguous validity signal.
        if abs(T) < 1e-12:
            # Zero-tally repair: equal raw counts no longer force p(A)=0.5.
            # A small signed validity-derived anti-validity term preserves
            # graded responding when the options differ only in which cue is
            # present.  This term is diluted by a factor that shrinks
            # exponentially with the number of discriminating features beyond
            # two, so multi-feature zero-tally rows stay near chance while
            # two-feature rows keep the graded contrast.
            sat_dead = D / (kappa_dead + abs(D))
            dilution = np.exp(-rho_disc * max(0.0, float(n_disc - 2)))
            score = psi * sign * sat_dead * dilution
        else:
            # Nonzero tally drives the ambiguous-regime choice.
            score = omega * T
    else:
        # Clear validity signal: mostly negative anti-validity response through
        # a moderately saturating continuous transform.
        sat = D / (tau + abs(D))
        score = sign * lam * sat + bias

    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - lapse) * p_a_core + lapse / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy(probs)`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```
