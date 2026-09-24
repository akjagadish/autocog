# Round 6 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_7_1` — KILLED ✗

**Description:** Thresholded anti-validity utility with Gaussian threshold noise and a small validity-consistent soft leak (TAVU-tn). Subjects compute a validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j). A subject-level latent sign dominates: most subjects use negative weighting and make the anti-validity choice when |D| exceeds a subject-level threshold tau_i, but threshold placement is noisy across trials, so the aggregate threshold boundary is graded rather than a step. Within the dead zone and below the sign-following ceiling, subjects fall back on a subject-level bias. A small plain-D sigmoidal leak in the validity-consistent direction weakly damps extreme anti choices at large |D|. A uniform lapse adds residual noise. The dominant process remains thresholded sign, not continuous magnitude accumulation.

**Rationale:** Minimal-diff update on the accepted iter-2 TAVU-tn base. The only changed knobs are the threshold placement and gradation: tau is moved to [0.16, 0.19] and tau_sd tightened to [0.02, 0.04], following the iter-7 diagnostic that Experiment 12's |D|~0.13 rows must stay below threshold while Experiment 8's low-mask rows near |D|~0.20 and Experiment 6's low-mask rows cross. All other ranges are frozen: follow_max remains a constant [0.66, 0.74], p_negative/bias/lapse/soft leak unchanged, so the suprathreshold ceiling and dead-zone fallback architecture are not altered. This intermediate tau band is intended to shrink the moderate-|D| over-adoption in E6/E8/E11/E12 without reintroducing the rejected too-low tau [0.08,0.11] or too-high tau [0.30,0.38] regimes, and without adopting the monotone graded-follow ramp that the most recent iteration's rejected candidate showed manufactures an upward E7 slope.

**Parameters:**
  - `gamma`: `[1.8, 2.2]`
  - `tau`: `[0.16, 0.19]`
  - `tau_sd`: `[0.02, 0.04]`
  - `p_negative`: `[0.88, 0.94]`
  - `sign_draw`: `[0.0, 1.0]`
  - `bias`: `[0.46, 0.54]`
  - `lapse`: `[0.12, 0.16]`
  - `soft_weight`: `[0.00, 0.03]`
  - `soft_beta`: `[0.7, 1.1]`
  - `follow_max`: `[0.66, 0.74]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np
import math


def predict(parameters, state, history):
    """Thresholded anti-validity utility with Gaussian threshold noise (TAVU-tn).

    D = sum_j v_j^gamma * (A_j - B_j). A subject-level sign dominates:
    most subjects apply a negative validity weighting and choose the
    anti-validity option when |D| exceeds a subject-level threshold.
    Threshold placement is noisy across trials, implemented analytically
    as p_above = Phi((|D| - tau) / tau_sd). Inside the dead zone, and
    below the sign-following ceiling, choices fall back on a subject-level
    bias. A small plain-D sigmoidal leak weakly opposes the anti-validity
    sign at large |D|, and a uniform lapse adds residual noise.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "TAVU-tn expects a (2, n_features) stimulus; "
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
    tau = float(parameters["tau"])
    tau_sd = float(parameters["tau_sd"])
    p_negative = float(parameters["p_negative"])
    sign_draw = float(parameters["sign_draw"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    w_soft = float(parameters["soft_weight"])
    beta_soft = float(parameters["soft_beta"])
    follow_max = float(parameters["follow_max"])

    # Latent validity-sign mixture: most subjects are negative weighters.
    sign = -1.0 if sign_draw < p_negative else 1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Analytic Gaussian threshold noise: probability that |D| crosses tau
    # when the trial-level threshold is perturbed.
    z_th = (abs(d) - tau) / tau_sd
    p_above = 0.5 * (1.0 + math.erf(z_th / math.sqrt(2.0)))
    p_above = float(np.clip(p_above, 0.0, 1.0))

    # Thresholded anti-validity target for p(A).
    if sign < 0.0:
        if d > 0.0:
            target_a = 0.0
        elif d < 0.0:
            target_a = 1.0
        else:
            target_a = bias
    else:
        if d > 0.0:
            target_a = 1.0
        elif d < 0.0:
            target_a = 0.0
        else:
            target_a = bias

    # Dominant thresholded sign rule, with a follow ceiling that keeps
    # some residual bias even after threshold crossing.
    core_threshold = (
        follow_max * (p_above * target_a + (1.0 - p_above) * bias)
        + (1.0 - follow_max) * bias
    )

    # Small validity-consistent soft leak. This is deliberately very weak
    # below |D| = 0.2 and only mildly damps extreme anti choices.
    z = beta_soft * d
    if z >= 0.0:
        soft_a = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        soft_a = ez / (1.0 + ez)

    core_a = (1.0 - w_soft) * core_threshold + w_soft * soft_a
    p_a = (1.0 - lapse) * core_a + lapse / 2.0
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

### `pi_8` → slot 1 (via `new_theory`)

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
