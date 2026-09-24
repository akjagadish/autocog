# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_9` — SURVIVED ✓

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


### slot 2 — `pi_5` — KILLED ✗

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

### `pi_10` → slot 2 (via `new_theory`)

**Description:** Gated compressive anti-validity with validity-curvature gain and sign-selectable dead-zone repair. Each subject forms D = sum_j v_j^gamma * (A_j - B_j), passes D through a saturating transform D/(tau + |D|), and mostly applies a negative validity-weighting sign. A subject-level threshold theta defines a mild dead zone around D=0: inside this zone the anti-validity score is attenuated and augmented by a bounded graded repair D/(kappa + |D|) whose sign is anti-validity for most subjects but validity-consistent for a minority; outside the zone the full compressive score plus bias is used. A two-component subject mixture provides moderate bulk responding with stronger validity curvature plus a near-deterministic tail with negative or sub-linear validity exponents. There is no strong unweighted tally switch and no fragmented-evidence dilution; only a small tally leak eta*T is retained.

**Rationale:** This is a minimal in-family edit of the accepted gated compressive anti-validity candidate. The core family is preserved: D = sum_j v_j^gamma * (A_j - B_j), saturating transform D/(tau+|D|), mostly negative sign mixture, thresholded dead zone, no n_disc term, and only a small eta tally leak. Four targeted changes are made. (1) Bulk validity curvature is increased to gamma_bulk 1.30-1.80, while lambda_bulk, beta_bulk, and tau_bulk are left unchanged, so top-cue anti-validity contrasts recover in Experiments 1, 2, 4, and 9 without the prior mistake of softening the core. (2) The dead-zone repair is switched from a constant sign(D) term to a bounded graded repair D/(kappa_repair+|D|), which restores low-evidence graded anti-validity contrast in Experiments 13 and 14 while keeping the small-D step in Experiment 12 bounded. (3) An independent repair-sign draw gives a minority of subjects a validity-consistent repair inside the dead zone, providing the threshold-crossing route needed for Experiment 18 without adding a separate full-D positive-sign branch. (4) Theta heterogeneity is widened to allow more varied threshold crossings. The repairs remain small/bounded, no strong tally switch is introduced, and the tail/gamma heterogeneity that generated Experiment 16 row separation is retained.

**Parameters:**
  - `comp_draw`: `[0.0, 1.0]`
  - `p_tail`: `[0.12, 0.20]`
  - `gamma_bulk`: `[1.30, 1.80]`
  - `theta_bulk`: `[0.06, 0.28]`
  - `tau_bulk`: `[0.80, 1.80]`
  - `lambda_bulk`: `[1.00, 1.80]`
  - `beta_bulk`: `[1.20, 2.50]`
  - `bias_bulk`: `[-0.05, 0.05]`
  - `rho_bulk`: `[0.10, 0.22]`
  - `c_repair_bulk`: `[0.03, 0.10]`
  - `gamma_tail`: `[-1.80, 0.50]`
  - `theta_tail`: `[0.00, 0.08]`
  - `tau_tail`: `[0.08, 0.30]`
  - `lambda_tail`: `[1.50, 3.00]`
  - `beta_tail`: `[4.00, 8.00]`
  - `bias_tail`: `[-0.04, 0.04]`
  - `rho_tail`: `[0.50, 0.80]`
  - `c_repair_tail`: `[0.15, 0.45]`
  - `p_positive`: `[0.02, 0.08]`
  - `sign_draw`: `[0.0, 1.0]`
  - `lapse`: `[0.01, 0.04]`
  - `eta`: `[0.0, 0.04]`
  - `p_repair_positive`: `[0.12, 0.25]`
  - `repair_sign_draw`: `[0.0, 1.0]`
  - `c_repair_pos`: `[0.20, 0.60]`
  - `kappa_repair`: `[0.03, 0.10]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            'GCAV expects a (2, n_features) stimulus; got shape '
            + str(stim.shape) + '.'
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters['validities'], dtype=float)
    if v.size != a.size:
        raise ValueError('validities length does not match n_features.')

    comp_draw = float(parameters['comp_draw'])
    p_tail = float(parameters['p_tail'])

    if comp_draw < p_tail:
        gamma = float(parameters['gamma_tail'])
        theta = float(parameters['theta_tail'])
        tau = float(parameters['tau_tail'])
        lam = float(parameters['lambda_tail'])
        beta = float(parameters['beta_tail'])
        bias = float(parameters['bias_tail'])
        rho = float(parameters['rho_tail'])
        c_repair = float(parameters['c_repair_tail'])
    else:
        gamma = float(parameters['gamma_bulk'])
        theta = float(parameters['theta_bulk'])
        tau = float(parameters['tau_bulk'])
        lam = float(parameters['lambda_bulk'])
        beta = float(parameters['beta_bulk'])
        bias = float(parameters['bias_bulk'])
        rho = float(parameters['rho_bulk'])
        c_repair = float(parameters['c_repair_bulk'])

    lapse = float(parameters['lapse'])
    eta = float(parameters['eta'])
    p_positive = float(parameters['p_positive'])
    sign_draw = float(parameters['sign_draw'])

    p_repair_positive = float(parameters['p_repair_positive'])
    repair_sign_draw = float(parameters['repair_sign_draw'])
    c_repair_pos = float(parameters['c_repair_pos'])
    kappa_repair = float(parameters['kappa_repair'])

    # Mostly negative validity-weighting sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Dead-zone repair sign is also mostly anti-validity, but a minority
    # receive a validity-consistent repair inside the dead zone.
    repair_sign = 1.0 if repair_sign_draw < p_repair_positive else -1.0

    diff = a - b
    D = float(np.dot(np.power(v, gamma), diff))
    T = float(np.mean(diff))

    sat = D / (tau + abs(D))
    core = sign * lam * sat + bias

    if abs(D) < theta:
        # Bounded graded repair, applied identically for all tallies.
        repair_sat = D / (kappa_repair + abs(D))
        if repair_sign > 0:
            repair = c_repair_pos * repair_sat
        else:
            repair = -c_repair * repair_sat
        score = rho * core + repair + eta * T
    else:
        score = core + eta * T

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
