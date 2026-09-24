# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Negative Validity-Weighted Additive (NWA): people combine all binary feature differences linearly, but assign negative weight to a feature advantage, and the absolute weight magnitude increases with the feature's stated validity. Formally, choice is based on S = sum_j w_j * (A_j - B_j), with w_j = -v_j^gamma (gamma > 0), so an option's cue advantage makes that option less likely to be chosen. Choice probability uses a softmax/sigmoid over this score with inverse temperature beta, plus a small uniform lapse epsilon.

**Rationale:** Previous theories failed because TTB produced a positive top-cue relation in Experiment 2 and Tallying produced a positive lower-cue tally relation in Experiment 1. NWA reverses both predicted relations by assigning negative validity-scaled weights to all cue advantages. Using all cues gives the negative lower-cue tally effect in Experiment 1, while validity-scaled negative weights generate the anti-TTB relation in Experiment 2. The gamma exponent around 2.25 compresses the validities enough to make lower cues compete with the top cue, and the moderately high beta around 2.7 yields graded but strongly anti-cue responding. A very small lapse preserves realistic stochasticity without destroying the predicted negative relations.

**Parameters:**
  - `beta`: `[2.65, 2.75]`
  - `gamma`: `[2.22, 2.28]`
  - `epsilon`: `[0.0, 0.02]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # NWA: negative validity-weighted additive integration of all cues.
    # state is shape (2, n_features): row 0 = option A, row 1 = option B.
    # history is ignored because there is no trial-by-trial feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"NWA expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Negative validity-scaled weights: an advantage on a cue is evidence
    # against that option, with stronger validity cues carrying more weight.
    weights = -(v ** gamma)

    score = float(np.dot(weights, a - b))

    # Stable sigmoid for p(A | score).
    z = beta * score
    if z >= 0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy(probs)`:**
```python
def policy(probs):
    p = np.asarray(probs, dtype=float)
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

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB). People order cues by stated validity and normally use the first discriminating cue, inverting its recommendation with subject-specific probability q_i. Attentional slips to later ordered cues are not driven by feature count alone; they occur mainly when the first discriminator's validity is below a threshold, and are strongly suppressed for first discriminators with validity 0.90 or higher. When a slip occurs, the later discriminating cue controls the response with a boosted inversion probability, reflecting stronger contra-indicator responding in the low-validity regime. Trials with no discriminating cue are resolved by subject-level bias, and a small lapse adds residual noise. There is no graded validity-weighted accumulation and no continuous logit over a compensatory score.

**Rationale:** This is a minimal edit of the prior SS-ITTB candidate. The main change is that late-cue slip is now validity-gated rather than feature-count-only: slip is allowed for first discriminators below gate_low, ramps down toward 0.90, and is zero at 0.90 or higher. This preserves the within-sign sensitivity needed for Experiment 1 while removing the slip that diluted the strong first-cue wrong-way effects in Experiments 3 and 5. The gated slip branch is strengthened by raising late2/late3 probabilities, increasing max_slip, and adding q_low_boost, which boosts inversion only when the first discriminator is in the gated low-validity regime. q_mid is raised to 0.55-0.68 to increase low-validity inversion, while q_90, q_92, and q_hi keep their previous centers to protect Experiments 2, 4, and 7. q_90, q_hi, and bias ranges are broadened to restore more realistic between-subject variance without changing their central tendencies. The model remains strictly lexicographic and non-compensatory.

**Parameters:**
  - `q_lo`: `[0.20, 0.40]`
  - `q_mid`: `[0.55, 0.68]`
  - `q_90`: `[0.66, 0.82]`
  - `q_92`: `[0.48, 0.62]`
  - `q_hi`: `[0.70, 0.86]`
  - `q_floor`: `[0.45, 0.65]`
  - `bias`: `[0.30, 0.70]`
  - `epsilon`: `[0.02, 0.08]`
  - `late2_base`: `[0.02, 0.08]`
  - `late2_slope`: `[0.80, 1.10]`
  - `late3_base`: `[0.02, 0.06]`
  - `late3_slope`: `[0.10, 0.25]`
  - `gate_low`: `[0.84, 0.88]`
  - `q_low_boost`: `[0.10, 0.25]`
  - `max_slip`: `[0.93, 0.98]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB).

    Cues are ordered by stated validity. The normative first
    discriminating cue sets the subject's base inversion probability
    q_norm. Attentional slips to later ordered cues are gated by the
    validity of that first discriminator: slips are allowed and
    strengthened when validity is below gate_low, ramp down between
    gate_low and 0.90, and are switched off at 0.90 or above. Slipped
    responses use a boosted inversion strength q_slip. If no cue
    discriminates, a subject-level bias is used. No graded cue
    accumulation enters the model.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "GSS-ITTB expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    n = int(a.size)

    q_lo = float(parameters["q_lo"])
    q_mid = float(parameters["q_mid"])
    q_90 = float(parameters["q_90"])
    q_92 = float(parameters["q_92"])
    q_hi = float(parameters["q_hi"])
    q_floor = float(parameters["q_floor"])
    bias = float(parameters["bias"])
    epsilon = float(parameters["epsilon"])
    late2_base = float(parameters["late2_base"])
    late2_slope = float(parameters["late2_slope"])
    late3_base = float(parameters["late3_base"])
    late3_slope = float(parameters["late3_slope"])
    gate_low = float(parameters["gate_low"])
    q_low_boost = float(parameters["q_low_boost"])
    max_slip = float(parameters["max_slip"])

    def q_of(vv):
        vv = float(vv)
        if vv > 0.97:
            return float(np.clip(q_floor, 0.0, 1.0))
        return float(np.interp(
            vv,
            np.array([0.50, 0.80, 0.90, 0.92, 0.95], dtype=float),
            np.array([q_lo, q_mid, q_90, q_92, q_hi], dtype=float),
        ))

    cue_order = np.argsort(-v, kind="stable")

    def first_disc(start_idx):
        for r in range(start_idx, n):
            j = int(cue_order[r])
            if a[j] != b[j]:
                return j
        return -1

    j0 = first_disc(0)

    if j0 < 0:
        # No discriminating cue anywhere: subject-level bias.
        p_core = bias
    else:
        # Base inversion strength is calibrated by the normative first
        # discriminating cue.
        q_norm = q_of(v[j0])

        # Feature-count pressure toward late search.
        extra = max(0.0, float(n - 5))
        late2 = float(np.clip(late2_base + late2_slope * extra, 0.0, 0.90))
        late3 = float(np.clip(late3_base + late3_slope * extra, 0.0, 0.60))

        # Validity gate on slip: full slip below gate_low, zero slip at
        # validity 0.90 or higher, linear ramp between.
        vj0 = float(v[j0])
        if vj0 < gate_low:
            gate = 1.0
        elif vj0 >= 0.90:
            gate = 0.0
        else:
            gate = (0.90 - vj0) / (0.90 - gate_low)
        gate = float(np.clip(gate, 0.0, 1.0))

        late2 *= gate
        late3 *= gate

        if late2 + late3 > max_slip:
            scale = max_slip / (late2 + late3)
            late2 *= scale
            late3 *= scale

        p0 = 1.0 - late2 - late3

        # Slipped-to cues are inverted more strongly in the gated regime.
        q_slip = float(np.clip(q_norm + q_low_boost * gate, 0.0, 1.0))

        def direction_prob(start_idx, q_val):
            j = first_disc(start_idx)
            if j < 0:
                return bias
            if a[j] > b[j]:
                # Effective cue favors A, so inverted tendency chooses B.
                return 1.0 - q_val
            # Effective cue favors B, so inverted tendency chooses A.
            return q_val

        p_core = (
            p0 * direction_prob(0, q_norm)
            + late2 * direction_prob(2, q_slip)
            + late3 * direction_prob(3, q_slip)
        )

    p_a_core = float(np.clip(p_core, 0.0, 1.0))

    # Uniform lapse over the two options.
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
