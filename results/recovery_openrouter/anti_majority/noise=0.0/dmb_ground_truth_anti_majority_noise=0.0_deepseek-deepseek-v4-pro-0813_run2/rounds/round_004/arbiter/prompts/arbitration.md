# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB). People order cues by stated validity and normally use the first discriminating cue, inverting its recommendation with subject-specific probability q_i. Attentional slips to later ordered cues are not driven by feature count alone; they occur mainly when the first discriminator's validity is below a threshold, and are strongly suppressed for first discriminators with validity 0.90 or higher. When a slip occurs, the later discriminating cue controls the response with a boosted inversion probability, reflecting stronger contra-indicator responding in the low-validity regime. Trials with no discriminating cue are resolved by subject-level bias, and a small lapse adds residual noise. There is no graded validity-weighted accumulation and no continuous logit over a compensatory score.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


## THEORY 2 — pi_5
Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.79]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** The design pits a first-discriminating-cue inversion process (GSS-ITTB) against a continuous validity-weighted integration process with saturation (CNVW-tau). With validities [0.90, 0.80, 0.70, 0.60, 0.55], the 0.90 expert discriminates on every unique pair, so GSS-ITTB's attentional gate is zero and no later cue can affect the response. GSS-ITTB therefore predicts only two flat choice probabilities: about 0.27 for A when the 0.90 cue favors A, and about 0.73 for A when the 0.90 cue favors B, with no effect of the lower-cue balance. CNVW-tau instead aggregates D = sum_j v_j^gamma*(A_j-B_j) and most subjects invert its sign. In the six top-A pairs, D takes approximately +0.789, +0.184, +0.472, -0.264, -0.581, -0.841; under central CNVW-tau parameters this gives p(A) near .18, .28, .21, .75, .80, .82, crossing from choosing B to choosing A while GSS-ITTB stays at about .27 throughout. The six top-B mirrors have the complementary D signs and likewise force CNVW-tau to cross from high A choice to low A choice while GSS-ITTB stays at about .73. The critical dissociation is the top-A negative-D trials and the top-B positive-D trials: GSS-ITTB continues choosing opposite the first cue, whereas CNVW-tau reverses because the lower-validity majority flips the sign of D. This sign-reversal signature cannot be produced by a flat first-cue rule regardless of q_90, and cannot be flattened by any CNVW-tau tau/lambda/beta/bias/lapse tuning while preserving the D-dependence.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Score is mean probability of choosing A on trials with negative validity-weighted difference D minus mean p(A) on trials with positive D, where D = sum_j validities_j^2 * (A_j - B_j). The 12 unique pairs split into six negative-D and six positive-D pairs, and each group contains three top-A and three top-B trials. Under GSS-ITTB the 0.90 cue discriminates on every trial and its slip gate is zero, so choices depend only on whether that cue favors A or B. Because both D-sign groups are balanced over top-A and top-B trials, both group means are 0.5 in expectation, giving a score of exactly 0 under GSS-ITTB. Under CNVW-tau most subjects use a negative sign, so negative-D trials produce high p(A) and positive-D trials produce low p(A), yielding a large positive score. Averaging 48 trials per D-sign group keeps the per-subject binomial noise small.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.79], dtype=float)
    gamma = 2.0
    weights = validities ** gamma

    def d_sign(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        d = float(np.dot(weights, a - b))
        if d > 0.0:
            return 1
        if d < 0.0:
            return -1
        return 0

    signs = data.apply(d_sign, axis=1)
    p_a = 1.0 - data['response'].astype(float)

    neg_mean = p_a[signs == -1].mean()
    pos_mean = p_a[signs == 1].mean()

    return float(neg_mean - pos_mean)

### RESULTS
- Predicted under pi_6 (simulated): 0.0171 (var=0.0067)
- Predicted under pi_5 (simulated): 0.4529 (var=0.1083)
- Observed on real data: 0.6008 (var=0.0611)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.95, 0.92, 0.9, 0.62, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 6: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1]
  trial 10: A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 12: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]

**Rationale:** This design runs a first-discriminating-cue validity sweep while keeping the validity-weighted difference D positive in the first six trials. The validities are [0.95, 0.92, 0.90, 0.62, 0.50], so at gamma=2.0 the CNVW-tau weights are approximately 0.9025, 0.8464, 0.8100, 0.3844, 0.2500. In trials 1-6 the first discriminating cue always favors A and D equals 0.9025, 0.8464, 0.8100, 0.6344, 0.2500, and 0.1344. With central CNVW-tau parameters (tau=0.175, lambda=1.4, beta=1.4, negative sign) the predicted p(A) values are about 0.18, 0.18, 0.18, 0.19, 0.25, 0.31, a low flat-to-graded pattern because psi(D)=D/(tau+abs(D)) saturates for large positive D and most subjects invert the sign. GSS-ITTB stops at the first discriminating cue and does not integrate the later cues continuously. Its first-cue q calibration is nonmonotonic across the first three validity levels, and it becomes strongly inverted for low-validity first cues, predicting p(A) of about 0.23, 0.45, 0.27, 0.55, 0.68, 0.55 for the same six trials. Trials 4 and 6 are especially diagnostic: they share the same first discriminator (0.62 favoring A), but trial 6 changes the later 0.50 cue against A. GSS-ITTB predicts about 0.55 on both trials, whereas CNVW-tau moves from about 0.19 to 0.31 because D drops from +0.63 to +0.13. The last six trials are the exact A/B mirrors, giving complementary high-side predictions and a symmetry check. This produces a rank-order and within-first-cue dissociation: CNVW-tau cannot produce the GSS zigzag or the majority-A choices on D-positive low-first-cue trials, while GSS-ITTB cannot produce the continuous D-driven separation between trials 4 and 6. There are 12 unique pairs, K=8, giving 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric targets the GSS-ITTB non-monotonicity across the three highest first-cue validities 0.95, 0.92, and 0.90. It estimates the anti-first-cue choice rate for each validity, defined as the rate of choosing the option opposite to the first discriminating cue. Both mirror directions are pooled: when A is all ones and B has one zero at the target feature, anti-cue is response B; when A has one zero and B is all ones, anti-cue is response A. This doubles the observations per subject per validity to 16 and removes directional A/B bias. With r95, r92, and r90 as the three anti-cue rates, the metric is r95 - 2*r92 + r90. GSS-ITTB assigns q values of about 0.78, 0.55, and 0.74 to validities 0.95, 0.92, and 0.90, so this curvature is approximately 0.78 - 2*0.55 + 0.74 = 0.42 before lapse compression, and about 0.40 after lapse. CNVW-tau instead passes saturating evidence psi(D) = D / (tau + abs(D)) through a smooth choice function, making anti-cue rates for these three high-validity first discriminators nearly identical, so the curvature is very close to 0. Positive-sign and negative-sign CNVW subjects both produce flat or near-flat rates across the three validity levels. The result is a large between-theory mean contrast while the within-subject contrast across 48 focal trials per subject keeps the between-subject variance modest.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_key = data["option_a_ratings"].apply(tuple)
    b_key = data["option_b_ratings"].apply(tuple)
    ones = (1, 1, 1, 1, 1)

    def anti_cue_rate(k):
        b_target = list(ones)
        b_target[k] = 0
        b_target = tuple(b_target)

        a_target = list(ones)
        a_target[k] = 0
        a_target = tuple(a_target)

        mask_b = (a_key == ones) & (b_key == b_target)
        mask_a = (a_key == a_target) & (b_key == ones)

        anti = np.concatenate([
            (data.loc[mask_b, "response"].to_numpy() == 1).astype(float),
            (data.loc[mask_a, "response"].to_numpy() == 0).astype(float),
        ])

        if anti.size == 0:
            return 0.5
        return float(anti.mean())

    r95 = anti_cue_rate(0)
    r92 = anti_cue_rate(1)
    r90 = anti_cue_rate(2)

    return r95 - 2.0 * r92 + r90


### RESULTS
- Predicted under pi_6 (simulated): 0.3900 (var=0.0990)
- Predicted under pi_5 (simulated): -0.0025 (var=0.0475)
- Observed on real data: 0.0675 (var=0.0553)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data):
    import numpy as np
    min_class = 12
    groups = {0: [], 1: []}
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)
        if a[0] > b[0]:
            grp = 0
        elif b[0] > a[0]:
            grp = 1
        else:
            continue
        tally = int(np.sum(a > b) - np.sum(b > a))
        chose_a = 1.0 if int(row['response']) == 0 else 0.0
        groups[grp].append((tally, chose_a))
    parts = []
    for grp in (0, 1):
        vals = groups[grp]
        if not vals:
            parts.append(0.0)
            continue
        d_a = [d for d, y in vals if y == 1.0]
        d_b = [d for d, y in vals if y == 0.0]
        if len(d_a) < min_class or len(d_b) < min_class:
            parts.append(0.0)
        else:
            parts.append((np.mean(d_a) - np.mean(d_b)) / 6.0)
    return float(np.mean(parts))
```

**Observed (real) value:** -0.4323 (var=0.0402)
**Predicted under pi_6:** -0.0094 (var=0.0045)
**Predicted under pi_5:** -0.2852 (var=0.0363)

### Experiment 4
**Design**
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    validities = np.array([0.9, 0.8, 0.65, 0.55, 0.5])
    cue_order = np.argsort(-validities, kind='stable')
    a_when_ttb_a = []
    a_when_ttb_b = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner == -1:
            continue
        chose_a = 1 if int(row['response']) == 0 else 0
        if ttb_winner == 0:
            a_when_ttb_a.append(chose_a)
        else:
            a_when_ttb_b.append(chose_a)
    if not a_when_ttb_a or not a_when_ttb_b:
        return 0.0
    return float(np.mean(a_when_ttb_a) - np.mean(a_when_ttb_b))
```

**Observed (real) value:** -0.2650 (var=0.0713)
**Predicted under pi_6:** -0.2996 (var=0.0120)
**Predicted under pi_5:** -0.2133 (var=0.0171)

### Experiment 5
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    import pandas as pd

    df = data.copy()
    df["pair_key"] = df.apply(
        lambda r: (tuple(r["option_a_ratings"]), tuple(r["option_b_ratings"])),
        axis=1,
    )

    nwa_negative_pairs = [
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    ]
    nwa_positive_pairs = [
        ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 0), (1, 1, 1, 0, 0, 0)),
    ]

    def p_a_for_key(d, key):
        mask = d["pair_key"] == key
        return 1.0 - float(d.loc[mask, "response"].mean())

    def subject_score(d):
        neg = [p_a_for_key(d, k) for k in nwa_negative_pairs]
        pos = [p_a_for_key(d, k) for k in nwa_positive_pairs]
        return sum([pos[i] - neg[i] for i in range(3)]) / 3.0

    scores = df.groupby("subject_id", sort=False).apply(subject_score)
    return float(scores.mean())

```

**Observed (real) value:** 0.3622 (var=0.0979)
**Predicted under pi_6:** 0.4500 (var=0.0174)
**Predicted under pi_5:** 0.5467 (var=0.1673)

### Experiment 6
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    diffs = [tuple(int(x) - int(y) for x, y in zip(a, b)) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    high_keys = {(-1, -1, 1, 1), (-1, 1, -1, 1)}
    low_keys = {(1, 1, -1, -1), (1, -1, 1, -1)}
    high_mask = np.array([d in high_keys for d in diffs], dtype=bool)
    low_mask = np.array([d in low_keys for d in diffs], dtype=bool)
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0
    return float(choose_a[high_mask].mean() - choose_a[low_mask].mean())
```

**Observed (real) value:** 0.5300 (var=0.0883)
**Predicted under pi_6:** 0.5700 (var=0.0173)
**Predicted under pi_5:** 0.5175 (var=0.0843)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    pair_specs = {
        'cp': ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        'ap': ((1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 0)),
        'cn': ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
        'an': ((0, 1, 0, 0, 0, 0), (1, 0, 1, 1, 1, 1)),
    }
    lookup = {v: k for k, v in pair_specs.items()}

    df = data.copy()
    df['_pair'] = [
        lookup.get((tuple(a), tuple(b)), 'other')
        for a, b in zip(df['option_a_ratings'], df['option_b_ratings'])
    ]
    df['_chose_a'] = (df['response'] == 0).astype(float)

    subject_effects = []
    for _, sub in df.groupby('subject_id'):
        means = sub.groupby('_pair')['_chose_a'].mean()
        if not {'cp', 'ap', 'cn', 'an'}.issubset(means.index):
            continue
        cp = means['cp']
        ap = means['ap']
        cn = means['cn']
        an = means['an']
        effect = (
            (abs(ap - 0.5) - abs(cp - 0.5))
            + (abs(an - 0.5) - abs(cn - 0.5))
        )
        subject_effects.append(effect)

    if not subject_effects:
        return float('nan')
    effects = np.asarray(subject_effects, dtype=float)
    if np.any(np.isnan(effects)):
        return float('nan')
    return float(effects.mean())
```

**Observed (real) value:** -0.0300 (var=0.0191)
**Predicted under pi_6:** 0.0075 (var=0.0255)
**Predicted under pi_5:** 0.0050 (var=0.0269)

### Experiment 8
**Design**
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Metric**
```python
import numpy as np

def metric(data):
    v = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5], dtype=float)

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    z = (a - b).dot(v ** 2)

    resp = np.asarray(data['response'], dtype=float)

    low_mask = z < 0.5
    high_mask = z > 0.9

    lows = resp[low_mask]
    highs = resp[high_mask]

    if lows.size == 0 or highs.size == 0:
        return float('nan')

    p_a_low = float(np.mean(lows == 0))
    p_a_high = float(np.mean(highs == 0))

    return float(p_a_low - p_a_high)

```

**Observed (real) value:** 0.0117 (var=0.0033)
**Predicted under pi_6:** -0.0083 (var=0.0131)
**Predicted under pi_5:** 0.0663 (var=0.0100)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.92, 0.84, 0.74, 0.62, 0.52], dtype=float)
    weights = validities ** 2.25

    a_rows = [np.asarray(r, dtype=float) for r in data['option_a_ratings']]
    b_rows = [np.asarray(r, dtype=float) for r in data['option_b_ratings']]
    if len(a_rows) == 0:
        return 0.0

    a = np.vstack(a_rows)
    b = np.vstack(b_rows)
    d = (a - b) @ weights

    resp = data['response'].to_numpy(dtype=float)
    sign_consistent = np.zeros(len(data), dtype=float)
    sign_consistent[(d > 0.0) & (resp == 1.0)] = 1.0
    sign_consistent[(d < 0.0) & (resp == 0.0)] = 1.0

    x = np.abs(d)
    xc = x - x.mean()
    denom = np.sum(xc * xc)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(xc * sign_consistent) / denom)

```

**Observed (real) value:** 0.0064 (var=0.0161)
**Predicted under pi_6:** -0.0130 (var=0.0249)
**Predicted under pi_5:** 0.0393 (var=0.0171)

### Experiment 10
**Design**
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    validities = np.array([0.95, 0.9, 0.8, 0.6, 0.5], dtype=float)
    weights = validities ** 2.25

    subject_contrasts = []
    for _, subj in data.groupby('subject_id'):
        cons = []
        high_mask = []
        low_mask = []

        for _, row in subj.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            d = float(np.dot(weights, a - b))
            ad = abs(d)

            # Negative-validity prediction: choose B when A has larger
            # validity-weighted advantage, otherwise choose A.
            pred_neg = 1 if d > 0 else 0
            cons.append(1.0 if int(row['response']) == pred_neg else 0.0)
            high_mask.append(ad >= 0.90)
            low_mask.append(ad <= 0.25)

        cons = np.array(cons)
        high_mask = np.array(high_mask, dtype=bool)
        low_mask = np.array(low_mask, dtype=bool)

        if high_mask.sum() == 0 or low_mask.sum() == 0:
            subject_contrasts.append(0.0)
            continue

        raw_contrast = float(cons[high_mask].mean() - cons[low_mask].mean())
        # Absolute value aligns negative-sign and positive-sign subjects.
        subject_contrasts.append(abs(raw_contrast))

    if not subject_contrasts:
        return 0.0
    return float(np.mean(subject_contrasts))
```

**Observed (real) value:** 0.1200 (var=0.0096)
**Predicted under pi_6:** 0.0919 (var=0.0049)
**Predicted under pi_5:** 0.1742 (var=0.0110)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the ten experiments, pi_5 is the stronger theory overall: summed absolute point-estimate error is about 0.79 for pi_5 versus 1.60 for pi_6, and pi_5 also matches the observed between-subject variance much better in most experiments (e.g., Experiment 1 var 0.061 observed vs 0.108 pi_5 vs 0.007 pi_6; Experiment 2 var 0.055 vs 0.048 vs 0.099; Experiment 6 var 0.088 vs 0.084 vs 0.017). pi_6 fails structurally on Experiments 1-3: it predicts near-zero D-sign/tally effects (0.017, -0.009) while the observed values are large (0.601, -0.432). This is not a parameter-tuning problem: pi_6's own theory description forbids later-cue influence when the first discriminator has validity 0.90 or higher and explicitly rules out graded validity-weighted accumulation, so no retuning of the current model can produce the observed sign reversals without abandoning the theory. pi_5 captures those sign effects but is not fully task-invariant either: it underestimates the strong anti-validity sign effects in Experiments 1 and 3, yet overpredicts graded D-magnitude effects in Experiments 5, 9, and 10. The observed pattern is most consistent with a thresholded anti-validity sign process: large D-sign effects, near-zero dependence on |D| once the sign is clear, and only a modest high-low |D| contrast. pi_6 is therefore degenerate and should be replaced by a new theory rather than retuned.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory to replace pi_6: a thresholded anti-validity sign rule. Subjects compute a validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j), with subject-level gamma_i around 1.5-3.0. Most subjects use negative weighting: they choose the option with the lower D when |D| exceeds a subject-level threshold tau_i; within the dead zone they fall back on a subject-level bias. A minority use positive weighting. Add a subject-level lapse. To capture the modest high-low |D| contrast in Experiment 10, allow a small soft/graded component (e.g., a low-weight sigmoid over D) or threshold noise, but the dominant process should be thresholded sign, not continuous magnitude. This differs from pi_6 by using the validity-weighted difference rather than a first-discriminating-cue search, and differs from pi_5 by replacing its strongly graded saturating transform and sigmoid with a mostly thresholded sign rule. Suggested parameter ranges: gamma ~ 1.5-3.0, tau ~ 0.05-0.40 in D units, p_negative ~ 0.70-0.90, lapse ~ 0.05-0.25, bias ~ 0.30-0.70, and soft-component weight ~ 0.00-0.20. This should preserve the large D-sign effects seen in Experiments 1 and 3 while reducing the excessive D-magnitude dependence seen in Experiments 5, 9, and 10."
}
```

## Usage

```json
{
  "input_tokens": 13462,
  "output_tokens": 10663,
  "reasoning_tokens": 9354,
  "provider": "CoreWeave"
}
```
