# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7_1" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7_1
Thresholded anti-validity utility with Gaussian threshold noise and a small validity-consistent soft leak (TAVU-tn). Subjects compute a validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j). A subject-level latent sign dominates: most subjects use negative weighting and make the anti-validity choice when |D| exceeds a subject-level threshold tau_i, but threshold placement is noisy across trials, so the aggregate threshold boundary is graded rather than a step. Within the dead zone and below the sign-following ceiling, subjects fall back on a subject-level bias. A small plain-D sigmoidal leak in the validity-consistent direction weakly damps extreme anti choices at large |D|. A uniform lapse adds residual noise. The dominant process remains thresholded sign, not continuous magnitude accumulation.

`predict(parameters, state, history) -> np.ndarray`:
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

## EXPERIMENT 1 (proposed by pi_7_1)

### DESIGN
**Validities (n_features=6):** [0.96, 0.94, 0.9, 0.8, 0.6, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** At gamma = 2.0 the six validity weights v^gamma are [0.9216, 0.8836, 0.81, 0.64, 0.36, 0.25]. The first eight rows use positive D = sum_j v_j^gamma * (A_j - B_j) values of 0.038, 0.0736, 0.1116, 0.17, 0.20, 0.2436, 0.2816, and 0.3116; the last eight rows are exact A/B mirrors. The staircase is built to bracket the TAVU threshold tau in [0.16, 0.19]: the first three D values lie in the TAVU dead zone, 0.17 sits at threshold, 0.20 just above it, and the remaining values are clearly above threshold. For the negative-weight majority, TAVU-tn predicts p(A) near 0.50 for the first three positive pairs, a sharp drop to about 0.26 at D = 0.20, and then a plateau near 0.20 to 0.21; the mirrored side gaps are about 0.00, 0.00, 0.01, 0.26, 0.47, 0.59, 0.59, and 0.59. CNVW-tau has no dead zone and predicts a smooth continuous decline in p(A) from about 0.42 at D = 0.038 to about 0.24 at D = 0.31, with mirrored side gaps growing monotonically from about 0.16 to 0.53. The critical dissociation is dead-zone flatness below threshold plus a threshold jump and plateau versus a smooth saturation curve, with the two side-gap functions crossing near D = 0.20. This quantitative interaction cannot be removed by beta, lambda, bias, or lapse within the stated ranges.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
For each unordered stimulus pair, the metric computes a side gap: the absolute difference in mean A-choice rate between the two mirror orientations. This difference cancels subject-level bias and lapse because those add equally to both orientations, leaving the evidence-dependent choice contrast. Using the fixed validities with the designed D values, low-evidence pairs are selected with D_ref < 0.15 and high-evidence pairs with D_ref > 0.22. TAVU-tn predicts a dead zone where low-D side gaps are near zero and high-D side gaps jump to a high plateau, so high_gap - 2*low_gap is strongly positive. CNVW-tau has no dead zone: low-D side gaps are already sizable, and its concave smooth side-gap curve makes twice the low gap roughly cancel the high gap, producing values near zero or negative. Averaging per subject keeps the metric aligned with between-subject variability and avoids sign-mixture cancellation.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.96, 0.94, 0.9, 0.8, 0.6, 0.6], dtype=float)

    def subject_contrast(group: pd.DataFrame) -> float:
        rows = []
        for _, row in group.iterrows():
            a = tuple(int(x) for x in row['option_a_ratings'])
            b = tuple(int(x) for x in row['option_b_ratings'])
            a_arr = np.array(a, dtype=float)
            b_arr = np.array(b, dtype=float)
            d_ref = float(np.dot(validities ** 2.0, a_arr - b_arr))
            pair = tuple(sorted((a, b)))
            orient = 0 if a < b else 1
            rows.append((pair, orient, abs(d_ref), 1.0 - float(row['response'])))

        recs = pd.DataFrame(rows, columns=['pair', 'orient', 'd_ref_abs', 'p_a'])
        means = recs.groupby(['pair', 'orient'])['p_a'].mean().reset_index()

        def gap_fn(s):
            vals = list(s)
            if len(vals) < 2:
                return float('nan')
            return abs(vals[0] - vals[1])

        gaps = means.groupby('pair')['p_a'].apply(gap_fn)
        d_ref = recs.groupby('pair')['d_ref_abs'].first()

        low = gaps[d_ref < 0.15]
        high = gaps[d_ref > 0.22]
        if low.empty or high.empty:
            return float('nan')
        return float(high.mean() - 2.0 * low.mean())

    per_subject = data.groupby('subject_id').apply(subject_contrast)
    return float(per_subject.mean())


### RESULTS
- Predicted under pi_7_1 (simulated): 0.1783 (var=0.0467)
- Predicted under pi_5 (simulated): -0.1583 (var=0.0665)
- Observed on real data: -0.5133 (var=0.1804)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=9):** [0.95, 0.87, 0.75, 0.68, 0.65, 0.62, 0.61, 0.6, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** This design isolates the shape of the D-to-choice mapping by using a one-hot validity staircase anchored on the 0.50 feature. With validities [0.95, 0.87, 0.75, 0.68, 0.65, 0.62, 0.57, 0.51, 0.50] and a central gamma of 2.0, the squared validity weights are approximately [0.9025, 0.7569, 0.5625, 0.4624, 0.4225, 0.3844, 0.3249, 0.2601, 0.2500]. Each positive-D trial lets A win one higher-validity feature and B win the 0.50 anchor, so D equals the simple difference between that feature's weight and 0.25. The positive D staircase is therefore 0.0101, 0.0749, 0.1344, 0.1725, 0.2124, 0.3125, 0.5069, 0.6525. The critical property is that TAVU-tn has a threshold and dead zone while CNVW-tau is continuous and compressive. At the three smallest positive D values, TAVU-tn remains inside its threshold dead zone because those values are roughly 2 to 3 threshold-noise units below tau, so it predicts p(A) close to the fallback level, about 0.50. CNVW-tau instead applies psi(D) = D / (tau + abs(D)) with tau around 0.175, so it predicts a graded anti-validity decline from about p(A) = 0.48 at D = 0.0101 down to about 0.33 at D = 0.1344. This pre-threshold gradient is a shape signature TAVU-tn cannot produce with tau_sd in [0.02, 0.04]. The 0.1725 pair sits at the TAVU threshold, producing a steep threshold-like drop for TAVU-tn, whereas CNVW-tau continues smoothly. The larger D values add a second signature: after crossing threshold, TAVU-tn plateaus near p(A) around 0.26 because follow_max and the soft leak bound its anti-validity ceiling, while CNVW-tau keeps compressing and reaches about p(A) = 0.22 at D = 0.6525. Under central parameters (CNVW-tau: gamma=2.0, tau=0.175, lambda=1.4, beta=1.4, bias=0, epsilon=0.05, p_positive=0.05; TAVU-tn: gamma=2.0, tau=0.17, tau_sd=0.03, p_negative=0.91, bias=0.5, lapse=0.14, soft_weight=0.02, soft_beta=0.9, follow_max=0.70), the positive-block p(A) values are approximately CNVW-tau [0.477, 0.378, 0.328, 0.307, 0.290, 0.262, 0.234, 0.223] versus TAVU-tn [0.500, 0.500, 0.472, 0.372, 0.275, 0.259, 0.258, 0.259]. This gives a rank-order crossover near D = 0.21 and a second ordering reversal at high D, so the design produces both a dead-zone-flatness dissociation and a plateau-versus-compression dissociation in one staircase. The eight negative-D mirror trials double the data and provide a symmetry check. There are 16 unique pairs, repeated 6 times each for the 96-trial budget.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject dead-zone contrast. For each subject I compute D = sum_j validities_j^2 * (A_j - B_j), then select low-D trials with 0.015 < abs(D) < 0.15. The scalar is mean p(A) on negative-D low trials minus mean p(A) on positive-D low trials. CNVW-tau compresses even small D, so this contrast moves away from zero: negative-sign subjects choose A more on negative-D low trials than on positive-D low trials. TAVU-tn leaves these small D values inside its threshold dead zone, so the same contrast stays near zero. High-D trials with abs(D) > 0.17 are used only to infer each subject's latent validity sign; subjects whose high-D pattern implies positive validity weighting have the low-D contrast sign-flipped. This aligns the small positive-sign mixture and reduces between-subject variance. The final metric is the mean of these aligned per-subject contrasts.

Source:
import numpy as np
import pandas as pd

_V = np.array([0.95, 0.87, 0.75, 0.68, 0.65, 0.62, 0.61, 0.60, 0.60], dtype=float)
_GAMMA = 2.0
_LOW_LO = 0.015
_LOW_HI = 0.15
_HIGH_CUT = 0.17

def metric(data):
    if data.empty:
        return 0.0

    values = []
    for _, sub in data.groupby('subject_id', sort=False):
        a = np.array(sub['option_a_ratings'].tolist(), dtype=float)
        b = np.array(sub['option_b_ratings'].tolist(), dtype=float)
        d = (a - b).dot(_V ** _GAMMA)

        pa = (sub['response'].to_numpy(dtype=float) == 0.0).astype(float)

        pos_low = (d > _LOW_LO) & (d < _LOW_HI)
        neg_low = (d < -_LOW_LO) & (d > -_LOW_HI)
        if pos_low.sum() == 0 or neg_low.sum() == 0:
            continue

        contrast = float(pa[neg_low].mean() - pa[pos_low].mean())

        high_pos = pa[d > _HIGH_CUT]
        high_neg = pa[d < -_HIGH_CUT]
        if high_pos.size > 0 and high_neg.size > 0:
            hp = float(high_pos.mean())
            hn = float(high_neg.mean())
            if hp - hn > 0.0:
                contrast = -contrast

        values.append(contrast)

    if not values:
        return 0.0
    return float(np.mean(values))

### RESULTS
- Predicted under pi_7_1 (simulated): 0.0022 (var=0.0196)
- Predicted under pi_5 (simulated): 0.2444 (var=0.0354)
- Observed on real data: 0.3889 (var=0.0632)

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
**Predicted under pi_7_1:** -0.2989 (var=0.0305)
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
**Predicted under pi_7_1:** -0.1992 (var=0.0231)
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
**Predicted under pi_7_1:** 0.5539 (var=0.0612)
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
**Predicted under pi_7_1:** 0.4750 (var=0.1189)
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
**Predicted under pi_7_1:** 0.0233 (var=0.0179)
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
**Predicted under pi_7_1:** 0.0592 (var=0.0113)
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
**Predicted under pi_7_1:** 0.0031 (var=0.0170)
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
**Predicted under pi_7_1:** 0.2039 (var=0.0121)
**Predicted under pi_5:** 0.1742 (var=0.0110)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.6008 (var=0.0611)
**Predicted under pi_7_1:** 0.4646 (var=0.0686)
**Predicted under pi_5:** 0.4529 (var=0.1083)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.0675 (var=0.0553)
**Predicted under pi_7_1:** 0.0112 (var=0.0581)
**Predicted under pi_5:** -0.0025 (var=0.0475)

### Experiment 13
**Design**
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    pos_pairs = {
        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 1)),
        ((1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 1)),
        ((1, 0, 1, 0, 0, 0), (0, 1, 0, 0, 0, 1)),
        ((1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 1)),
        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
    }
    neg_pairs = {
        ((1, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0)),
        ((0, 1, 0, 0, 0, 1), (1, 0, 0, 0, 1, 0)),
        ((0, 1, 0, 0, 0, 1), (1, 0, 1, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 1), (1, 1, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),
    }

    def pair_key(r):
        return (tuple(r['option_a_ratings']), tuple(r['option_b_ratings']))

    df = data.copy()
    df['_pair'] = df.apply(pair_key, axis=1)

    def subject_score(g):
        pos = g.loc[g['_pair'].isin(pos_pairs), 'response'].eq(0).mean()
        neg = g.loc[g['_pair'].isin(neg_pairs), 'response'].eq(0).mean()
        if pd.isna(pos) or pd.isna(neg):
            return np.nan
        return float(abs(neg - pos))

    scores = df.groupby('subject_id', sort=False).apply(subject_score).dropna()
    if scores.empty:
        return 0.0
    return float(scores.mean())

```

**Observed (real) value:** 0.3030 (var=0.0429)
**Predicted under pi_7_1:** 0.4210 (var=0.0090)
**Predicted under pi_5:** 0.4985 (var=0.0248)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    v = np.array([0.92, 0.92, 0.65, 0.65, 0.7, 0.6, 0.5, 0.5], dtype=float)
    df = data.copy()

    def _d(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        return float(np.sum((v ** 2.0) * (a - b)))

    df['__d'] = df.apply(_d, axis=1)
    df['__pa'] = 1.0 - df['response'].astype(float)

    out = []
    for _, sub in df.groupby('subject_id', sort=False):
        orient = sub[sub['__d'].abs() >= 0.16]
        small = sub[(sub['__d'].abs() > 0.03) & (sub['__d'].abs() < 0.16)]
        if len(orient) == 0 or len(small) == 0:
            continue

        p_pos_orient = orient.loc[orient['__d'] > 0, '__pa'].mean()
        p_neg_orient = orient.loc[orient['__d'] < 0, '__pa'].mean()
        if pd.isna(p_pos_orient) or pd.isna(p_neg_orient):
            continue

        p_pos_small = small.loc[small['__d'] > 0, '__pa'].mean()
        p_neg_small = small.loc[small['__d'] < 0, '__pa'].mean()
        if pd.isna(p_pos_small) or pd.isna(p_neg_small):
            continue

        if p_neg_orient >= p_pos_orient:
            step = p_neg_small - p_pos_small
        else:
            step = p_pos_small - p_neg_small

        out.append(float(np.clip(step, 0.0, 1.0)))

    if not out:
        return float('nan')
    return float(np.mean(out))
```

**Observed (real) value:** 0.0356 (var=0.0037)
**Predicted under pi_7_1:** 0.0889 (var=0.0111)
**Predicted under pi_5:** 0.3367 (var=0.0255)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The two targeted experiments were explicitly designed to separate a dead-zone threshold (THEORY 1) from continuous compressive anti-validity weighting (THEORY 2), and the observed data clearly reject the dead-zone account. In Experiment 2 the observed aligned low-D contrast is 0.3889: THEORY 1 predicts 0.0022 because it treats the small D values as uninformative dead-zone trials, while THEORY 2 predicts 0.2444, correct in sign and shape but roughly 0.14 too low. In Experiment 1 the observed high-minus-2-low gap is -0.5133: THEORY 1 predicts +0.1783, the wrong sign, whereas THEORY 2 predicts -0.1583, the correct sign but only about one-third of the observed magnitude. Across the other experiments the two theories split the comparisons (THEORY 2 is closer on Experiments 4, 5, 6, 7, 10 and the two new experiments; THEORY 1 is closer on Experiments 3, 8, 9, 11, 12, 13, 14), but THEORY 1 fails the two most diagnostic dissociations. THEORY 2 is the better theory overall, yet it is not task-invariant: its fixed compressor severely overpredicts the low-D contrast in Experiment 14 (0.3367 versus observed 0.0356) and underpredicts the critical contrasts in Experiments 1 and 2. The data therefore require a graded anti-validity effect whose small-D sensitivity depends on the structure of the stimulus conflict, not a single universal threshold or a single universal half-saturation constant. Since both theories are only partially adequate, a new theory is warranted.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace THEORY 1 with a conflict-modulated continuous anti-validity theory. Keep a mostly negative subject-level sign and a continuous validity-weighted difference D = sum_j v_j ** gamma_i * (A_j - B_j), but do not impose a fixed threshold and do not use one fixed half-saturation constant. Use an evidence-gated response such as S_i = -lambda_i * G(C) * D / (kappa_i + abs(D) + r_i) + bias_i + w_t * T, where G(C) is a subject-level gain that is large when D is concentrated in one or a few high-validity experts and small when D comes from many low-validity or mutually conflicting experts. C can be an inverse-validity weighted count of disagreeing features or the largest single feature contribution divided by abs(D). A small tally or configural term T = (1/n) * sum_j (A_j - B_j) with a sign-dependent weight should also be included to handle count-driven designs. This theory differs from THEORY 1 by having no dead zone, and differs from THEORY 2 by making gain and saturation depend on conflict structure rather than a fixed tau. The resulting shape can give a strong low-D gradient in the one-hot staircase of Experiment 2, a very negative high-minus-2-low-gap statistic in Experiment 1, and near-zero low-D contrast in the jumbled conflict design of Experiment 14."
}
```

## Usage

```json
{
  "input_tokens": 18480,
  "output_tokens": 8002,
  "reasoning_tokens": 7306,
  "provider": "Baidu"
}
```
