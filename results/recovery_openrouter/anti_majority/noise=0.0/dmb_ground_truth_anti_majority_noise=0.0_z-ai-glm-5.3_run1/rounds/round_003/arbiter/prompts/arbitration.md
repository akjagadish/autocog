# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Skeptical Defect Integration (SDI). Subjects do not treat an expert's positive rating ('1') as an asset; they treat it as a liability — an endorsement from a fallible expert is noise-laden, adverse evidence. On each trial the decision maker computes a penalty for each option, P(X) = sum_j d_j * x_j, where d_j = (1 - v_j)^delta is the skepticism weight attached to expert j (delta = 0 recovers pure unweighted defect counting — every endorsement is one unit of suspicion; larger delta discounts endorsements from low-validity experts more steeply, since a bad expert's praise is especially untrustworthy). The subject chooses the option with the LOWER total penalty, i.e., the option that attracted less endorsement. Choice probability is P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. Exact penalty ties are broken by a weak validity-weighted endorsement term alpha * sum_j logit(v_j) * (a_j - b_j), which captures residual heterogeneity: as alpha grows, SDI smoothly recovers VWEI-like endorsement-following behavior. The empirically human region sits at delta near 0 (near-pure defect counting) with moderate noise. SDI is the only account in this domain that reproduces the SIGN of every observed metric: it follows the TTB winner whenever the tally winner carries a larger endorsing coalition (Exp 1 metric positive), follows the tally loser / top-cue favorite when the tally winner has more endorsements (Exp 2 metric negative), prefers the option with fewer endorsements even when a dominant high-validity cue endorses the rival (Exp 3 metric strongly negative, HIGH minus LOW), and produces a negative signed-evidence slope because P(choose TTB winner) DECREASES as validity-weighted evidence accumulates behind the more-endorsed option (Exp 4). It makes a novel falsifiable prediction: on designs where the high-validity option carries MORE positive ratings than the low-validity option, SDI predicts majority choices OPPOSITE to both TTB and VWEI.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Skeptical Defect Integration (SDI).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Penalty for option X:  P(X) = sum_j d_j * x_j,
    #   d_j = (1 - v_j)^delta  (skepticism weight; delta = 0 -> pure
    #   unweighted defect counting).
    # Decision variable:      D = (P(B) - P(A)) + alpha * E,
    #   E = sum_j logit(v_j) * (a_j - b_j)  (weak validity-weighted
    #   endorsement tie-break).
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so skepticism weights are fixed for the block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SDI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard against v = 1 (infinite logit) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)

    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])

    # Skepticism weights: an endorsement from expert j is a liability,
    # scaled by how fallible the expert is.
    d = (1.0 - v) ** delta

    # Total penalty (accumulated suspicion) for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Weak validity-weighted endorsement term (tie-break only).
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A (A carries fewer / more-heavily-discounted
    # liabilities than B, plus a small endorsement edge).
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties and endorsement) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_5
Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=15):** [0.95, 0.9, 0.9, 0.9, 0.85, 0.8, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 19: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 22: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-5): the first head-to-head between the two DEFECT-INTEGRATION theories. SDI and VPS agree that every endorsement is a liability, but they disagree about the WEIGHT PROFILE attached to the endorser's validity: SDI's skepticism weight (1-v)^delta is nearly FLAT across the whole validity range (0.86-0.98 at delta <= 0.05, a <14% spread; at delta=0 exactly flat), so SDI is essentially a validity-blind endorsement COUNTER; VPS's weight c0 + c1*v^kappa INCREASES steeply in validity (a 0.95 expert's endorsement costs ~2.9x a 0.50 expert's at mid parameters), so VPS actively prefers options endorsed by WEAKER experts. The design places the two theories in orthogonal indifference regimes: SDI is pinned at chance exactly when endorsement counts match but validity composition differs; VPS is pinned at (near) chance exactly when counts differ but its validity-scaled penalties cancel. Validities (15 experts): [0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65, 0.60, 0.60, 0.60, 0.55, 0.50, 0.50, 0.50, 0.50] -- duplicates are required to build coalitions; the spread 0.50-0.95 spans the full range. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below are given as P(A) at SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025) and VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1), with full-box ranges verified sign-robust. FAMILY 1 -- VALIDITY-GAP LADDER AT MATCHED COUNTS (pairs 1-5): 1v1 pairs with shrinking endorser gaps (.95/.50, .90/.55, .85/.60, .80/.65) plus the doubled 2v2 max-gap pair (.95+.90 vs .55+.50). VPS predicts a monotone ladder BELOW 50%: 0.24, 0.29, 0.34, 0.40, 0.13 -- it always prefers the LOWER-validity endorser, structurally (its penalty is strictly increasing in v for any c1>0, so D<0 at matched counts for EVERY parameterization). SDI is pinned at 0.51 +/- 0.02 (box 0.50-0.56): its penalty weights vary <14% and its only tilt, the alpha<=0.05 tie-break, pushes TOWARD the high-validity endorser -- i.e., AWAY from VPS's prediction, so no SDI parameterization can mimic the ladder. A monotone descending P(A) in gap falsifies SDI; a flat ~0.51 line falsifies VPS. FAMILY 2 -- SIGN CONFLICTS, OPPOSITE MAJORITIES (pairs 6-8): (a) 3x0.50 vs 2x0.90: VPS 0.79 (box 0.60-0.90) vs SDI 0.26 (box 0.24-0.30); (b) 4x0.50 vs 3x0.90: VPS 0.89 (box 0.75-0.93) vs SDI 0.25 (box 0.26-0.31); (c) 2x0.90 vs 3x0.60: VPS 0.31 (box 0.17-0.45) vs SDI 0.72 (box 0.68-0.75). Every conflict is sign-robust across BOTH entire parameter boxes (e.g., pair 6: VPS D = -c0 + c1(2*0.9^kappa - 3*0.5^kappa) > 0 always since c0/c1 <= 0.214 < 0.700; pair 8: D < 0 always since c0/c1 <= 0.214 < 0.365). A single majority choice on any conflict cell adjudicates the theories with no free parameters able to reverse it. FAMILY 3 -- JUNK-COUNT GRADIENT VS FIXED 2x0.90 (pairs 9, 10, 6): k = 1, 2, 3 endorsements from 50% experts on A against the same two 90% endorsements on B. SDI declines 0.66 -> 0.45 -> 0.26, CROSSING 50% between k=1 and k=2 (each junk endorsement adds ~0.97 penalty units; crossing at count parity is structural). VPS declines only 0.91 -> 0.86 -> 0.79 and does NOT cross through k=3, because each 0.50 endorsement costs only ~0.35 penalty units vs ~0.91 for a 0.90 endorsement -- it takes ~2.5 junk endorsements to offset one 90% endorsement for VPS but only 1 for SDI. The LOCATION OF THE CROSSING POINT is the quantitative signature. FAMILY 4 -- INDIFFERENCE ASYMMETRY (pairs 11-12): pair 11 (one 0.90 vs two 0.60s) is engineered so 2*0.6^kappa ~= 0.9^kappa (within 0.04-0.19 across kappa in [1.6, 2.4]), with the c0 count term partially cancelling the residual: VPS's D ranges only [-0.20, +0.20] over its ENTIRE box, keeping P(A) in [0.40, 0.61] (mid 0.50) -- near chance -- while SDI delivers a clear majority, P(A) = 0.70 (box 0.65-0.71), because B carries one extra endorsement. Pair 12 (3x0.90 vs 2x0.50) is the floor test: both theories prefer B, but SDI only moderately (P(A) = 0.35, box 0.28-0.42) while VPS is pinned at the floor (P(A) = 0.07, box 0.04-0.10) -- a 25+-point magnitude gap where sign cannot adjudicate. THE DISSOCIATION IS UN-FAKEABLE IN BOTH DIRECTIONS: a VPS subject is a descending ladder in Family 1 (all rungs < 0.47), stays above 0.60 through k=3 in Family 3, and sits at chance on pair 11; an SDI subject is a flat ~0.51 line across the entire ladder (its alpha tie-break can only push it ABOVE 0.5, opposite to VPS), crosses 50% at k=2 in Family 3, and shows a 0.70 majority on pair 11. Because SDI's stimulus-driven variation at matched counts is bounded by alpha*delta-logit <= 0.25 penalty units while VPS's ladder spans D in [-1.6, -0.41], and because VPS's indifference on pair 11 is arithmetic (2*0.6^kappa ~= 0.9^kappa across its kappa box) while SDI's count signal there is a full penalty unit, neither theory's parameter box can approximate the other's profile. All pairs are non-degenerate (every pair has discriminating endorsements, no all-tie pairs, no pair where both theories guess -- the closest, pair 11, is precisely the cell where they maximally diverge), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design's experts split cleanly into a high-validity group (features 0-6, validities 0.65-0.95) and a low-validity group (features 7-14, validity 0.60), and every trial pits a pure-high option against a pure-low option. This yields two orthogonal conflict families. FAMILY 1 (matched endorsement counts, 12 unique pairs, 48 trials/subject: e.g. one 0.95 endorser vs one 0.60 endorser; 2x0.90 vs 2x0.60): SDI's skepticism weight (1-v)^delta is nearly flat over the validity range (delta in [0, 0.05]), so its penalty difference is ~0 and its weak alpha tie-break pushes it slightly TOWARD the high-validity option (P ~ 0.51-0.55). VPS's penalty c0 + c1*v^kappa rises steeply in validity, so it systematically prefers the LOW-validity option (P(high) ~ 0.28 at mid parameters, and < 0.47 across its entire parameter box since D = c1*(v_lo^kappa - v_hi^kappa) < 0 structurally). FAMILY 2 (count conflicts where the FEWER-endorsed option is the high-validity one, 8 unique pairs, 32 trials/subject: 3x0.60 vs 2x0.90; 4x0.60 vs 3x0.90; 2x0.90 vs 3x0.60; 1x0.90 vs 2x0.60): SDI follows the endorsement count and picks the high-validity option with P ~ 0.70-0.73; VPS finds the low-validity majority cheaper per endorsement and picks the high-validity option only ~0.14-0.50 of the time (mean ~0.30). Cells where the fewer-endorsed option is the LOW-validity one (trials 9, 12, 21, 24) are agreed upon by both theories and are deliberately excluded so they cannot dilute the contrast. The metric averages P(choose the high-validity-group option) over the two families: SDI predicts ~0.62, VPS predicts ~0.29 — a gap of ~0.33 with per-subject standard deviation of only ~0.05 (binomial noise over 48 + 32 trials), giving a Welch t-statistic around 20 at N=25 (p << 0.001). The contrast is sign-robust across both theories' full parameter boxes: even at VPS's most SDI-like corner (c0=0.15, c1=0.7, kappa=1.6, beta=1.6, eps=0.13) the composite only rises to ~0.40, while SDI's least favorable corner stays ~0.59. The metric uses only the high/low validity grouping (threshold 0.65), which is invariant to the exact validities assigned to features 7-14, and it is well-defined both on the pooled DataFrame and on any single subject's 96-trial slice.

Source:
def metric(data: pd.DataFrame) -> float:
    # "Validity-following rate in theory-conflict cells"
    #
    # The 15 experts split into a HIGH-validity group (features 0-6:
    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a
    # LOW-validity group (features 7-14: validity 0.60). Every trial in
    # this design pits a "pure high" option against a "pure low"
    # option. Two families of cells separate SDI from VPS:
    #
    # Family 1 (matched endorsement counts, validity gap): SDI's
    #   penalty is a near-flat function of validity (delta ~ 0), so it
    #   is pinned at ~0.51 for the high-validity option (its weak
    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's
    #   penalty rises steeply in validity, so it systematically picks
    #   the LOW-validity option (~0.28).
    #
    # Family 2 (count conflict where the FEWER-endorsed option is the
    #   high-validity one): SDI follows the endorsement count and
    #   picks the high-validity option (~0.72). VPS's validity-scaled
    #   distrust makes the low-validity majority cheaper, so it picks
    #   the high-validity option only ~0.30 of the time.
    #
    # The metric is the average of P(choose the high-validity-group
    #   option) over these two families. SDI predicts ~0.62; VPS
    #   predicts ~0.29.

    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,
                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

    f1_hits = 0.0
    f1_n = 0
    f2_hits = 0.0
    f2_n = 0

    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        m = int(min(a.size, b.size, V.size))
        if m == 0:
            continue
        a = a[:m]
        b = b[:m]

        top = V[:m] >= 0.65  # high-validity expert group
        a_top = float(np.sum(a[top]))
        a_bot = float(np.sum(a[~top]))
        b_top = float(np.sum(b[top]))
        b_bot = float(np.sum(b[~top]))

        # Require each option to be endorsed purely by one group
        # (all trials in this design satisfy this; guards oddities).
        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):
            continue
        ca = a_top + a_bot
        cb = b_top + b_bot
        if ca == 0 or cb == 0:
            continue

        a_is_top = a_top > 0
        b_is_top = b_top > 0
        if a_is_top == b_is_top:
            continue

        chose_a = (int(round(float(r))) == 0)
        chose_top = chose_a if a_is_top else (not chose_a)

        if ca == cb:
            # Family 1: matched counts, validity gap.
            f1_n += 1
            f2_unused = None
            f1_hits += 1.0 if chose_top else 0.0
        else:
            # Family 2: count conflict; keep only cells where the
            # FEWER-endorsed option is the high-validity one
            # (the cells where SDI and VPS point in opposite
            # directions). Cells where the fewer-endorsed option is
            # the low-validity one are agreed upon by both theories
            # and are excluded.
            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)
            if fewer_is_top:
                f2_n += 1
                f2_hits += 1.0 if chose_top else 0.0

    p1 = f1_hits / f1_n if f1_n > 0 else np.nan
    p2 = f2_hits / f2_n if f2_n > 0 else np.nan
    vals = [p for p in (p1, p2) if not np.isnan(p)]
    if not vals:
        return 0.5
    return float(np.mean(vals))

### RESULTS
- Predicted under pi_4 (simulated): 0.6227 (var=0.0042)
- Predicted under pi_5 (simulated): 0.2844 (var=0.0060)
- Observed on real data: 0.3815 (var=0.0238)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=20):** [0.95, 0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.75, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.65, 0.65, 0.65]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 18: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 19: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Rationale:** NOVEL AXIS (vs. Exps 0-6): MATCHED-COUNT SIGNAL AMPLIFICATION plus the full COUNT-VALIDITY EXCHANGE ECONOMICS. Exp 6 probed VPS vs SDI at unit matched count (1v1 validity ladder) and against one fixed high side (k x 0.50 vs 2x0.90). This design instead manipulates the COUNT dimension itself, exploiting a structural fact neither prior design used: at matched endorsement counts the two theories' decision variables have OPPOSITE SIGNS and IDENTICAL linear scaling in count. VPS: D = pen_B - pen_A = k*c1*(v_hi^kappa - v_lo^kappa) > 0 (the flat c0 term cancels EXACTLY, so matched-count trials are a pure assay of the confidence-distrust term c1*v^kappa), growing linearly in k, favoring the LOW-validity coalition ever more strongly. SDI: D = k*[(w_hi - w_lo) + alpha*Delta-logit] < 0 (its skepticism weight (1-v)^delta DECREASES in v, and the alpha tie-break also favors the high-validity side), also growing linearly in k, favoring the HIGH-validity coalition. Matched count k is therefore a signal AMPLIFIER for a sign conflict: at k=1 the theories differ by ~20 points; at k=4 they are at floor-vs-ceiling in opposite directions. Validity pool (20 experts): [0.95 x3, 0.90 x4, 0.85 x2, 0.75 x1, 0.65 x3, 0.60 x4, 0.55 x3] -- duplicates are required to build matched coalitions. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. All predictions below: P(A) at VPS mid-params (kappa=2, beta=2, eps=.10, c0=.1, c1=1) and SDI mid-params (delta=.025, beta=1, eps=.115, alpha=.025), with full-box ranges [min,max] verified at all box corners; CRITICALLY, the VPS and SDI prediction intervals are DISJOINT on every single pair. FAMILY 1 -- MATCHED-COUNT GAP AMPLIFICATION (pairs 1-6): k x 0.60 vs k x 0.90 at k=1,2,3,4, and k x 0.55 vs k x 0.95 at k=2,3. VPS climbs toward ceiling: 0.69, 0.82, 0.89, 0.93 (gap .90/.60) and 0.87, 0.93 (gap .95/.55); box ranges [0.60,0.80], [0.68,0.92], [0.76,0.96], [0.81,0.96], [0.73,0.96], [0.81,0.96]. SDI stays pinned at/below chance, drifting slightly TOWARD the high-validity side as k grows: 0.48, 0.47, 0.45, 0.43 and 0.45, 0.42; box ranges [0.46,0.50] down to [0.34,0.50]. The signature is a sign x count interaction: VPS's slope in k is positive and large (+24 points over k=1..4 at mid params, and >= +21 points even between worst-case corners); SDI's slope is NEGATIVE or zero (its matched-count D is bounded in [-k*(0.064+2.74*alpha), 0], so P(A) <= 0.50 always and >= 0.34 at k=4). No SDI parameterization can produce P(A) > 0.5 on ANY matched-count pair (its validity sensitivity has the wrong sign and is bounded by 12% + alpha<=0.05), while no VPS parameterization can stay below 0.60 at k=4 (its D >= 4*0.7*0.403 = 1.13 across the entire box). FAMILY 2 -- EXCHANGE-RATE LOCUS (pairs 7-9): 2x0.55 vs 1 x v_hi with v_hi = 0.95, 0.85, 0.75. SDI is a validity-blind endorsement COUNTER here: the single endorser always carries fewer penalties, and its D varies by < 0.1 across the whole v_hi range, so SDI is FLAT at 0.28 / 0.29 / 0.29 (box [0.25,0.31]). VPS's preference tracks its internal exchange rate (v_hi/v_lo)^kappa ~ 2: it PREFERS two 55% endorsements over one 95% endorsement (0.59, box [0.49,0.74] -- a sign conflict with SDI on every parameterization), is indifferent at 0.85 (0.51, box [0.45,0.62]), and leans single at 0.75 (0.44, box [0.42,0.49]). VPS's D is strictly increasing in v_hi (slope sign parameter-independent), producing a monotone 15-point descending curve that crosses chance exactly where the exchange rate predicts; SDI's flat line cannot cross chance at any v_hi. FAMILY 3 -- VPS INDIFFERENCE RIDGE vs SDI COUNT GRADIENT (pairs 10-12): unmatched-count cells engineered so k*(c0 + c1*v_lo^kappa) ~= m*(c0 + c1*v_hi^kappa) at kappa~2 (the box center), pinning VPS at chance by arithmetic while SDI delivers a count-difference-graded majority for the fewer-endorsement option. Pair 10 (3x0.65 vs 2x0.85, count diff 1): VPS 0.54 (box [0.46,0.67]) vs SDI 0.29 (box [0.25,0.30]). Pair 11 (4x0.60 vs 2x0.90, count diff 2, since 2*0.9^2 = 4*0.6^2 - 0.18 with the c0 term closing the gap): VPS 0.49 (box [0.38,0.70]) vs SDI 0.16 (box [0.14,0.17]) -- SDI's majority deepens with count difference while VPS stays at chance, and even at VPS's most extreme box corner (0.38) it exceeds SDI's ceiling (0.17) by 21 points. Pair 12 (3x0.55 vs 2x0.85) is the fixed-count-ratio validity probe: same 3-vs-2 structure as pair 10 but with a larger coalition validity gap, VPS jumps from 0.54 to 0.68 (box [0.54,0.84], sign-robust) while SDI stays at 0.28 (box [0.25,0.31]) -- at fixed count difference, VPS's preference grows with the validity gap and SDI is structurally flat. UN-FAKEABILITY IN BOTH DIRECTIONS: (i) SDI cannot mimic Family 1 because its per-endorsement matched-count signal is bounded by |w_hi - w_lo| <= 0.064 plus alpha*Delta-logit <= 0.14, all of the WRONG SIGN (toward the high-validity side), so its P(A) is <= 0.50 on every matched-count pair while VPS's is >= 0.60 at k>=1 (>= 0.68 at k>=2); (ii) VPS cannot mimic Families 2-3 because on those cells its D is pinned within [-0.36, +0.39] (pair 11) and [-0.25, +0.48] (pair 7) by the arithmetic of k*v_lo^kappa ~= m*v_hi^kappa across its whole kappa box, keeping P(A) in [0.38,0.70] where SDI sits at 0.14-0.31 -- VPS has no parameterization that produces a 0.16 majority for the 2-endorsement option on pair 11, and SDI has none that produces a 0.59 majority for the 2-endorsement option on pair 7. The design also isolates VPS's two components: Family 1 cancels c0 exactly (pure c1*v^kappa assay), while Families 2-3 force c0 (the count term) to trade off against c1*v^kappa, with the ridge cells showing where they cancel. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- the closest cells, pairs 8 and 11, are precisely where the boxes are maximally separated), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design's Family 1 (pairs 1-6 and their mirrors 13-18) is a pure matched-count assay: both options carry exactly k endorsements, so every pure count term (SDI's near-uniform skepticism weights at delta near 0, and VPS's flat c0 term) cancels exactly, leaving only the validity-sensitive component. There the two theories make strictly opposite-sign predictions that are parameter-robust across their entire boxes: VPS prefers the LOWER-validity coalition (mid-parameter P ~ 0.69, 0.82, 0.89, 0.93, 0.87, 0.93 for k = 1..4 ladders; box floors still >= 0.60 and rising with k), while SDI is pinned at or below 0.50 on every matched-count pair (its validity sensitivity has the wrong sign and is bounded by ~0.064 + alpha*Delta-logit, giving ~0.42-0.48 at mid parameters). The metric therefore is simply the proportion of choices of the low-validity coalition over all matched-count trials, identified stimulus-side from the rating vectors and the fixed validity list (order randomization and A/B mirroring are handled automatically because the side is computed per trial from content, not position). Expected values: VPS ~0.85, SDI ~0.45 — a mean gap of ~0.40. Per subject this is a proportion over 48 Bernoulli trials (binomial SD ~0.05-0.07), and even adding parameter heterogeneity across the boxes the per-subject SD stays ~0.08-0.09 on each side, so at N=25 the Welch t-statistic is on the order of 15+, far beyond the alpha=0.01 threshold. The metric works identically on the pooled DataFrame (canonical point estimate) and on each subject's slice (for the between-subject variance), and uses only scalar groupings (row iteration), avoiding any hashing of list-valued cells.

Source:
def metric(data: pd.DataFrame) -> float:
    # Matched-count validity-reversal index.
    #
    # On trials where A and B carry the SAME number of endorsements (k vs k),
    # the two theories make opposite-sign predictions:
    #   VPS: the coalition of LOWER-validity endorsers is preferred
    #        (a high-validity endorsement is a bigger suspected defect),
    #        with preference strengthening in k.
    #   SDI: at matched count its validity sensitivity has the wrong sign
    #        and is bounded, so it stays at or below chance for the
    #        low-validity coalition on every such trial.
    #
    # The metric is the empirical proportion of choices of the
    # LOW-validity coalition across all matched-count trials
    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96
    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.
    # It is well-defined on a single subject's slice (48 Bernoulli
    # trials) and on the pooled data.

    v = np.array([
        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,
        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,
    ], dtype=float)

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:
            continue

        ca = float(a.sum())
        cb = float(b.sum())
        # Keep only matched-count trials with at least one endorsement
        # on each side (the validity-reversal assay cells).
        if ca == 0.0 or cb == 0.0 or ca != cb:
            continue

        # Mean validity of the endorsing coalition on each side.
        va = float(np.mean(v[a > 0]))
        vb = float(np.mean(v[b > 0]))
        if va == vb:
            continue

        low_is_a = va < vb
        chose_a = (int(row["response"]) == 0)
        hits.append(1.0 if (chose_a == low_is_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_4 (simulated): 0.4600 (var=0.0059)
- Predicted under pi_5 (simulated): 0.8263 (var=0.0045)
- Observed on real data: 0.7808 (var=0.0201)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree
```

**Observed (real) value:** 0.7117 (var=0.0205)
**Predicted under pi_4:** 0.7792 (var=0.0018)
**Predicted under pi_5:** 0.7272 (var=0.0025)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5
```

**Observed (real) value:** -0.2562 (var=0.0128)
**Predicted under pi_4:** -0.2684 (var=0.0040)
**Predicted under pi_5:** -0.2456 (var=0.0038)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Validity-sensitivity contrast for the cue-2 cascade design.

    For every trial we compute (a) the Take-The-Best winner (first
    discriminating cue in validity order) and (b) the VWEI evidence
    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed
    relative to the TTB winner (Erel = E * (+1 if TTB winner is A
    else -1)).

    Two trial classes:
      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides
             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).
      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)
             decides but the weighted evidence of the opposing
             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).

    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).

    TTB predicts the SAME choice probability on every non-tied pair
    (its score is binary), so this difference is 0 in expectation.
    VWEI predicts a large positive difference (~ +0.42). The contrast
    cancels each subject's overall response rate (beta / epsilon),
    leaving only the stimulus-driven validity sensitivity.
    """
    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])
    lo = np.log(v / (1.0 - v))
    w = lo ** 3.0
    order = np.argsort(-v, kind="stable")

    highs = []
    lows = []

    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.shape[0] != 9 or b.shape[0] != 9:
            continue

        # VWEI-style evidence for A over B (gamma = 3).
        d = np.sign(a - b)
        E = float(np.dot(w, d))

        # TTB winner: first discriminating cue in validity order.
        win = None
        for j in order:
            if a[j] > b[j]:
                win = 0
                break
            if b[j] > a[j]:
                win = 1
                break
        if win is None:
            continue  # no discriminating cue (does not occur here)

        # Evidence relative to the TTB winner.
        s = 1.0 if win == 0 else -1.0
        Erel = E * s

        r_val = int(r)
        chose_winner = 1.0 if r_val == win else 0.0

        if Erel > 3.0:
            highs.append(chose_winner)
        elif Erel < -0.5:
            lows.append(chose_winner)

    if not highs or not lows:
        return 0.0

    return float(np.mean(highs) - np.mean(lows))
```

**Observed (real) value:** -0.5467 (var=0.0421)
**Predicted under pi_4:** -0.4146 (var=0.0116)
**Predicted under pi_5:** -0.5708 (var=0.0133)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Signed-evidence slope.

    For every trial we compute the VWEI evidence
        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3
    and the TTB winner (first discriminating cue in validity order;
    validities are strictly descending so cue order = feature order).

    Define the signed regressor  x = +E if TTB's winner is A, x = -E if
    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's
    winner.  Return the OLS slope of y on x.

    TTB's probability of choosing its own winner is a single constant
    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected
    slope is exactly 0.  VWEI's probability of choosing TTB's winner is
    sigmoid(beta * x) on every trial, giving a clearly positive slope
    (~0.05 at the design center, sign-stable over the whole parameter
    box).  The two theories are maximally separated on this statistic.
    """
    if data is None or len(data) == 0:
        return 0.0

    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64], dtype=float)
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)
    w = np.log(v / (1.0 - v)) ** 3.0

    def _to_vec(cell):
        if isinstance(cell, str):
            return [float(ch) for ch in cell if ch in '01']
        return [float(val) for val in cell]

    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):
        return 0.0

    resp = data['response'].to_numpy(dtype=float)

    d = np.sign(a - b)                     # +1 favors A, -1 favors B
    E = d @ w                              # VWEI evidence for A

    disc = d != 0
    keep = disc.any(axis=1)                # drop degenerate all-tie rows
    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]
    if d.shape[0] == 0:
        return 0.0

    # First discriminating cue (validities strictly descending -> order 0..8)
    first_idx = disc.argmax(axis=1)
    rows = np.arange(d.shape[0])
    d_first = d[rows, first_idx]
    winner_a = d_first > 0                  # TTB's winner is option A

    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner
    y = np.where(winner_a, 1.0 - resp, resp)
    x = np.where(winner_a, E, -E)

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    yc = y - y.mean()
    slope = float(np.dot(xc, yc)) / denom
    return slope
```

**Observed (real) value:** -0.1938 (var=0.0027)
**Predicted under pi_4:** -0.2020 (var=0.0005)
**Predicted under pi_5:** -0.2142 (var=0.0004)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Top-cue allegiance under endorsement-majority conflict.

    Selected trials: the 95%-validity expert (feature index 0) endorses
    exactly one option, AND that option carries strictly MORE total
    endorsements (sum of ratings) than its rival.  On these trials the
    metric is the empirical rate of choosing the option endorsed by the
    top expert.

    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98
    here: the amplified log-odds weight of v=.95 dominates every
    possible opposing coalition of the remaining cues, and 50%-expert
    endorsements contribute exactly zero weight.
    SDI must AVOID the top-cue option here (~0.15-0.25): every
    endorsement is a liability, and the top-cue option is also the
    endorsement-majority option.
    """
    df = data.reset_index(drop=True)

    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)
    resp = df['response'].to_numpy(dtype=float)

    if a.ndim != 2 or a.shape[0] == 0:
        return 0.5

    chose_a = resp < 0.5

    # Which option does the top (95%) expert endorse?
    top_a = a[:, 0] > 0.5
    top_b = b[:, 0] > 0.5
    # Keep only trials where exactly one option gets the top expert's
    # endorsement (guards against degenerate rows).
    xor_top = np.logical_xor(top_a, top_b)

    # Total endorsement counts per option.
    cnt_a = a.sum(axis=1)
    cnt_b = b.sum(axis=1)

    # Is the top-cue-endorsed option also the endorsement-majority
    # option (strictly more endorsements)?
    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)

    sel = xor_top & top_option_more
    if not np.any(sel):
        return 0.5

    # Did the subject choose the option endorsed by the top expert?
    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])
    return float(np.mean(followed))
```

**Observed (real) value:** 0.1143 (var=0.0077)
**Predicted under pi_4:** 0.1348 (var=0.0038)
**Predicted under pi_5:** 0.0852 (var=0.0016)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-dissociation composite built on the two strongest contrasts
    # engineered into this design:
    #
    # COMPONENT 1 ("big-gap single endorsers", pairs 6/18, 8 trials/subject):
    #   both options carry exactly ONE endorsement; the endorsers' validities
    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the
    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);
    #   SDI is structurally near-blind at matched endorsement counts
    #   (P ~ 0.50-0.55 across its whole box).
    #
    # COMPONENT 2 ("ridge-crest coalition", pairs 10/22, 8 trials/subject):
    #   one option carries a single 0.80-validity endorsement, the other a
    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing
    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly
    #   favors the single endorser (coalition P ~ 0.10).
    #
    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.
    # Both components use only 8 trials each but their gaps point the SAME
    # way after sign alignment, so the composite has a large mean gap and
    # modest binomial-only between-subject variance.

    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    scores_gap = []    # 1 = chose the higher-validity single endorser
    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser

    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != V.shape or b.shape != V.shape:
            continue
        ca = int(round(a.sum()))
        cb = int(round(b.sum()))
        chose_a = (int(r) == 0)

        if ca == 1 and cb == 1:
            # Single endorsement on each side: who endorsed, and how good
            # is that expert?
            ia = int(np.argmax(a))
            ib = int(np.argmax(b))
            va = V[ia]
            vb = V[ib]
            if abs(va - vb) >= 0.25:
                hi_is_a = (va > vb)
                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)

        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):
            # Ridge-crest conflict: verify it is exactly the designed cell
            # (single 0.80 expert vs four 0.75 experts) and score the
            # coalition choice.
            if ca == 1:
                single_vec, coal_vec = a, b
                single_is_a = True
            else:
                single_vec, coal_vec = b, a
                single_is_a = False
            i_single = int(np.argmax(single_vec))
            coal_idx = np.where(coal_vec > 0.5)[0]
            if (
                abs(V[i_single] - 0.80) < 1e-9
                and len(coal_idx) == 4
                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))
            ):
                # +1 when the subject chose the coalition side
                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)

    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5
    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5
    return 0.5 * (r_gap + r_coal)
```

**Observed (real) value:** 0.1925 (var=0.0159)
**Predicted under pi_4:** 0.3037 (var=0.0089)
**Predicted under pi_5:** 0.1862 (var=0.0087)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments, VPS (pi_5) is decisively the better of the two theories. Scorecard (|predicted \u2212 observed|): Exp 1: SDI 0.241 vs VPS 0.097 (VPS); Exp 2: SDI 0.321 vs VPS 0.046 (VPS, decisively \u2014 SDI's flat ~0.45 prediction is flatly contradicted by the observed 0.78 preference for the LOW-validity coalition at matched counts); Exp 3: SDI 0.068 vs VPS 0.016 (VPS); Exp 4: SDI 0.012 vs VPS 0.011 (tie); Exp 5: SDI 0.132 vs VPS 0.024 (VPS); Exp 6: SDI 0.008 vs VPS 0.020 (slight SDI edge, both excellent); Exp 7: SDI 0.021 vs VPS 0.029 (near tie, both capture the strong anti-top-cue effect); Exp 8: SDI 0.111 vs VPS 0.006 (VPS, decisively). VPS wins 5 of 8 clearly, ties 2, and loses only marginally on one. The diagnosis of SDI's failure is structural, not parametric: SDI's skepticism weight (1\u2212v)^delta is nearly flat over the validity range, making it a validity-blind endorsement counter at matched counts. Experiments 1, 2, and 8 all show that real subjects are NOT validity-blind at matched endorsement counts \u2014 they systematically prefer the option endorsed by the LOWER-validity expert (Exp 2: 0.78; Exp 8 gap component contributing to 0.19; Exp 1's 0.38 sits well below SDI's sign-robust floor of ~0.59). No reparameterization of SDI can fix this because its validity sensitivity has the wrong sign and is bounded by construction. However, VPS is not perfect either: in Exp 1 the observed 0.3815 lies between the two theories (VPS 0.28, SDI 0.62), and its full-box floor was claimed at ~0.40 \u2014 the observed value is at/below VPS's most SDI-like corner, suggesting real subjects' validity-distrust is somewhat dampened relative to VPS's mid parameters, with meaningful between-subject heterogeneity (observed variances 0.02\u20130.04 are 3\u20136x the simulated ones for both theories, hinting at subject-level mixing). Since SDI is structurally degenerate but VPS slightly overshoots in places, the right move is not to regenerate SDI's parameters (which cannot change its sign structure) but to replace SDI with a new theory that is a genuinely stronger competitor to VPS: one that keeps VPS's confidence-distrust core but adds a small residual validity-following (evidence-integration) component and graded dampening, so it can hit the intermediate values (Exp 1's 0.38) that VPS's pure distrust sometimes overshoots, while still reproducing the matched-count validity-reversal signature that killed SDI.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace SDI (pi_4) with a new theory, tentatively named 'Dampened Confidence-Distrust Integration (DCD)'. Core claim: every endorsement is a liability whose magnitude grows with the endorser's claimed validity (as in VPS), but subjects also carry a small residual validity-weighted evidence term and the distrust signal is dampened/graded rather than extreme. Formal sketch: per-endorsement penalty d_j = c0 + c1 * (2*v_j \u2212 1)^kappa (kappa in [0.8, 3], c0 in [0, 0.3], c1 in [0.5, 1.5]); decision variable D = (P(B) \u2212 P(A)) + alpha * sum_j logit(v_j) * (a_j \u2212 b_j), with alpha allowed small positive values in [0, 0.15] (a residual VWEI-like endorsement-following tendency that pulls matched-count choices back toward and slightly past chance); P(A) = sigmoid(beta*D) mixed with lapse epsilon in [0.05, 0.20]; beta in [0.8, 3]. Key contrasts with the two incumbent theories: (i) unlike SDI, DCD's distrust weight is steeply INCREASING in validity, so it reproduces the matched-count validity-reversal ladder (Exp 2 observed 0.78, Exp 8's 0.19) that structurally falsified SDI; (ii) unlike VPS, the alpha term and the (2v\u22121)^kappa rescaling (which is flatter near v=0.5 than v^kappa) soften the extremes, letting DCD land on intermediate values like Exp 1's observed 0.38 that pure VPS undershoots, while still predicting clear majorities in the same direction as VPS on Exps 2, 5, and 8. The between-subject variance pattern (real var 2\u20136x simulated) further suggests allowing subject-level heterogeneity in alpha and kappa, which the parameter ranges above accommodate. This makes DCD a strict generalization that nests VPS as a limiting case (alpha\u21920, rescaling absorbed) while repairing the one region (Exp 1's composite) where VPS's point predictions were furthest from the data on the VPS-favoring side."
}
```

## Usage

```json
{
  "input_tokens": 27556,
  "output_tokens": 2688,
  "reasoning_tokens": 1450,
  "provider": "Sail Research"
}
```
