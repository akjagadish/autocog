# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_3
Validity-Weighted Evidence Integration (VWEI) with subjective validity amplification. On each trial the decision maker computes, for every rating dimension j, the discriminant d_j = sign(a_j - b_j) in {-1, 0, +1}, weights it by a subjectively amplified validity w_j = [log(v_j/(1-v_j))]^gamma, and accumulates evidence E = sum_j w_j * d_j. Choice probability is a sigmoid of beta * E mixed with a small uniform lapse epsilon. gamma = 1 recovers Bayes-optimal log-odds integration; gamma -> infinity recovers Take-The-Best; the empirically human region sits at gamma near 3, where the top cue dominates coalitions of weaker cues on steep validity gradients (Exp 2) but lower cues still attenuate allegiance on shallow gradients (Exp 1). This round's refinement is purely parametric: the mechanism is unchanged, but the sampling region is re-centered on the hand-verified sweet spot (gamma = 3, beta = 0.2, epsilon = 0), the lapse range is restored to (a tightened version of) the arbiter's [0, 0.2], and the beta floor is raised so that generic flattening noise no longer competes with the gamma-structured weight profile for attenuation work.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Weighted Evidence Integration (VWEI) with subjective
    # validity amplification. Stimulus is the pair of option feature
    # vectors for the current trial: array-like of shape
    # (2, n_features), row 0 = option A, row 1 = option B.
    #
    # Evidence for A over B:  E = sum_j w_j * sign(a_j - b_j),
    # with w_j = [log(v_j / (1 - v_j))]^gamma.
    #   gamma = 1  -> Bayes-optimal log-odds weighting (full integration)
    #   gamma -> inf -> Take-The-Best (top cue dominates every coalition)
    # P(A) = sigmoid(beta * E), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VWEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the log-odds against v = 1 (infinite weight) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)
    log_odds = np.log(v / (1.0 - v))

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Subjective cue weights: amplified log-odds validity.
    w = np.sign(log_odds) * np.abs(log_odds) ** gamma

    # Per-cue discriminant: +1 favors A, -1 favors B, 0 is uninformative.
    d = np.sign(a - b)

    # Accumulated evidence for A (positive) vs B (negative).
    E = float(np.dot(w, d))

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # When E == 0 this is exactly 0.5 (uniform guessing), for any beta.
    x = beta * E
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
**Validities (n_features=10):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 22: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 25: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 26: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 27: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 29: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 31: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-3): ZERO-WEIGHT EXPERTS. All prior designs dissociated theories that all read endorsements as positive evidence. SDI versus VWEI is a clash of SIGN and of WEIGHTING PROFILE: SDI treats every endorsement as a liability (penalty d_j = (1-v_j)^delta, nearly uniform at ~0.86-1.0 across the whole validity range for delta in [0, 0.05]) and prefers the option with FEWER endorsements; VWEI weights endorsements by amplified log-odds (w_j = logit(v_j)^gamma, enormous spread: 25.5 at v=.95 down to 0.61 at v=.70) and prefers the option with MORE validity-weighted wins. The design's keystone is four experts at validity EXACTLY 0.50: VWEI's weight for a 50% expert is logit(0.5)^gamma = 0 for EVERY gamma — a 50% expert's endorsement is structurally invisible to VWEI under any parameterization — while SDI's skepticism weight is (1-0.5)^delta in [0.966, 1], its NEAR-MAXIMUM penalty. A coin-flip expert's praise is, for SDI, a full unit of suspicion. Validities: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.50, 0.50, 0.50, 0.50] (10 experts; VWEI weights at gamma=3: 25.5, 10.6, 5.2, 2.7, 1.3, 0.6, 0, 0, 0, 0; SDI penalty weights at delta=0.025: 0.93, 0.94, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98). 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject). In every base pair VWEI's winner is A, so the design is perfectly side-balanced. FOUR QUANTITATIVE SIGNATURES (P(A) at SDI mid-params delta=.025, beta=1, eps=.115, alpha=.025; VWEI mid-params gamma=3, beta=.205, eps=.015; all signs verified robust across the FULL parameter boxes): (1) JUNK-ENDORSEMENT GRADIENT, VWEI STRUCTURALLY FLAT (pairs 1-5): the 95% expert endorses A; k = 0..4 of the 50% experts also endorse A. VWEI: E = 25.5 regardless of k (50% weight is exactly 0 at any gamma), so P(A) is flat at 0.98-0.99. SDI: each junk endorsement adds ~0.98 penalty units to A, so P(A) declines monotonone 0.32 -> 0.18 -> 0.11 -> 0.08 -> 0.06 (~6-7 points per junk endorsement; box ranges 0.32-0.33 down to 0.05-0.08). ANY measurable effect of 50%-expert endorsements on choice falsifies VWEI outright — it has no mechanism, at any gamma, beta, or epsilon, for a zero-weight cue to move its prediction off a constant; flatness at ~0.99 with no decline falsifies SDI. Pairs 6-7 run the mirror logic (junk endorsements on B): VWEI flat 0.99; SDI rises 0.53 -> 0.72 as B accumulates 'suspicion'. (2) OPPOSITE-SLOPE DOUBLE DISSOCIATION (pairs 10-14): cue 1 (95%) endorses A while the opposing coalition among cues 2-6 grows from 1 to 5. VWEI: E = 25.5 - cumulative weight, so P(A) DECLINES 0.95 -> 0.87 -> 0.81 -> 0.76 -> 0.74. SDI: each opposing endorsement adds a penalty to B, so P(A) RISES 0.51 -> 0.69 -> 0.82 -> 0.89 -> 0.92. The two profiles are anti-correlated and CROSS between k=2 and k=3 — slope signs are parameter-independent (VWEI's E strictly decreases in k because every added cue has positive weight; SDI's D strictly increases because every added endorsement adds positive penalty), so the sign of the empirical slope across this 5-point gradient adjudicates the theories with no free parameters able to reverse it. (3) SIGN CONFLICT AT CEILING (pairs 8-9, plus mirrors 24-25): A endorsed by 3 (then 5) high-validity experts, B by none. VWEI: P(A) ~ 0.99. SDI: P(A) = 0.12 (3 endorsements) falling to 0.07 (5 endorsements) — majority choices OPPOSITE to VWEI on stimuli where VWEI is at ceiling, which is SDI's novel falsifiable prediction (high-validity option carrying MORE endorsements). SDI additionally predicts a within-family decline (0.12 -> 0.07) where VWEI is flat at ceiling. (4) VALIDITY-BLINDNESS AT MATCHED ENDORSEMENT COUNTS (pairs 15-16): exactly one endorsement per option, varying expert identity (95% vs 70%; 90% vs 70%). SDI: P(A) = 0.51-0.52 regardless — its penalty weights span only ~9% across the entire validity range at delta <= 0.05, so at matched counts SDI is essentially indifferent to WHO endorses (its only tilt, +1-2 points toward the higher-validity endorser, comes from delta and the alpha tie-break). VWEI: 0.99 and 0.88 — a 40+ point sensitivity to expert identity that SDI cannot produce under any delta in [0, 0.05]. No pair is degenerate: every pair has discriminating cues, no pair leaves both theories guessing (the closest SDI gets to 0.5 is on D-1/E pairs where VWEI is at 0.88-0.95), and every pair requires actually comparing the two rating vectors. The conjunction is un-fakeable in both directions: a VWEI subject is a flat line at ~0.99 across the entire junk gradient and a declining line across the coalition gradient; an SDI subject is a specific descending curve on junk endorsements, a specific ascending curve on opposing coalitions, near-50% at matched counts, and B-majorities on the ceiling-conflict pairs. Neither theory's parameter box can approximate the other's profile because the key contrasts (zero vs. near-maximal weight on 50% experts; fewer- vs. more-endorsements preference) are structural, not parametric.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the sharpest structural sign clash in the design. Ten of the 32 base pairs (trials 1-5, 8-9 and their A/B mirrors 17-21, 24-25; 30 observations per subject after 3 repetitions) have the property that the 95%-validity expert endorses one option and that same option carries strictly more total endorsements than its rival (which is either empty or carries only junk 50%-expert endorsements). On exactly these trials the two theories make opposite majority predictions with no parameter able to reverse the sign. VWEI: the v=.95 cue's weight logit(.95)^gamma (~25.5 at gamma=3, still the largest weight for any gamma in [2.8,3.2]) dominates every possible opposing coalition of cues 2-6, and 50%-expert endorsements carry exactly zero weight at every gamma, so P(follow top cue) sits at ~0.97-0.99 (sigmoid(0.205*25.5) with the small lapse). SDI: every endorsement adds a near-unit penalty (d_j in [0.93,0.98] for delta in [0,0.05]), and the top-cue option is always the endorsement-majority option here, so P(follow top cue) averages ~0.18 pre-lapse, ~0.22 after the epsilon lapse — i.e., the majority choice is the LESS-endorsed option. Expected gap ~0.75 (SDI ~0.22 vs VWEI ~0.98) with per-subject binomial standard error of only ~0.07 (30 trials, p near 0.2 or 0.98), so at N=25 the Welch t-test separates the theories by dozens of standard errors. The metric is a plain choice rate, so it is well defined both on the pooled DataFrame and on each single subject's slice, and it uses only scalar hashing of the response column (sequence columns are converted to arrays, never hashed). Unlike a slope estimate over the junk-endorsement gradient (which is noisy at 6 samples per cell), this is a simple proportion over 30 trials per subject, keeping between-subject variance tight. It is falsifiable in both directions: near-ceiling following of the top cue under endorsement-majority conflict refutes SDI; majority avoidance of the top cue refutes VWEI, which has no mechanism (at any gamma, since gamma only rescales positive log-odds weights) for preferring the option with fewer endorsements.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.1348 (var=0.0038)
- Predicted under pi_3 (simulated): 0.9867 (var=0.0003)
- Observed on real data: 0.1143 (var=0.0077)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=10):** [0.9, 0.85, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.65, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-4): a CARDINAL WEIGHT-MATCHING RIDGE built from duplicate moderate validities, with no zero-validity experts and no dominant top cue. Exp 4 dissociated VWEI from SDI structurally (50% experts are invisible to VWEI, penalized by SDI) and via 95%-anchored ceiling conflicts. This design instead engineers trials where VWEI's evidence sum is arithmetically pinned to ~0 by matching coalitions of moderate experts against single stronger experts, and dual trials where SDI's penalty difference is pinned to ~0 by matching endorsement counts. The validities [0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60] are chosen so that VWEI's amplified log-odds weights satisfy 2*w(0.75) = w(0.80) exactly at gamma = ln2/ln(1.2616) = 2.98 and 4*w(0.75) = w(0.85) exactly at gamma = ln4/ln(1.579) = 3.03 -- both crossing points sit at the CENTER of VWEI's own gamma box [2.8, 3.2], so the weight-matching holds (|E| <= 0.53, hence P within +/-2.5 points of 0.50) across the ENTIRE parameter box, verified at gamma in {2.8, 3.0, 3.2} x beta in {0.18, 0.23}. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- EVIDENCE-MATCHED COUNT GRADIENT (pairs 1-5): one option carries 1 (or 2) endorsements, the other carries a weight-matched coalition of 2-4 moderate experts (e.g., A = {85%} vs B = {80%, 75%, 75%}: E = 5.22 - 5.32 = -0.10). VWEI is pinned at chance: P(fewer-endorsement option) = 0.49-0.50 on all five pairs (box range 0.48-0.52) -- it has no mechanism, at any gamma/beta/epsilon in its box, for these stimuli to move it off a guess, because its cardinal weights cancel by construction. SDI climbs monotonically with endorsement-count difference: 0.69 (1v2), 0.70 (1v2, second realization), 0.83 (1v3), 0.83 (2v4), 0.89 (1v4) (box ranges 0.65-0.73 up to 0.86-0.92), since every net endorsement adds ~0.96 penalty units and its D strictly increases with count difference -- the slope sign is parameter-independent. Any systematic departure from 50% toward the fewer-endorsement option on pairs 1-5 falsifies VWEI; flatness at 50% falsifies SDI. FAMILY 2 -- COUNT-MATCHED VALIDITY-GAP GRADIENT (pairs 6-9): exactly one endorsement per side, with the endorser validity gap shrinking (.90 vs .60; .85 vs .65; .85 vs .80; .80 vs .75). SDI is pinned at 0.50-0.52 on all four (its penalty weights (1-v)^delta span <5% across the whole validity range at delta <= 0.05, and the alpha tie-break adds at most 3 points) -- structurally near-blind to WHO endorses at matched counts. VWEI sweeps 0.89 -> 0.73 -> 0.62 -> 0.56 (box ranges 0.84-0.95 down to 0.55-0.59), a 33-point monotone gradient in endorser validity that no SDI parameterization can produce. FAMILY 3 -- SIGN CONFLICT AT THE RIDGE CREST (pair 10): four 75% experts (total weight 5.30) vs one 80% expert (weight 2.66). VWEI: majority for the 4-endorsement coalition, P(A) = 0.63 (box 0.61-0.65). SDI: majority for the single endorsement, P(A) = 0.10 (box 0.08-0.12) -- opposite majorities, both far from chance, sign-robust across both boxes. BASELINE/SUB-RIDGE (pairs 11-12): pair 11 ({90%} vs {65%, 60%}) is the one agreement cell (VWEI 0.89, SDI 0.71 -- both prefer A, calibrating each theory's scale where they concur); pair 12 ({80%, 75%} vs {85%}) sits just BELOW the matching ridge (VWEI 0.43, SDI 0.30 -- both prefer B but at parametrically distinct depths, gap >= 9 points across boxes). THE DOUBLE DISSOCIATION IS UN-FAKEABLE: the two theories are placed in orthogonal indifference regimes -- VWEI is indifferent exactly when endorsement counts differ at matched evidence (Family 1), SDI is indifferent exactly when counts match but validity composition differs (Family 2) -- and no reparameterization of either theory can escape its own indifference condition (VWEI's weights cancel by arithmetic fixed at gamma ~ 3; SDI's penalty weights are count-driven and validity-flat by delta <= 0.05). A VWEI subject is a flat 50% line across the entire count gradient and a steep monotone curve across the gap gradient; an SDI subject is a steep monotone climb across the count gradient and a flat ~51% line across the gap gradient, plus a strong majority for the single endorser on pair 10 where VWEI favors the coalition. All pairs are non-degenerate (every pair has discriminating cues; no all-tie pairs; no pair leaves both theories guessing), and every trial requires actually comparing the two rating vectors.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design's double dissociation is exploited at its two strongest, sign-aligned points. COMPONENT 1 uses the count-matched big-validity-gap trials (pairs 6/18: a single 0.90 endorsement vs a single 0.60 endorsement, 8 trials per subject). Here VWEI's amplified log-odds weights produce P(follow the 0.90 expert) ≈ 0.84-0.94 across its entire (gamma, beta, epsilon) box, while SDI is pinned near chance (0.50-0.55) because its penalty difference at equal endorsement counts is driven only by delta ≤ 0.05 and alpha ≤ 0.05 — it is structurally near-blind to WHO endorses when counts match. COMPONENT 2 uses the ridge-crest conflict (pairs 10/22: four 0.75 experts vs one 0.80 expert, 8 trials per subject), scored +1 for choosing the coalition. VWEI's evidence sum favors the coalition (P ≈ 0.61-0.65 box-wide, since 4*w(0.75) > w(0.80) at gamma ≈ 3), while SDI's penalty difference of ~3 units at beta ≈ 1 drives P(coalition) ≈ 0.08-0.12 box-wide. Averaging the two aligned rates yields an expected pooled value of ~0.75 under VWEI vs ~0.31 under SDI — a ~0.44 mean gap. Between-subject variance is essentially binomial (each component uses 8 trials; var of the average ≈ 0.01 for both theories), so at N = 25 the Welch t-statistic is expected around 15 (p << 0.001). The metric is a plain choice rate on structurally identified trial cells (no theory fitting, no likelihood evaluation), works identically on the pooled DataFrame and on single-subject slices, avoids hashing any sequence-valued column, and degrades gracefully to 0.5 if a cell is somehow absent. Note the sign alignment is essential: both components are scored in the direction VWEI predicts (follow the strong expert; follow the weight-matched coalition), so the two gaps add rather than cancel.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.3037 (var=0.0089)
- Predicted under pi_3 (simulated): 0.7625 (var=0.0125)
- Observed on real data: 0.1925 (var=0.0159)

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
**Predicted under pi_3:** 0.6869 (var=0.0033)

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
**Predicted under pi_3:** -0.2566 (var=0.0023)

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
**Predicted under pi_3:** 0.4433 (var=0.0115)

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
**Predicted under pi_3:** 0.0475 (var=0.0011)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, SDI (pi_4) is decisively the better account, but it is not yet a fully calibrated one, and VWEI (pi_3) is degenerate. Experiment-by-experiment: (1) Exp 1 (top-cue allegiance under endorsement-majority conflict): observed 0.1143 vs SDI 0.1348 / VWEI 0.9867 \u2014 a catastrophic sign error for VWEI; it has no mechanism at any gamma for preferring the less-endorsed option, and the 50%-expert invisibility it was built around is empirically irrelevant. (2) Exp 2 (double-dissociation composite): observed 0.1925 vs SDI 0.3037 / VWEI 0.7625 \u2014 again VWEI is on the wrong side of the gap by ~0.57, while SDI gets the sign right but undershoots the extremity by ~0.11. (3) Exp 3 (TTB-vs-tally conflict): observed 0.7117 vs SDI 0.7792 / VWEI 0.6869 \u2014 the one experiment where VWEI is marginally closer (off by 0.025 vs SDI's 0.067); both theories get the sign right. (4) Exp 4 (tally-allegiance under top-cue conflict): observed -0.2562 vs SDI -0.2684 / VWEI -0.2566 \u2014 both essentially perfect; VWEI's near-exact hit here shows it is not worthless, but a theory that nails one metric while inverting the sign on four others is not task-invariant. (5) Exp 5 (validity-sensitivity contrast): observed -0.5467 vs SDI -0.4146 / VWEI +0.4433 \u2014 VWEI's sign is wrong; SDI's sign is right but again undershoots the extremity by ~0.13. (6) Exp 6 (signed-evidence slope): observed -0.1938 vs SDI -0.2020 / VWEI +0.0475 \u2014 SDI is nearly exact; VWEI has the wrong sign. Net: SDI wins 4 of 6 decisively (1, 2, 5, 6), ties VWEI on 4, and loses marginally only on 3. VWEI fails outright on 4 of 6 \u2014 it is degenerate in this domain and should be replaced. However, SDI shows a systematic bias: on Exps 1, 2 and 5 the observed data are MORE extreme in SDI's own predicted direction than SDI's simulated values (0.114<0.135, 0.193<0.304, -0.547<-0.415). The largest discrepancy, Exp 2, is diagnostic: the composite's first component (single 0.90 endorser vs single 0.60 endorser, matched counts) requires a rate well BELOW 0.5 to reconcile the observed 0.193 with the near-zero coalition rate, yet SDI is structurally pinned at ~0.5 there (its penalty weights (1-v)^delta span <10% over the whole validity range for delta in [0, 0.05], and its alpha tie-break can only push TOWARD the higher-validity endorser). The data hint that at matched endorsement counts subjects lean toward the LOWER-validity endorser \u2014 i.e., the liability of an endorsement grows with the endorser's claimed validity, the opposite of SDI's near-flat, slightly decreasing skepticism weights. This is a real, mechanistic refinement, not a parameter tweak of SDI, and it is exactly the axis on which a stronger competitor can be built.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 (VWEI) with a brand-new competitor to SDI: Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Mechanism: each endorsement is a liability whose magnitude scales with the endorser's claimed validity \u2014 a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. Concretely, the penalty for option X is P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1=0 recovers pure SDI-like counting) and c1 * v_j^kappa is the confidence-distrust term (penalty INCREASING in validity, kappa controlling how steeply). The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon; history is ignored (validities are in the instructions). Hand-verified signatures at kappa=2, beta=2, epsilon=0.1, c0=0.1, c1=1: Exp 1 metric ~0.10 (more endorsements -> more penalty -> avoid the top-cue option; matches observed 0.114); Exp 2 gap component: P(follow 0.90 endorser) ~ 0.35-0.40 < 0.5 (the 0.90 endorsement carries a larger penalty than the 0.60 endorsement at matched counts \u2014 the key improvement over SDI, which is pinned at 0.5), coalition component ~0.05-0.10, composite ~0.15-0.20 (matches observed 0.193, which SDI cannot reach); Exp 3 ~0.70-0.75 (fewer endorsements wins, but the high-validity single endorsement is partially self-defeating, tempering the rate toward the observed 0.712 \u2014 better than SDI's 0.779); Exp 4 ~ -0.25 to -0.30 (penalty of the 0.95 cue nearly cancels the 3-endorsement coalition, leaving a moderate TTB-ward tilt; matches observed -0.256); Exp 5 ~ -0.50 to -0.60 (HIGH trials: the 0.90-endorsed TTB winner carries the largest single penalty -> strongly avoided; LOW trials: the 7-endorsement coalition drowns in accumulated penalty -> TTB winner strongly preferred; matches observed -0.547 and fixes SDI's undershoot); Exp 6 slope ~ -0.15 to -0.25 (higher validity-weighted evidence behind the TTB winner means higher-validity endorsements means higher penalty -> negative slope; matches observed -0.194). Suggested parameter ranges: kappa in [1, 3], beta in [1, 4], epsilon in [0.03, 0.15], c0 in [0, 0.3], c1 in [0.5, 1.5]. VPS differs from SDI on a sharp, falsifiable axis: matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser). SDI predicts ~0.50 (validity-blind at matched counts); VPS predicts a clear majority for the LOWER-validity endorser. The Exp 2 observed composite of 0.193 is only reachable if that component sits well below 0.5, so existing data already favor VPS over SDI on this axis while both agree on every sign SDI got right. VPS also differs from VWEI everywhere VWEI failed: it treats endorsements as liabilities (not assets), it is sensitive to 50%-expert endorsements (penalty c0 + c1*0.5^kappa > 0), and it predicts negative signed-evidence slopes. Next-round experiments should manipulate the c0/c1 trade-off: designs with matched endorsement counts and graded validity gaps (VPS: monotone preference for weaker endorsers; SDI: flat ~0.5), and designs pitting flat defect counting against validity-proportional suspicion (e.g., one 0.95 endorsement vs one 0.55 endorsement plus one 0.60 endorsement \u2014 SDI prefers the single, VPS can prefer the pair if c1 is large)."
}
```

## Usage

```json
{
  "input_tokens": 19399,
  "output_tokens": 7650,
  "reasoning_tokens": 5847,
  "provider": "Baidu"
}
```
