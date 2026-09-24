# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 13: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 14: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 15: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 17: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 18: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 19: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 20: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 22: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 24: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. The advocated theory (SCWI) and the competitor (Tallying) have a deep structural relationship on binary cues: because SCWI's signed value difference between options is sum_j w_j*(a_j - b_j) with w_j = -(v_j + tau*v_j^2), tied cues cancel exactly, and SCWI effectively computes an ANTI-VALIDITY-WEIGHTED tally: it prefers the option that LOSES on more (and more heavily weighted) cues. Tallying prefers the option that WINS on more cues, ignoring validity entirely. Two things follow. (1) SIGN: on every trial with a nonzero tally margin the two theories prefer OPPOSITE options. (2) SLOPE: Tallying's evidence is the raw count margin; SCWI's evidence is the validity-weighted (inverted) margin, so the two theories order trials differently. The design exploits both, with a parameter-free kill shot: on any trial where the feature-wise counts are TIED (a_wins = b_wins), Tallying's softmax is exactly uniform for ANY beta and ANY epsilon (lapse is symmetric), pinning P(A) = 0.500 exactly. SCWI is not so pinned: a count tie with unequal validity composition yields a nonzero weighted margin. PROBE A (pairs 1-8, count ties): I construct 8 count-tie trials whose SCWI weighted margin M = sum|w| over B's winning cues minus sum|w| over A's winning cues sweeps from -0.92 to +0.92 (computed at tau=0.35; the sweep is robust across tau in [0.25,0.45]). Tallying predicts EXACTLY 0.500 on all eight, a horizontal line with zero free parameters. SCWI predicts a monotone psychometric sweep of P(A) from ~0.25 up to ~0.75, with the specific ordering implied by the supralinear anti-validity weights (e.g., pair 3, where A wins the two top experts and B wins the two worst, yields P(A)~0.25; pair 4, the cue-label swap, yields ~0.75). Any reliable deviation from 0.5 on these trials falsifies Tallying outright; the graded M-ordering then tests SCWI quantitatively. PROBE B (pairs 9-15, tally margin fixed at +1 for A: A wins exactly 3 cues, B wins 2): Tallying predicts a FLAT P(A) = sigmoid(beta)*(1-eps)+eps/2 on all seven, while SCWI predicts a strictly graded function of which cues A wins — from P(A)~0.11 when A wins the three most trusted experts (its 'defect load' is maximal) to P(A)~0.49 when A wins the three least trusted. This is a flat-vs-graded slope dissociation at fixed tally evidence, and every SCWI prediction sits BELOW 0.5 while Tallying's sits ABOVE 0.5: a double (sign + slope) dissociation. PROBE D (pairs 16-19) mirrors four Probe-B trials (A/B swapped) to absorb side and response bias; Tallying stays flat at sigmoid(-beta), SCWI stays graded above 0.5. ANCHORS (pairs 20-23): extreme trials where A wins all cues (pair 20), B wins all (21), or A wins four low-validity cues against B's single top cue (22/23). Tallying approaches deterministic choice of the tally winner (sigmoid(5*beta), sigmoid(3*beta)); SCWI approaches deterministic choice of the tally LOSER, with confidence tracking the weighted margin M rather than the count margin — plotting P(A) against count margin gives a rising function under Tallying and a falling function under SCWI. Pair 24 is a full tie (identical vectors) where both theories predict exactly 0.5, providing a pure lapse/side-bias baseline. With 24 unique pairs, K = 96//24 = 4 repetitions (96 trials). DECISION RULE: (i) if P(A) on pairs 1-8 is reliably different from 0.5 in either direction, Tallying is falsified (it cannot produce this for any parameter values); (ii) if P(A) on pairs 9-15 is flat, Tallying is supported and SCWI is falsified; (iii) if P(A) across pairs 1-15 is a monotone function of the anti-validity weighted margin M (increasing in M, crossing 0.5 near M=0, with the tally loser preferred), SCWI is supported; (iv) a preference for the tally WINNER that is nonetheless graded by validity weights would falsify both pure forms (it would indicate pro-validity weighted integration).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the OLS slope of the binary choice (indicator of choosing A) on the fixed, parameter-free signed evidence D = sum_j (v_j + 0.35 v_j^2)(b_j - a_j), computed over all 96 trials. This exploits both structural knives in the design simultaneously. (1) SIGN: SCWI's evidence for A is exactly the anti-validity weighted margin; with beta in [1.1,1.7] and epsilon in [0.05,0.15], the average derivative of its choice probability w.r.t. D is about 0.20-0.31 (softmax gain times average p(1-p) ~ 0.2), so the slope is robustly positive. (2) Tallying is pinned at slope <= 0 for ANY parameter values: on the eight count-tied probe trials (pairs 1-8) its softmax is exactly uniform (P(A)=0.5 regardless of beta and epsilon), contributing zero slope regardless of how D sweeps there (D ranges -0.98 to +0.98 across those trials); on all remaining trials its choice probability is a function of the raw count margin, which is strongly anti-correlated with D in this stimulus set (the tally winner is the weighted-margin loser on essentially every non-tied trial, e.g. pair 20: count margin +5 for A but D = -4.70). Hence Tallying's slope is ~0 for small beta and clearly negative for large beta — never positive. The per-subject sampling noise of the slope is small: with Var(D) ~ 2.9 across the 96 trials and binary residuals, the binomial standard error is ~0.03, and parameter heterogeneity adds only ~0.03 more (SCWI slope varies within [0.20, 0.31] across its parameter box; the sign of D is invariant over tau in [0.25,0.45], so the fixed tau=0.35 in the metric only mildly attenuates the slope). A mean gap of at least ~0.2 against a per-subject SD of ~0.04 gives a Welch t around 15-25 at N=25, far below alpha=0.01. The metric works identically on the pooled DataFrame and on a single subject's 96-trial slice, so between-subject variance is well defined.

Source:
def metric(data: pd.DataFrame) -> float:
    """Slope of P(choose A) regressed on the anti-validity weighted cue
    margin D = sum_j u_j * (b_j - a_j), with u_j = v_j + tau * v_j^2
    (tau fixed at 0.35, the midpoint of SCWI's tau range).

    D > 0 means option A carries fewer/smaller anti-validity 'defects'
    and is therefore preferred by SCWI. SCWI predicts a POSITIVE slope
    (~0.2-0.3 with beta in [1.1,1.7], epsilon in [0.05,0.15]).
    Tallying cannot produce a positive slope: on count-tied trials
    (pairs 1-8, 24) it outputs exactly 0.5 for any beta/epsilon, and on
    all remaining trials its choice probability tracks the raw count
    margin, which is strongly ANTI-correlated with D across this design
    (the tally winner is almost always the weighted-margin loser). So
    Tallying's slope is <= 0 (approximately 0 for small beta, clearly
    negative for large beta). The sign separation is parameter-free.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    tau = 0.35
    u = validities + tau * np.square(validities)

    def _to_arr(cell):
        arr = np.asarray(cell, dtype=float).ravel()
        if arr.size != u.size:
            raise ValueError(
                f"Expected {u.size} ratings, got {arr.size}"
            )
        return arr

    a = np.vstack([_to_arr(c) for c in data["option_a_ratings"]])
    b = np.vstack([_to_arr(c) for c in data["option_b_ratings"]])

    # Signed SCWI evidence in favor of A (positive -> A preferred).
    D = (b - a) @ u

    # response: 0 = chose A, 1 = chose B  ->  indicator of choosing A.
    choice_a = 1.0 - np.asarray(data["response"], dtype=float).ravel()

    Dc = D - D.mean()
    Yc = choice_a - choice_a.mean()
    denom = float(np.sum(Dc * Dc))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(Dc * Yc) / denom)

### RESULTS
- Predicted under pi_3 (simulated): 0.1378 (var=0.0002)
- Predicted under pi_2 (simulated): -0.1263 (var=0.0006)
- Observed on real data: 0.1227 (var=0.0008)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [1.0, 0.95, 0.65, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 11: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** ADVERSARIAL LOGIC (the 'transpose' of the count-tie trick). On binary cues, Tallying's evidence is the integer count margin m = a_wins - b_wins, while SCWI's evidence is the anti-validity weighted margin D = -sum_j (v_j + tau*v_j^2)*s_j (s_j = +1 if A wins cue j, -1 if B wins). The already-explored design pinned TALLYING at 0.5 (count ties) and let SCWI vary. I do the exact inverse: I engineer WEIGHTED TIES (D ~= 0, robustly across the full tau range [0.25, 0.45]) at NONZERO tally margins, so SCWI is pinned near 0.5 while Tallying is graded — and then hold D at a large fixed negative value while m varies, so SCWI is flat-low while Tallying rises. This yields a two-probe double slope dissociation in the opposite assignment from the prior design.

VALIDITY ENGINEERING. With validities [1.00, 0.95, 0.65, 0.54, 0.51, 0.50], a weighted tie between a cue set X and cue set Y requires sum_X v + tau*sum_X v^2 = sum_Y v + tau*sum_Y v^2. I choose sets where the constant and tau-coefficient nearly cancel simultaneously: e.g., the four low cues {3,4,5,6} (sum v = 2.20, sum v^2 = 1.224) vs the two top cues {1,2} (1.95, 1.9025): residual c(tau) = 0.25 - 0.678*tau, i.e., |c| <= 0.080 for ALL tau in [0.25, 0.45]. Similar cancellations: {4,5} vs {2} (|c| <= 0.058) and {3,6} vs {1} (|c| <= 0.068).

PROBE A (pairs 1-6, weighted ties at nonzero tally margin): pairs 1-2 have m = +1 and pair 3 has m = +2, all with |D| <= 0.08 across the entire SCWI parameter box. SCWI therefore predicts P(A) = 0.500 +/- 0.032 on ALL SIX (including mirrors) for ANY beta in [1.1,1.7], tau in [0.25,0.45], epsilon in [0.05,0.15] — a horizontal line with essentially no free parameters. Tallying predicts a graded ladder: P(A) = sigmoid(m*beta)*(1-eps)+eps/2, with lapse-corrected logits exactly proportional to m (the m=+1 and m=+2 trials must sit in 1:2 logit ratio, symmetric under mirroring). Any reliable, mirror-symmetric departure from 0.5 that grows with |m| falsifies SCWI outright; perfect flatness at 0.5 falsifies Tallying (which must exceed 0.5 at m>0 for any parameters).

PROBE B (pairs 7-12, fixed weighted evidence, varying tally margin): three trials with D pinned in a narrow confident band — pair 7: m=+1, D in [-1.30,-1.13]; pair 8: m=+2, D in [-1.18,-1.02]; pair 9: m=+3, D in [-1.30,-1.26]. SCWI predicts a FLAT-LOW P(A) ~ 0.15-0.26 on all three (mirrors ~0.74-0.85), because its evidence barely changes. Tallying predicts P(A) >= 0.5 always (m>0), strictly increasing in m: ~0.86 -> 0.96 -> 0.97 at beta=2. Huge vertical separation plus opposite slope sign. Note this is the inverse hold of the prior design's fixed-margin probe (there: m fixed, D varied; here: D fixed, m varies).

MINIMAL PAIR (pairs 3 vs 9): identical stimuli except cue 2 changes from a B-win to a tie. Tallying: m goes +2 -> +3, P(A) barely moves (0.96 -> 0.97). SCWI: D jumps from -0.01 to -1.28, P(A) collapses 0.50 -> 0.18. A single-cue manipulation that one theory treats as negligible and the other as decisive.

ANCHORS AND CONTROLS: pairs 13/14 (one option wins all six cues, m = +/-6, |D| ~ 5.0-5.6) are maximum-separation trials — Tallying ~0.97 vs SCWI ~epsilon/2 (~0.05). Pairs 15/16 are full ties (identical vectors, one all-0 and one mixed) where BOTH theories predict exactly 0.500, providing a pure lapse/side-bias baseline and verifying that tie values (0 vs 1) are inert. All critical pairs are mirrored to absorb side/response bias. 16 unique pairs, K = 96//16 = 6 repetitions = 96 trials.

DECISION RULES: (i) If P(A) on pairs 1-6 deviates from 0.5 symmetrically and grows with |m| (logit ratio ~1:2 for m=1 vs m=2), Tallying is supported and SCWI is falsified — SCWI cannot exceed 0.532 or dip below 0.468 there for any allowed parameters. (ii) If pairs 1-6 are flat at 0.5 while pairs 7-12 sit flat at ~0.2 (mirrors ~0.8) and pair 13 shows P(A) ~ 0.05 (anti-following the all-1s option), SCWI is supported and Tallying is falsified — Tallying cannot go below 0.5 - eps/2 at m>0. (iii) Flat 0.5 on Probe A combined with a Tallying-like increase on Probe B (or any preference for the tally winner graded by validity weights) falsifies both pure forms. Bonus: pair 9 also separates TTB (which picks B via cue 1) from Tallying, so a TTB-like subject is identifiable rather than confounded with SCWI.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric isolates the trials the design engineered as maximal-conflict: Probe B (pairs 7-12) and the anchors (pairs 13-14), where the tally margin is positive (|m| in {1,3,6}) but the SCWI signed evidence strongly favors the tally loser (|D| between ~1.0 and ~5.2). Selection is done from the raw ratings themselves: I recompute the tally margin and the anti-validity weighted evidence with tau = 0.35 (the midpoint of SCWI's box; the selection is insensitive to tau because Probe B's |D| stays in [1.0, 1.3] and Probe A's stays <= 0.08 across the whole tau range, so the 0.8 threshold separates them robustly). Full-tie trials (pairs 15/16) are dropped since both theories predict exactly 0.5 there. On the selected 48 trials per subject, Tallying predicts P(follow tally winner) = (1-eps)*sigmoid(beta*|m|) + eps/2, which is strictly above 0.5 for any beta > 0 and epsilon < 1 (worst case ~0.53 at beta=0.1, eps=0.5; ~0.95+ for typical beta). SCWI predicts (1-eps)*(1-sigmoid(beta*|D|)) + eps/2, i.e. ~0.16-0.20 on Probe B and ~eps/2 (~0.075) on the anchors, averaging ~0.16 — far below 0.5, and below Tallying's prediction in every corner of either parameter box. So the two theories produce visibly different numbers (Tallying > 0.5 always, SCWI ~ 0.16) with the sign of the gap guaranteed, while per-subject variance stays tight (48 Bernoulli trials per subject, p near 0 or 1 under both theories for typical parameters, giving SD ~ 0.04-0.07). This makes Welch's t-test at N=25 overwhelmingly significant. The metric is also a genuine behavioral signature rather than a tautology: it asks whether human subjects side with the count-based winner precisely when a weighted integrator would strongly oppose it — the single cleanest dissociation the design provides (e.g., the pair-3 vs pair-9 minimal contrast: changing cue 2 from a B-win to a tie barely moves Tallying but collapses SCWI's preference).

Source:
def metric(data: pd.DataFrame) -> float:
    # Conflict-set tally-following rate.
    #
    # Selection rule: keep only trials where (i) the tally is non-zero (one
    # option strictly wins more feature-wise comparisons) and (ii) a
    # validity-weighted signed integrator with anti-validity weights
    # w_j = v_j + tau * v_j^2 (tau = 0.35, the midpoint of SCWI's range)
    # STRONGLY prefers the tally LOSER (|evidence| > 0.8).
    # In this design that selects exactly Probe B (pairs 7-12, |D| in
    # [1.0, 1.3]) plus the anchors (pairs 13-14, |D| ~ 5.2), i.e. 8 of the
    # 16 unique pairs = 48 of 96 trials per subject.
    #
    # Metric value = proportion of choices that follow the TALLY WINNER on
    # those conflict trials.
    #   Tallying: P >= sigmoid(beta*|m|) with |m| in {1,3,6}  -> > 0.5 for
    #             ANY beta > 0, epsilon < 1; ~0.95+ for typical beta.
    #   SCWI:     P = (1-eps)*(1-sigmoid(beta*|D|)) + eps/2 ~ 0.16-0.20 on
    #             Probe B and ~ eps/2 ~ 0.075 on anchors -> far below 0.5.
    # The two theories are separated by ~0.4-0.8 in mean with tight
    # per-subject binomial variance (48 trials/subject).

    import numpy as np

    VALIDITIES = np.array([1.0, 0.95, 0.65, 0.5, 0.5, 0.5])
    TAU = 0.35      # midpoint of SCWI's tau box [0.25, 0.45]
    THRESH = 0.8    # cleanly above Probe A's |D| <= 0.08, below Probe B's |D| >= 1.0

    w = VALIDITIES + TAU * np.square(VALIDITIES)  # positive anti-validity magnitudes

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    n_sel = 0
    n_match = 0
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.size == 0 or a.size != b.size or a.size != w.size:
            continue

        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            # Full-tie trials (pairs 15/16): both theories say 0.5, uninformative.
            continue

        # Signed evidence for option A under a weighted anti-validity
        # integrator: value_A - value_B = sum_j w_j * (b_j - a_j).
        ev = float(np.dot(w, b - a))

        tally_a = a_wins > b_wins
        if tally_a:
            # A is the tally winner; keep only if the integrator strongly
            # prefers B (the tally loser).
            if ev > -THRESH:
                continue
        else:
            # B is the tally winner; keep only if the integrator strongly
            # prefers A (the tally loser).
            if ev < THRESH:
                continue

        r = int(round(float(resp)))  # 0 = chose A, 1 = chose B
        chose_tally_winner = (r == 0) if tally_a else (r == 1)
        n_sel += 1
        if chose_tally_winner:
            n_match += 1

    if n_sel == 0:
        return 0.5  # degenerate fallback (no conflict trials found)
    return float(n_match) / float(n_sel)

### RESULTS
- Predicted under pi_3 (simulated): 0.1608 (var=0.0029)
- Predicted under pi_2 (simulated): 0.8538 (var=0.0090)
- Observed on real data: 0.1625 (var=0.0119)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Slope of P(choose the top-cue-favored option) regressed on the
    # tally margin of that favored option, across all trials where the
    # highest-validity discriminating cue exists (ties like trials 17/18
    # are skipped). Under TTB this probability is flat in the tally margin
    # (slope ~ 0); under Tallying it rises monotonically with the tally
    # margin (positive slope).
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.65])
    cue_order = np.argsort(-validities, kind="stable")

    xs = []
    ys = []
    ra_col = data["option_a_ratings"]
    rb_col = data["option_b_ratings"]
    resp_col = data["response"]

    for ra, rb, resp in zip(ra_col, rb_col, resp_col):
        a = np.asarray(ra, dtype=float)
        b = np.asarray(rb, dtype=float)

        # Find the first discriminating cue in descending-validity order.
        favored = None
        for j in cue_order:
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break
        if favored is None:
            # Full tie: no top cue, no tally signal. Skip.
            continue

        # Tally margin from A's perspective, then re-sign so it is from
        # the top-cue-favored option's perspective.
        d = float(np.sum(a > b) - np.sum(b > a))
        if favored == 1:
            d = -d

        xs.append(d)
        ys.append(1.0 if int(resp) == favored else 0.0)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0

    xm = xs.mean()
    ym = ys.mean()
    sxx = float(np.sum((xs - xm) ** 2))
    if sxx <= 0.0:
        return 0.0

    slope = float(np.sum((xs - xm) * (ys - ym)) / sxx)
    return slope
```

**Observed (real) value:** -0.1747 (var=0.0108)
**Predicted under pi_3:** -0.1877 (var=0.0014)
**Predicted under pi_2:** 0.1952 (var=0.0019)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Cue cascade order for validities [0.9, 0.8, 0.7, 0.6, 0.5]:
    # strictly descending, so TTB consults features 0,1,2,3,4 in order.
    cue_order = [0, 1, 2, 3, 4]

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(v) for v in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    follow_disagree = 0
    n_disagree = 0
    follow_agree = 0
    n_agree = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != 5 or b.size != 5:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying must be decisive and the tally margin must be exactly 1:
        # on every such trial the softmax probability of following the tally
        # is sigmoid(beta * 1), identical across trials, so Tallying predicts
        # a perfectly FLAT follow-rate regardless of subset.
        if a_wins == b_wins:
            continue
        if abs(a_wins - b_wins) != 1:
            continue

        tally_picks_a = a_wins > b_wins

        ttb_picks_a = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_picks_a = True
                break
            if b[j] > a[j]:
                ttb_picks_a = False
                break
        if ttb_picks_a is None:
            continue

        chose_a = (float(resp) == 0.0)

        if ttb_picks_a == tally_picks_a:
            n_agree += 1
            if chose_a == tally_picks_a:
                follow_agree += 1
        else:
            n_disagree += 1
            if chose_a == tally_picks_a:
                follow_disagree += 1

    if n_disagree == 0 or n_agree == 0:
        return 0.0

    p_dis = follow_disagree / float(n_disagree)
    p_agr = follow_agree / float(n_agree)
    return p_dis - p_agr

```

**Observed (real) value:** 0.2737 (var=0.0930)
**Predicted under pi_3:** 0.1862 (var=0.0095)
**Predicted under pi_2:** -0.0006 (var=0.0060)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the evidence is strikingly one-sided in favor of THEORY 1 (pi_3, SCWI). Experiment 1 (the anti-validity slope metric): observed 0.1227 vs SCWI 0.1378 and Tallying -0.1263 \u2014 SCWI is essentially on top of the observed value, Tallying has the wrong sign. Experiment 2 (conflict-set tally-following rate, proposed by Tallying itself as a kill shot against SCWI): observed 0.1625 vs SCWI 0.1608 and Tallying 0.8538 \u2014 the observed value lands almost exactly on SCWI's prediction and catastrophically far from Tallying's; this is especially damaging because the experiment was engineered by the losing theory to be maximally adversarial to SCWI, and the real data sided with SCWI anyway. Experiment 3 (top-cue-favored slope): observed -0.1747 vs SCWI -0.1877 and Tallying +0.1952 \u2014 again SCWI matches in sign and magnitude, Tallying is sign-flipped. Experiment 4 (disagree-minus-agree follow rate): observed 0.2737 vs SCWI 0.1862 and Tallying -0.0006 \u2014 SCWI gets the sign and rough magnitude right (though it under-predicts by ~0.09, and the between-subject variance is large, var=0.093), while Tallying predicts approximately zero. In short, SCWI correctly predicts the qualitative signature (subjects avoid highly-rated options, with avoidance strength growing with instructed validity) in every experiment in the domain, while Tallying (pi_2) gets the sign of the key metric wrong in Experiments 1, 2, and 3 and predicts a null in Experiment 4. Tallying is not merely losing; it is structurally falsified: its core commitment (choice probability tracks the raw count margin and is pinned at exactly 0.5 on count-tied trials) is directly contradicted by the observed anti-validity gradients. A theory this degenerate cannot be rescued by re-parameterizing its predict function \u2014 its qualitative mechanism is wrong. The right move is therefore to replace pi_2 with a brand-new theory that preserves nothing of the count-margin commitment but competes seriously with SCWI. Notably, SCWI is not perfect: in Experiment 4 it under-predicts the observed 0.2737 (predicted 0.1862), and its fixed supralinear form w_j = -(v_j + tau*v_j^2) with tau in [0.25, 0.45] may be too rigid. A stronger competitor should generalize SCWI's anti-validity integration with (a) a free exponent/power-law mapping from validity to anti-weight (rather than the fixed quadratic form), and (b) explicit heterogeneity in inversion strength across subjects, which would also accommodate the large between-subject variance seen in Experiment 4.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying (pi_2) with a new theory: 'Power-Law Anti-Validity Integration with Heterogeneous Inversion' (PAHI). Core claims: (1) Subjects integrate all cues additively into a per-option subjective value, but interpret a rating of 1 as evidence AGAINST the option (comprehension inversion), exactly as SCWI holds. (2) The magnitude of the anti-weight for cue j is a free power-law of the instructed validity, |w_j| = v_j^gamma, with gamma a free parameter (gamma=1 recovers linear anti-validity; gamma>1 supralinear; gamma<1 compresses the validity gradient). This replaces SCWI's rigid fixed quadratic form w_j = -(v_j + tau*v_j^2) with a more flexible one-parameter family, allowing steeper validity gradients where SCWI under-predicted (Experiment 4's observed 0.2737 vs SCWI's 0.1862 is consistent with a steeper-than-quadratic weighting, i.e., larger gamma, which widens the value gap on TTB/tally-agreement trials and drives p_agree lower). (3) Heterogeneity: each subject draws gamma from a distribution (e.g., gamma in [1.0, 3.0]) and an inversion sign s in {-1, +1} with P(s=-1) high (e.g., 0.85), capturing occasional pro-validity subjects and the large between-subject variance (var=0.093) seen in Experiment 4. (4) Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon, exactly as in SCWI. This theory differs from both incumbents: unlike Tallying it uses validity-weighted (inverted) evidence rather than raw counts, so it correctly predicts the sign of all four observed metrics; unlike SCWI it has a flexible power-law weight map and subject-level heterogeneity, so it can match Experiment 4's more extreme observed dissociation while remaining fully consistent with Experiments 1-3 (where gamma in [1.5, 2.5] reproduces SCWI's near-perfect fits). Suggested parameter ranges: beta in [0.8, 2.0], gamma in [1.0, 3.0], epsilon in [0.05, 0.20], P(inverted) in [0.75, 0.95]."
}
```

## Usage

```json
{
  "input_tokens": 11975,
  "output_tokens": 2141,
  "reasoning_tokens": 951,
  "provider": "Sail Research"
}
```
