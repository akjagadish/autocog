# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_4
Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI). Subjects integrate all cues additively into a per-option subjective value, but a rating of 1 is treated as evidence AGAINST the option (comprehension inversion, s = -1 for ~99% of subjects). The anti-weight of cue j is a free power law of the instructed validity, |w_j| = v_j^gamma, with gamma a free per-subject parameter drawn from a heterogeneous population distribution (gamma ~ 1.4-2.4, mildly supralinear on average). Choice is a softmax over beta * s * sum_j v_j^gamma * rating_j with a symmetric lapse epsilon. Because gamma, beta, and epsilon are drawn broadly per subject, the population reproduces both the mean-level signatures (negative Exp-1 slope, positive Exp-2 dissociation, positive Exp-3 slope, low Exp-4 conflict-following) and the large between-subject variance seen in the real data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Power-Law Anti-Validity Integration with Heterogeneous Inversion (PAHI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = s * v_j**gamma per unit
    # rating, where s = -1 for the (vast majority of) comprehension-inverted
    # subjects and +1 for the rare pro-validity subject; gamma is a free
    # power-law exponent applied to the instructed validity. Option values
    # are the cue-weighted sums; choice is a numerically stable softmax over
    # beta * values, mixed with a symmetric lapse epsilon. History is
    # ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PAHI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s_txt = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s_txt.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    sign_seed = float(parameters["sign_seed"])

    # Heterogeneous comprehension inversion: with population probability
    # 0.99 a subject inverts the rating direction (a 1 counts AGAINST the
    # option); the remaining subjects read ratings pro-validity.
    s = -1.0 if sign_seed < 0.99 else 1.0

    # Power-law anti-validity cue weights: the anti-weight of expert j is
    # v_j**gamma (on average mildly supralinear in the instructed validity).
    w = s * np.power(val, gamma)

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

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=12):** [1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. SCWI and PAHI are structurally near-identical: both are anti-validity additive integrators (a rating of 1 counts AGAINST an option, tied cues cancel exactly, choice is a softmax plus symmetric lapse). The ONLY substantive difference is the validity-to-weight mapping: SCWI uses w(v) = v + tau*v^2 with tau in [0.25, 0.45]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Two mathematical facts drive the whole design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-elasticity is (1+2*tau*v)/(1+tau*v) <= 1.9/1.45 = 1.31 everywhere on v in [0.5, 1], while PAHI's elasticity is gamma >= 1.4. Therefore for ANY two validities, PAHI's relative weight w(v_hi)/w(v_lo) strictly exceeds SCWI's: PAHI is a uniformly steeper validity gradient. (2) LEVEL CROSSING: at v = 1.0 PAHI's weight is EXACTLY 1 (gamma-invariant) versus SCWI's 1.25-1.45, while at v = 0.5 SCWI's weight is 0.56-0.61 versus PAHI's 0.19-0.38. Consequence: on 'concentration' trials pitting a few high-validity endorsements against many low-validity endorsements, the preference flips at a knife-edge count ratio that differs between the theories, and integer ratios can be placed in the gap between the two knife-edges. THE KILL ZONE: with validities exactly 1.00 (two cues) and 0.50 (eight cues), let d = (number of 0.5-cue 1s on the low-cue option) - (number of 0.5-cue 1s on the top-cue option). SCWI prefers the top-cue option H iff d > 2(1+tau)/(0.5+0.25*tau), i.e., d in (4.44, 4.74) over the whole tau box -> flips strictly between d=4 and d=5. PAHI prefers H iff d > 2/0.5^gamma = 2^(gamma+1) in (5.28, 10.55) -> flips at d=6 or LATER, never before 5.28. So at d=5 the two theories make OPPOSITE predictions for every admissible parameter value: SCWI P(H) in [0.54, 0.62] (its evidence is 0.5-0.75*tau > 0 since tau < 2/3), PAHI P(H) in [0.04, 0.46] (its evidence is 5*0.5^gamma - 2 < 0 since gamma > log2(2.5) = 1.32). Note v = 1.0 exactly is essential: with v = 0.99, SCWI's flip condition tau < 0.389 fails inside its own box, destroying the parameter-free separation. PRIMARY INSTRUMENT (pairs 1-12): a d-sweep from 3 to 8, fully mirrored. SCWI predicts a psychometric curve crossing 0.5 strictly between d=4 (evidence = -tau < 0 always) and d=5, then saturating (P(H) ~ 0.67/0.77/0.84 at d=6/7/8; at d=8 its evidence is exactly +2.0 for ALL tau - a parameter-free anchor). PAHI predicts a crossing at d* = 2^(gamma+1) in (5, 10.6): at or beyond d=6 for gamma >= 1.59, and for gamma > 2 the curve never exceeds 0.5 within the observable range d <= 8. Three parameter-free statistics separate them: (i) the SIGN of P(H) at d=5 (16 trials per subject: pairs 3, 9, 13, 14 x 4 reps); (ii) the CROSSING LOCATION (SCWI: 4-5 for every subject, since the tau range is too narrow to move it; PAHI: >= 5.3, heterogeneous across subjects - SCWI predicts essentially zero between-subject variance in crossing point, PAHI predicts large variance); (iii) the saturation level at d=8 (SCWI >= 0.84; PAHI can sit below 0.5). SECONDARY KILLS at other validity mixtures, so the dissociation is not an artifact of the 1.0/0.5 pair alone: pair 15/16 (G): {1.0, 1.0, 0.85} vs seven 0.5-cues - SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option, PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.14, 1.34] favors the low option; P(H) in [0.55, 0.66] vs [0.10, 0.46]. Pair 17/18 (S): two 0.85-cues vs four 0.5-cues - SCWI margin 0.3-0.445*tau > 0 always (tau < 0.674) favors the 0.85 option, PAHI margin 0.5^gamma*(1.7^gamma - 2) > 0 always (gamma > 1.305) favors the 0.5 option. Pair 19/20 (M2): two 1.0-cues vs {0.85, 0.85, 0.5} - SCWI margin 0.2-0.305*tau > 0 always (tau < 0.656), PAHI margin 2-2*0.85^gamma-0.5^gamma in [0.03, 0.48] > 0 always. GAMMA-LADDER (pairs 21/22 and d=6/7/8): pair W ({1.0, 0.85} vs five 0.5-cues) splits PAHI at gamma ~= 1.47; d=6 splits at 1.59; d=7 at 1.81; d=8 at 2.0. SCWI predicts uniformly high P(H) on all of W/d6/d7/d8 (>= 0.55/0.67/0.77/0.84). If PAHI is correct, the observed flips across this ladder must recover a single consistent gamma AND that gamma must place the d=5 choice below 0.5; if SCWI is correct, all ladder points sit high while d=5 sits above 0.5. ROBUSTNESS VARIANT (pairs 13/14): d=5 realized with an extra 1 on BOTH options (3 vs 6 total 1s instead of 2 vs 5) - identical difference, different totals; both theories predict behavior identical to pair 3, ruling out raw count-of-1s heuristics. CONTROLS: pair 23 (all-1 vs all-0) is a maximum-separation anti-validity anchor where both theories predict P(all-1 option) ~= epsilon/2, calibrating the lapse; pair 24 (identical mixed vectors) is a full tie where both predict exactly 0.500, calibrating side/response bias. All critical pairs are mirrored to absorb side bias. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. DECISION RULES: (i) If the d=5 composite H-choice rate exceeds 0.5 (with d=4 below and d=6 above), SCWI is supported and PAHI is falsified - PAHI cannot exceed 0.463 there for any (beta, gamma, epsilon) in its box. (ii) If the d=5 rate is below 0.5, or the psychometric crossing sits at d >= 6 (or never), PAHI is supported and SCWI is falsified - SCWI cannot cross later than 4.74 low-cue units. (iii) If the crossing point varies widely across subjects (some never crossing by d=8), that specifically favors PAHI's heterogeneous gamma over SCWI's narrow tau. (iv) If the secondary kills G/S/M2 flip sign relative to the primary kill (e.g., high option preferred at 1.0/0.5 but not at 0.85/0.5), BOTH pure forms are falsified, since each theory's weight-shape prediction is sign-consistent across all these mixtures by construction.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
METRIC: the choice rate of SCWI's preferred option on the subset of trials where the two theories make OPPOSITE predictions for EVERY admissible parameter value (a parameter-free sign separation). The selection is computed inside the metric from the ratings and the known validities: a trial is kept iff (i) SCWI's evidence difference has a fixed sign with margin >= 0.08 at BOTH extremes of its tau box [0.25, 0.45], (ii) PAHI's evidence difference has a fixed sign with margin >= 0.05 at BOTH extremes of its gamma box [1.4, 2.4], and (iii) the two guaranteed signs are OPPOSED. This rule provably selects exactly the 8 designed kill pairs x 4 reps = 32 trials per subject: the d=5 trials (pairs 3, 9 and the robustness variants 13, 14), the G trials (15, 16: {1.0,1.0,0.85} vs seven 0.5-cues), and the S trials (17, 18: two 0.85-cues vs four 0.5-cues). WHY THESE TRIALS: (a) d=5: SCWI's evidence for the top-cue option is 0.5-0.75*tau in [0.16, 0.31] > 0 for all tau < 2/3, while PAHI's is 5*0.5^gamma - 2 in [-1.05, -0.11] < 0 for all gamma > log2(2.5) = 1.32 — SCWI P(H) in [0.54, 0.61], PAHI P(H) <= 0.47. (b) G: SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option; PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.15, 1.35] favors the low option. (c) S: SCWI margin 0.3-0.445*tau in [0.10, 0.19] favors the 0.85 option; PAHI margin 2*0.85^gamma-4*0.5^gamma in [0.08, 0.60] favors the 0.5 option. All other trials are excluded precisely because they cannot separate the theories for all parameter values: d=3,4 (both prefer the low-cue option), d=6,7,8 and W (PAHI's preference flips inside its gamma box, so no guaranteed sign), M2 (margins too thin), and the controls (ties / both agree). EXPECTED VALUES: under SCWI the metric is a weighted average of per-type probabilities all in [0.52, 0.60] -> population mean ~0.57, and because SCWI's parameter box is narrow the per-subject true rate varies only over ~[0.55, 0.58]; the dominant per-subject noise is binomial (32 trials, sd ~ 0.09), giving between-subject variance ~0.008. Under PAHI every selected trial has P(H) < 0.5 for every (beta, gamma, epsilon), so the population mean is ~0.20-0.25; its wider box produces per-subject spread roughly [0.05, 0.47] (var ~0.017) plus binomial noise (~0.005), total ~0.022. Welch's t-test at N=25 on (0.57, 0.008) vs (0.21, 0.022) gives t ~ 10 (p << 0.001): a large mean gap that is NOT drowned by per-subject variance, because the metric deliberately (i) uses only guaranteed-sign trials (so PAHI's heterogeneity cannot push any subject above SCWI's range on the selected set) and (ii) pools 32 trials per subject to shrink binomial noise. On real human data, if SCWI is correct the metric should land near 0.55-0.60; PAHI predicts at most ~0.47 and realistically ~0.2, so the metric cleanly adjudicates between them.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Instructed validities of this experiment (fixed at design time).
    VAL = np.array([1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    n_f = VAL.size

    # Extreme points of each theory's ADMISSIBLE parameter box.
    # SCWI: w(v) = -(v + tau*v^2), tau in [0.25, 0.45].
    W_SC = [-(VAL + t * VAL ** 2) for t in (0.25, 0.45)]
    # PAHI: w(v) = -v^gamma, gamma in [1.4, 2.4].
    W_PA = [-(VAL ** g) for g in (1.4, 2.4)]

    # Minimum |evidence margin| required to call a preference "guaranteed"
    # across the whole parameter box of a theory.
    M_SC = 0.08
    M_PA = 0.05

    def parse_vec(cell):
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks], dtype=float)
        return np.asarray(cell, dtype=float).ravel()

    def parse_resp(r):
        if isinstance(r, str):
            rs = r.strip().upper()
            if rs == 'A':
                return 0
            if rs == 'B':
                return 1
            return int(float(rs))
        return int(round(float(r)))

    hits = []
    for a_cell, b_cell, r_cell in zip(data['option_a_ratings'],
                                       data['option_b_ratings'],
                                       data['response']):
        try:
            a = parse_vec(a_cell)
            b = parse_vec(b_cell)
            if a.size != n_f or b.size != n_f:
                continue
            d_sc = [float(np.dot(w, a) - np.dot(w, b)) for w in W_SC]
            d_pa = [float(np.dot(w, a) - np.dot(w, b)) for w in W_PA]
            resp = parse_resp(r_cell)
        except Exception:
            continue

        sc_lo, sc_hi = min(d_sc), max(d_sc)
        if sc_lo > M_SC:
            sc_pref = 1      # SCWI prefers A for EVERY admissible tau
        elif sc_hi < -M_SC:
            sc_pref = -1     # SCWI prefers B for EVERY admissible tau
        else:
            continue         # preference not guaranteed -> skip

        pa_lo, pa_hi = min(d_pa), max(d_pa)
        if pa_lo > M_PA:
            pa_pref = 1      # PAHI prefers A for EVERY admissible gamma
        elif pa_hi < -M_PA:
            pa_pref = -1     # PAHI prefers B for EVERY admissible gamma
        else:
            continue

        if sc_pref == pa_pref:
            continue         # theories agree -> no diagnostic content

        # Trial is a guaranteed sign-opposition trial: SCWI's preferred
        # option is exactly the one PAHI rejects (and vice versa).
        scwi_option_is_a = (sc_pref == 1)
        chose_scwi = (resp == 0) if scwi_option_is_a else (resp == 1)
        hits.append(1.0 if chose_scwi else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_3 (simulated): 0.5531 (var=0.0069)
- Predicted under pi_4 (simulated): 0.2281 (var=0.0125)
- Observed on real data: 0.6837 (var=0.0643)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=28):** [1.0, 1.0, 1.0, 1.0, 0.65, 0.65, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competitor) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel, choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY substantive difference is the validity-to-weight map: PAHI uses w(v)=v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v)=v+tau*v^2 with tau in [0.25, 0.45]. Three mathematical facts drive the design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-local elasticity is (1+2*tau*v)/(1+tau*v), which lies in [1.11, 1.31] for ALL v in [0.5, 1] and ALL tau in its box; PAHI's elasticity is exactly gamma >= 1.4. The elasticity ranges are DISJOINT, so for every v < 1, PAHI's top-to-low weight ratio w(1)/w(v) strictly exceeds SCWI's. (2) CROSSING-WINDOW DISJOINTNESS AT r=2: fix counts of 2 top cues (v=1.00, where PAHI's weight is exactly 1, gamma-invariant - a fixed 'currency') versus 4 low cues at validity v. The choice flips at v* solving 2*w(v) = w(1). SCWI: v* in [0.550, 0.576] across its ENTIRE parameter box - a 0.026-wide, effectively parameter-free window. PAHI: v* = 0.5^(1/gamma) in [0.610, 0.749]. The windows are disjoint, so each subject's measured flip validity classifies the theories with zero free parameters. (3) FLATNESS AT r=2.5: at counts 2:5, SCWI's flip validity v* in [0.449, 0.477] lies BELOW the floor 0.5, so SCWI must prefer the top-cue option for EVERY v in [0.5, 1] and every parameter - a parameter-free flat-H prediction. PAHI flips at 0.4^(1/gamma) in [0.520, 0.683] and must prefer the LOW option at v=0.50 for every gamma - parameter-free flat-L at the floor. Opposite pinned predictions. WHERE THIS SITS RELATIVE TO PRIOR DESIGNS: experiment [4] swept the COUNT RATIO at fixed validities (1.0 vs 0.5) and used a single knife-edge (d=5). This design is its TRANSPOSE: it fixes the count ratio and sweeps the LOW-CUE VALIDITY, making the statistic the CROSSING VALIDITY (its location, its between-subject spread, and its cross-ratio consistency), plus amplified stacked kills at validity mixtures (1.0/0.60/0.50 and 1.0/0.55) that [4] never touched. Only one cell (2x1.00 vs 5x0.50) coincides with [4]'s d=5 anchor, retained deliberately as a cross-experiment replication cell. INSTRUMENT: 28 experts - 4 at 1.00, 5 at 0.65, 5 at 0.60, 9 at 0.55, 5 at 0.50. Two ladders at r=2 (2:4) and r=2.5 (2:5) over v in {0.50, 0.55, 0.60, 0.65}, two stacked kills (4x1.00 vs 4x0.60+5x0.50; 4x1.00 vs 9x0.55, i.e. r=2.25 sitting inside the (2.11, 2.32) gap at v=0.55), all mirrored, plus tie / lapse / agreement controls. 24 unique pairs x K = 96//24 = 4 = 96 trials. KILL CELLS (parameter-free sign separation; SCWI evidence is linear in tau so endpoint checks suffice, PAHI evidence is monotone decreasing in gamma on every kill cell): R2-60 (2x1.00 vs 4x0.60): SCWI evidence +0.148..+0.26, P(top) in [0.535, 0.603]; PAHI evidence -0.044..-0.826, P(top) <= 0.487. This cell is the r=2 crossing detector: SCWI's crossing (<=0.576) is below 0.60, PAHI's (>=0.610) is above. R25-50: SCWI +0.163..+0.313, P >= 0.538; PAHI -0.106..-1.053, P <= 0.468. STK (4x1.00 vs 4x0.60+5x0.50): SCWI +0.311..+0.573, P in [0.572, 0.742]; PAHI -0.149..-1.879, P <= 0.456. STK2 (4x1.00 vs 9x0.55): SCWI +0.375..+0.630, P in [0.588, 0.733]; PAHI -0.103..-1.856, P <= 0.469. R2-55 is a REVERSE kill: SCWI sits exactly at its crossing (evidence -0.156..+0.002, P(top) in [0.464, 0.501], i.e. pinned at chance) while PAHI is confidently opposite (evidence -0.268..-1.047, P(top) <= 0.420). R25-55: SCWI P in [0.62, 0.73]; PAHI prefers the low option for 87% of its gamma range. GAMMA-MAPPING RUNGS: R2-65 (SCWI [0.61, 0.70]; PAHI splits at gamma=1.61, 79% prefer the low option), R25-60 (SCWI [0.70, 0.82]; PAHI splits at gamma=1.79, 61% low), R25-65 (SCWI [0.76, 0.88]; PAHI splits at gamma=2.13, 73% top). POPULATION SHAPE AND HETEROGENEITY SIGNATURES: plotting P(top-option) versus low validity at r=2, SCWI crosses 0.5 inside (0.55, 0.60) for EVERY subject with near-zero between-subject spread (the whole tau box moves v* by only 0.026), whereas PAHI crosses in (0.60, 0.75) with crossing = 0.5^(1/gamma) spread over 0.14 validity units - a fivefold variance dissociation in crossing location. At r=2.5, SCWI never crosses (flat above 0.5 down to v=0.50) while PAHI crosses inside (0.52, 0.68). Moreover PAHI's two crossings must be jointly consistent with a single gamma: v*_{2.5} = 0.8^(1/gamma) * v*_2, i.e. v*_{2.5} in [0.85, 0.91] x v*_2; SCWI has no observable r=2.5 crossing at all. CONTROLS: pair 21 (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories; pair 22 (all-1 vs all-0) calibrates the lapse (both predict P(all-1 option) ~ epsilon/2, confirming anti-validity processing); pairs 23/24 are confident agreement cells (e.g. one 0.50-cue 1 vs two 1.00-cue 1s) calibrating the shared softmax/lapse on the anti-validity side. DECISION RULES: (i) SCWI is supported, and PAHI falsified, if P(top) > 0.5 on all four kill cells (especially STK/STK2 >= 0.57, which PAHI cannot exceed at 0.456/0.469 for any admissible parameters), the r=2 crossing lies in (0.55, 0.60) with tight between-subject clustering, r=2.5 is flat-H all the way down to v=0.50, and R2-55 sits at ~0.50. (ii) PAHI is supported, and SCWI falsified, if P(top) < 0.5 on the kill cells (STK/STK2 <= 0.47, below SCWI's 0.572/0.588 floor), the r=2 crossing lies at or above 0.61 (low-option choices persist through v=0.60 and top-option choices emerge only at 0.65+), an r=2.5 crossing is observed inside (0.52, 0.68) and is consistent with the r=2 crossing through a single gamma, R2-55 is clearly below 0.5, and crossing locations vary widely across subjects. (iii) If subjects prefer options with MORE high-validity 1s (pro-validity integration), or crossings appear below 0.55, or the profile fits neither map, both pure forms are falsified.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
CORE CONTRAST. Both theories are anti-validity additive integrators differing only in the validity-to-weight map. The decisive quantity is the weight ratio between a perfect cue (v=1.00) and a low cue: SCWI's map w(v)=v+tau*v^2 has log-elasticity (1+2*tau*v)/(1+tau*v) in [1.11, 1.31], i.e. nearly linear, so w(1)/w(0.6) is only 1.81-1.90 and w(1)/w(0.5) only 2.04-2.58. PAHI's map w(v)=v^gamma gives w(1)/w(0.6) = (1/0.6)^gamma in [2.05, 3.14] and w(1)/w(0.5)=2^gamma in [2.64, 5.28]. CONSEQUENCE on the ladder cells: on every 2-perfect-cue vs 4/5-low-cue cell and every 4-perfect-cue vs 9-low-cue cell, the low option's total anti-weight EXCEEDS the top option's under SCWI (so SCWI picks the TOP option, since more anti-weight = worse) but is BELOW the top option's under PAHI (so PAHI picks the LOW option). Concretely: cell 2-vs-4x0.60: SCWI evidence +0.148..+0.26 (P(top) ~0.56), PAHI evidence -0.046..-0.728 (P(top) ~0.29); cell 2-vs-5x0.50: SCWI +0.16..+0.31 (P ~0.57), PAHI -0.105..-1.053 (P ~0.25); cell 4-vs-(4x0.60+5x0.50): SCWI +0.31..+0.57 (P ~0.63), PAHI -0.149..-1.78 (P ~0.20); cell 4-vs-9x0.50: SCWI -0.29..+0.06 (P ~0.47), PAHI -0.59..-2.29 (P ~0.12); cell 2-vs-4x0.65: SCWI +0.46..+0.52 (P ~0.65), PAHI mostly negative (P ~0.39). METRIC: the proportion of choices of the top-cue option (the option whose 1s sit exclusively on the four v=1.00 cues), pooled over exactly these five diagnostic cells (12 unique trials x 4 reps = 48 trials per subject). Predicted pooled value: PAHI ~0.25, SCWI ~0.58 - a ~0.33 gap. CELL SELECTION FOR VARIANCE CONTROL: the 2-vs-4x0.50 cell is excluded because BOTH theories prefer the low option there (no separation); the 2-vs-5x0.60 and 2-vs-5x0.65 cells are excluded because PAHI's preference flips inside its gamma range (gamma=1.63 and 2.13), so those cells inflate PAHI's between-subject variance without enlarging the mean gap. The retained cells are ones where PAHI prefers the low option for essentially its ENTIRE parameter box, so its per-subject values are one-sided (range roughly 0.05-0.43, variance ~0.017 including binomial noise over 48 trials), while SCWI's per-subject values are tightly pinned (0.55-0.64, variance ~0.005, dominated by binomial noise) because its whole tau box moves the evidence by only ~0.15. Welch t at N=25: t ~ 10-12, p << 0.001. The metric is a simple proportion, well-defined on any subject's slice and on the pooled data, uses no theory-internal parameters, and the trial classification is purely structural (which option carries the perfect-cue 1s), so it is robust to trial randomization and response coding.

Source:
def metric(data: pd.DataFrame) -> float:
    # "Top-cue adherence" on steepness-diagnostic ladder cells.
    # The design's validity vector: 4 cues at 1.00, 5 at 0.65, 5 at 0.60,
    # 14 at 0.50 (hardcoded from the experimental design).
    VALID = np.array([1.0] * 4 + [0.65] * 5 + [0.60] * 5 + [0.50] * 14)
    NF = VALID.size

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    # Excluded cells (signature = (top-cue count, sorted low-cue validities)):
    #  (2, 4x0.50): both theories prefer the low option -> no separation;
    #  (2, 5x0.60) and (2, 5x0.65): PAHI splits across its gamma range on
    #  these cells -> they add between-subject noise without adding gap.
    EXCLUDE = {
        (2, (0.5, 0.5, 0.5, 0.5)),
        (2, (0.6, 0.6, 0.6, 0.6, 0.6)),
        (2, (0.65, 0.65, 0.65, 0.65, 0.65)),
    }

    hits = []
    for a_raw, b_raw, resp in zip(
        data['option_a_ratings'], data['option_b_ratings'], data['response']
    ):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != NF or b.size != NF:
            continue
        ta = int(np.sum(a[:4] > 0.5))
        tb = int(np.sum(b[:4] > 0.5))
        # Exactly one option must carry 1s on the perfect-validity cues.
        if ta > 0 and tb > 0:
            continue
        if ta == 0 and tb == 0:
            continue
        if ta > 0:
            top_vec, low_vec, top_idx, tc = a, b, 0, ta
        else:
            top_vec, low_vec, top_idx, tc = b, a, 1, tb
        if tc not in (2, 4):
            continue
        # Top option must have 1s ONLY on perfect cues; low option none.
        if np.sum(top_vec[4:] > 0.5) > 0:
            continue
        if np.sum(low_vec[:4] > 0.5) > 0:
            continue
        low_mask = low_vec > 0.5
        if int(np.sum(low_mask)) < 4:
            continue
        sig = (tc, tuple(sorted(float(x) for x in np.round(VALID[low_mask], 2))))
        if sig in EXCLUDE:
            continue
        hits.append(1.0 if int(resp) == top_idx else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_3 (simulated): 0.5687 (var=0.0053)
- Predicted under pi_4 (simulated): 0.2267 (var=0.0083)
- Observed on real data: 0.6300 (var=0.0703)

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
**Predicted under pi_4:** -0.1883 (var=0.0047)

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
**Predicted under pi_4:** 0.2669 (var=0.0115)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.1227 (var=0.0008)
**Predicted under pi_3:** 0.1378 (var=0.0002)
**Predicted under pi_4:** 0.1448 (var=0.0005)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.1625 (var=0.0119)
**Predicted under pi_3:** 0.1608 (var=0.0029)
**Predicted under pi_4:** 0.1696 (var=0.0044)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, pi_3 (SCWI) is the clearly superior theory, and pi_4 (PAHI) is falsified on the two experiments specifically designed to adjudicate between them. Experiment 1 (pi_3's adversarial design) observed 0.6837 on the guaranteed sign-opposition cells \u2014 decisively on SCWI's side of 0.5 (predicted 0.5531) and categorically incompatible with PAHI (predicted 0.2281; PAHI's steep power-law cannot exceed ~0.46 there for ANY admissible gamma). Experiment 2 (pi_4's own adversarial design) backfired completely: the observed top-cue adherence was 0.6300, again on SCWI's side (predicted 0.5687) and far from PAHI's 0.2267. PAHI's core claim \u2014 a steep power-law validity-to-weight map with gamma in [1.4, 2.4] \u2014 is therefore refuted in both directions: subjects' effective weight gradient is much flatter (near-linear-plus-mild-quadratic), exactly as SCWI posits. On the shared experiments both theories perform similarly because they are structurally near-identical anti-validity integrators: Exp 3 (observed -0.1747 vs -0.1877/-0.1883, tie), Exp 5 (observed 0.1227 vs 0.1378/0.1448, slight edge pi_3), Exp 6 (observed 0.1625 vs 0.1608/0.1696, edge pi_3). PAHI's only relative win is Exp 4 (0.2669 vs observed 0.2737, beating pi_3's 0.1862), but a modest magnitude miss on one experiment cannot compensate for two catastrophic sign failures on the diagnostic experiments. The consistent picture: subjects ARE anti-validity additive integrators with a near-linear weight map (SCWI's qualitative claim is fully supported \u2014 the sign of every metric across all six experiments matches SCWI's prediction). However, SCWI's current parameterization is systematically miscalibrated in two ways: (1) it UNDERSHOOTS the extremity of behavior on diagnostic cells (0.553 vs 0.684 observed in Exp 1; 0.569 vs 0.630 in Exp 2), meaning subjects follow the anti-validity evidence more strongly than the current beta/epsilon ranges imply; and (2) it drastically underestimates between-subject variance (predicted var ~0.005-0.007 vs observed 0.064-0.070 on the diagnostic experiments), meaning the population is far more heterogeneous than SCWI's narrow tau box allows. Both deficiencies are parameterization problems, not structural ones \u2014 they can be fixed by regenerating the parameter ranges (higher/heterogeneous beta, lower epsilon, wider per-subject tau spread) without touching the theory's core mechanism. Since SCWI is the theory that explains ALL the data qualitatively, it should be kept and its model regenerated; PAHI should be discarded as falsified.",
  "verdict": "new_model",
  "target_theory_idx": 1,
  "recommendation": "Keep the SCWI theory description (anti-validity additive integration with w_j = -(v_j + tau*v_j^2), softmax plus lapse) but regenerate the parameterization to fix the two systematic miscalibrations. (1) STRENGTHEN THE CORE: raise beta from [1.1, 1.7] to a per-subject range of roughly [2.0, 6.0] and lower epsilon from [0.05, 0.15] to [0.02, 0.10]. This pushes the diagnostic-cell predictions up toward the observed 0.63-0.68 (d=5 evidence of 0.16-0.31 with beta~3-4 and epsilon~0.05 yields P~0.62-0.72) and also raises the Exp-4 dissociation toward the observed 0.274 (higher beta makes agree-trial anti-following more deterministic while margin-1 disagree trials stay near chance). (2) INTRODUCE REAL HETEROGENEITY: draw beta, epsilon, and tau independently per subject from broad population distributions (e.g., beta log-uniform on [1.5, 6.0], epsilon uniform [0.02, 0.14], tau uniform [0.15, 0.60] \u2014 keeping tau below the ~0.65 sign-consistency bound so the theory's guaranteed-sign structure is preserved). A mixture of strongly deterministic anti-validity subjects (driving the diagnostic cells to 0.7+) and noisier subjects (keeping the Exp-6 conflict-following rate near the observed 0.16 and Exp-3/5 slopes intact) naturally reproduces the large observed between-subject variance (~0.06-0.07) that the current narrow, homogeneous parameter box cannot. (3) Optionally add a small per-cue attention/validation jitter parameter (multiplicative noise on the subjective validities) as an additional variance source, which leaves the mean-level predictions unchanged but fattens the between-subject spread on near-threshold trials. Verify the regenerated model against all six experiments: it must simultaneously produce ~0.63-0.68 with var ~0.06 on Exp 1, ~0.60-0.65 with var ~0.07 on Exp 2, slope ~-0.17 to -0.19 on Exp 3, dissociation ~0.25-0.30 on Exp 4, slope ~0.12-0.14 on Exp 5, and ~0.16 on Exp 6."
}
```

## Usage

```json
{
  "input_tokens": 22507,
  "output_tokens": 4047,
  "reasoning_tokens": 2804,
  "provider": "Sail Research"
}
```
