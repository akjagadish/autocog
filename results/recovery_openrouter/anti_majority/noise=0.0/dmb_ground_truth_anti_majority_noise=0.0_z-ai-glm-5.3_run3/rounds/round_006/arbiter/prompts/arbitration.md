# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_8") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_8" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

## THEORY 2 — pi_8
Extended-Precision Anti-Endorsement Integration (EPAI-xp). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it — a 'fewer red flags' / rarity-implies-quality reading — so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, a raw inverse temperature beta, and a validity exponent gamma, combined as p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population (the 'xp' extension) is characterized by three refinements: (i) gamma follows a concave-then-heavy two-component mixture on [0, 2.4] — 55% uniform on [0, 1.2] and 45% uniform on [1.2, 2.4] — placing roughly 30-35% of subjects above gamma = 1.5 (more than RSD's 25%) while extending support past 2.0 to absorb the observation that steep-gamma subjects behave somewhat shallower than a hard uniform-[0,2] edge would predict; (ii) beta is log-uniform on [2.0, 7.0], raising the floor above RSD's 1.8 (no evidence of any subject below ~1.8) and extending the ceiling to 7 to steepen probe confidence; (iii) m in [0.78, 0.96] and epsilon in [0.03, 0.10] keep the mean effective attenuation A_eff = (1-eps)(2m-1) near 0.68-0.72, matching observed extremity anchors. The resulting subject family is 'steep normalized slope x moderate asymptote': essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) is far above the 9.5 threshold that separates raw-weight from normalized-weight integration, while asymptotic preference stays moderate. This yields falsifiable predictions distinct from RSD-vgp: a non-uniform gamma histogram with mass above 2.0, a beta floor at 2.0 rather than 1.8, per-subject flip-ladder inversion rates on F3/F4 of ~20-27%/6-8%, and deeper Exp 11-style deviations.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Extended-Precision Anti-Endorsement Integration (EPAI-xp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"EPAI-xp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component
    # concave-then-heavy mixture on [0, 2.4] ----
    # gamma_u ~ U[0, 1]. With probability 0.55 the subject draws gamma
    # uniformly from [0, 1.2] (near-flat validity weighting); with
    # probability 0.45 from [1.2, 2.4] (the steep-gating subpopulation).
    # This places ~30-35% of subjects above gamma = 1.5 while extending
    # support past 2.0 (the 'extended precision' regime).
    u = float(parameters["gamma_u"])
    if u < 0.55:
        gamma = 1.2 * (u / 0.55)
    else:
        gamma = 1.2 + 1.2 * ((u - 0.55) / 0.45)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 2.0, ln 7.0]  =>  beta
    # log-uniform over [2.0, 7.0]. Floor raised above RSD-vgp's 1.8
    # (no subject shows evidence of beta below ~1.8); ceiling extended
    # to 7.0 to steepen probe confidence. Calibrated against the RAW
    # evidence scale below so that every subject's effective normalized
    # slope beta * sum(v^gamma) stays far above the 9.5 threshold
    # (min ~ 2.0 * 7.4 ~ 15 on any plausible validity vector).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=12):** [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 18: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 20: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** STRUCTURAL ANALYSIS. RSD-vgp and EPAI-xp share the identical core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), raw (unnormalized) weights, anti-endorsement polarity. RSD-vgp's two-component gamma mixture is algebraically EXACTLY Uniform[0,2] (both components have density 0.5). EPAI-xp's gamma mixture has density 0.4583 on [0,1.2] and 0.375 on [1.2,2.4]. The theories are therefore the SAME model with different parameter SUPPORTS, and only three support regions can dissociate them: (i) gamma in (2.0, 2.4]: 15% of EPAI subjects, 0% of RSD; (ii) beta in [1.8, 2.0): 8.8% of RSD subjects (ln(2/1.8)/ln(6/1.8)), 0% of EPAI (the (6,7] region is behaviorally undetectable due to saturation -- acknowledged one-sidedness); (iii) A_eff = (1-eps)(2m-1): RSD support [0.427, 0.970] vs EPAI [0.504, 0.892] -- roughly 12% of RSD subjects fall below 0.504 and 8% above 0.892, while EPAI places exactly ZERO subjects outside its box. The design makes all three boundaries empirically decisive.

ARCHITECTURE (12 experts): three cues at validity 1.00 (power weight exactly 1 for EVERY gamma -- gamma-invariant unit cues) and nine near the validity floor: seven at 0.50, one at 0.52, one at 0.55. The unit trio enables cells with EXACT integer evidence E = -1, -2, -3 for every subject at every gamma; the near-0.5 block's weights halve approximately per unit of gamma, giving fine-grained, exactly computable flip points against the gamma-invariant weight-2 top pair. Unanimous total weight S(gamma) >= 4.77 for all gamma in [0, 2.4], so the unanimous cells saturate every subject of both theories (beta*|E| >= 8.6 even at RSD's lowest beta = 1.8).

DISSOCIATION 1 -- FLIP LADDER (gamma ceiling; headline). On flip cells, A is endorsed on the two 1.00 cues (endorsement mass exactly 2 for all gamma) plus a shared canceling endorsement on the third 1.00 cue; B is endorsed on a block of k low cues. E(gamma) = W_block(gamma) - 2, strictly decreasing, with a single sign change at the flip point g*: subjects with gamma < g* prefer A (the count-sparser, top-cue-endorsed option); gamma > g* prefer B. Flip points: F4 (k=4 at 0.50): g* = 1.000; F5 (k=5): 1.322; F6 (k=6): 1.585; F7 (k=7): 1.807; F8 (k=8: 7x0.50+0.52): 2.015; F9 (k=9: 7x0.50+0.52+0.55): 2.223. Predicted population inversion rates P(gamma > g*): RSD 0.500/0.339/0.207/0.097/0.000/0.000; EPAI 0.542/0.404/0.306/0.222/0.144/0.066. HARD CELLS: F8 and F9 have E(2.0) = +0.020 and +0.323 > 0, and E is strictly decreasing, so EVERY RSD subject (gamma <= 2) has E > 0 and therefore p(A) > 0.5 STRICTLY (m >= 0.74 > 0.5 keeps p_core > 0.5 whenever E > 0; lapse only pulls toward 0.5). No RSD subject can prefer B on F8/F9. EPAI-xp REQUIRES 14.4% of subjects below 0.5 on F8 (inversion depth down to p(A) ~ 0.26 at gamma = 2.4, beta = 3.7) and 6.6% on F9. Any subject with a credibly inverted preference on F8/F9 categorically falsifies RSD-vgp while remaining inside EPAI's box; conversely, a hard zero of inversions past g* = 2.0 where EPAI demands 14% falsifies EPAI. The full survival curve (inversion rate vs g*) is piecewise-linear hitting zero AT 2.0 (RSD) vs declining smoothly to zero at 2.4 (EPAI). Within-subject, inverting F9 implies inverting F8, F7, F6... (monotone in gamma). Corroboration: EPAI's gamma density drops 18% at 1.2 (0.4583 -> 0.375), predicting a visible kink in the per-subject gamma histogram; RSD's histogram is flat to its hard edge at 2.0.

DISSOCIATION 2 -- A_eff SUPPORT (m/epsilon boxes). The unanimous cells U/Um saturate every subject, reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002. RSD support for p_U: [0.714, 0.985]; EPAI hard box: [0.752, 0.946]. RSD places ~12% of subjects below 0.752 and ~8% above 0.946 (its m extends to 1.0 with epsilon down to 0.03, allowing near-deterministic subjects; EPAI's m <= 0.96 and epsilon >= 0.03 cap A_eff at 0.892). Any subject with credibly p_U < 0.752 or > 0.946 falsifies EPAI-xp categorically; RSD requires ~20% of subjects in these forbidden zones.

DISSOCIATION 3 -- BETA FLOOR (exact ratio ladder). B1/B2/B3 use only the unit cues with exactly canceling shared endorsements (a shared endorsement on the same-validity cue contributes w*(1-1) = 0), giving E = -1, -2, -3 EXACTLY for every gamma. The per-subject ratio r12 = (0.5 - p_B2)/(0.5 - p_B1) = (sigmoid(2*beta) - 0.5)/(sigmoid(beta) - 0.5) is A_eff-free and lapse-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8, 6]); EPAI: [1.003, 1.266] (beta in [2.0, 7.0]). The region r12 in (1.266, 1.322] (i.e., beta in [1.8, 2.0)) is RSD-EXCLUSIVE: 8.8% of RSD subjects, 0% of EPAI. The independent second ratio r13 = (0.5 - p_B3)/(0.5 - p_B1) has RSD-exclusive region (1.307, 1.384]. The upper-beta region (6, 7] produces only saturation-level differences in both ratios and is acknowledged as undetectable -- this weapon is one-sided.

CONTROLS. C/Cm: a 3-3 endorsement-count tie (A endorsed on the two 1.00 cues plus the shared cue; B on two 0.50 cues plus the shared cue): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.79) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2: identical vectors -- both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by target side (top-pair side for flip cells, non-endorsed side for B-cells, sparse side for U). All informative trials carry distinct, non-trivially comparable rating vectors on both bars (locked-prompt requirement).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject. Pooled mid-box differences are modest and honestly acknowledged (approximate pooled p(top-pair side): F7 ~0.77 RSD vs ~0.71 EPAI; F8 ~0.77 vs ~0.71; F9 ~0.80 vs ~0.78; F4 ~0.50 both; U ~0.845 both; B1 ~0.20 both) because the populations' central tendencies nearly coincide -- the arbiter is SUBJECT-LEVEL population structure. Requires N >= 60 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject with credibly p(A) < 0.5 on F8 or F9 categorically falsifies RSD-vgp (E(2.0) > 0 over its entire gamma-box), while EPAI requires 14.4%/6.6% of subjects there; (b) inversion-rate survival curve over g* = 1.00/1.32/1.59/1.81/2.02/2.22: RSD 0.50/0.34/0.21/0.10/0/0 vs EPAI 0.54/0.40/0.31/0.22/0.14/0.07 -- a hard zero past 2.0 (RSD) vs a smooth decline to 2.4 (EPAI); (c) per-subject gamma histogram: flat with hard edge at 2.0 (RSD) vs an 18% density drop at 1.2 and 15% of mass in (2.0, 2.4] (EPAI); (d) per-subject p_U outside [0.752, 0.946]: ~20% (RSD) vs 0% (EPAI); (e) per-subject r12 in (1.266, 1.322]: ~9% (RSD) vs 0% (EPAI); (f) within-subject monotone flip pattern (inverting F9 implies inverting F8..F4); (g) count-tie C preferring the low-validity side above 50% and ties at exactly 50% (shared controls).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The two theories share the EXACT same per-trial likelihood p(A) = 0.5 + A_eff*(sigmoid(beta*E_gamma) - 0.5) with E_gamma = sum_j v_j^gamma*(b_j - a_j) and A_eff = (1-eps)(2m-1); they differ ONLY in the population priors over (gamma, beta, A_eff). The statistically efficient discriminative statistic for 'which population generated these subjects' is therefore the log marginal likelihood ratio (log Bayes factor) between the two priors, which I compute by deterministic quadrature (96 gamma bins on [0,2.4], 12 log-spaced beta bins on [1.8,7], 28-point A_eff grids from each theory's m/epsilon uniforms, each with its own exact prior weights: RSD gamma~U[0,2], beta~logU[1.8,6]; EPAI gamma = 0.55*U[0,1.2]+0.45*U[1.2,2.4], beta~logU[2,7]). Per subject i the metric evaluates LLR_i = ln m_RSD(data_i) - ln m_EPAI(data_i); on the pooled frame it returns sum_i LLR_i, which is exactly the joint log Bayes factor of the whole dataset (subjects are independent), i.e. the total evidence the experiment carries about which population model generated it. This is a genuine two-sided model comparison, not a trivially-true statistic: if EPAI generated the data it goes strongly negative. WHY IT SEPARATES: EPAI places 15% of subjects at gamma in (2.0, 2.4], where RSD's U[0,2] prior has literally zero density; behaviorally those subjects prefer the endorsement-DENSE option on the F8/F9 flip cells (E > 0 for every gamma <= 2 on those cells), a pattern RSD's prior can only produce through binomial lapse noise, so their marginal likelihood under RSD collapses (per-subject LLR about -1 to -2.5), while under EPAI they are well explained. Conversely, for subjects whose gamma-hat lies inside [0,2] the prior density ratio f_RSD/f_EPAI is >= 0.5/0.458 > 1 everywhere, so typical RSD subjects each contribute a small positive LLR (+0.09 to +0.29). Expected per-subject LLR is roughly +0.15 under RSD-simulated data versus roughly -0.08 under EPAI-simulated data, with tight per-subject spread (variance approx 0.02 under RSD, approx 0.3 under EPAI, since only the 15% EPAI tail subjects produce large-magnitude values). Because the pooled statistic accumulates this per-subject evidence over all simulated subjects, the two simulation arms land far apart in units of the per-subject variability, passing the Welch test by a wide margin at N=25. The metric is well defined on any single subject's slice (it returns that subject's individual log Bayes factor), so the between-subject variance is always computable; it is deterministic (pure quadrature, no randomness), uses only the schema columns, and avoids hashing the list-valued rating columns (stimulus types are keyed by projected tuples).

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Joint log Bayes factor between the two population models.
    # Both theories share the SAME per-trial likelihood
    #   p(A) = 0.5 + A_eff * (sigmoid(beta * E_gamma) - 0.5),
    #   E_gamma = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1).
    # They differ ONLY in the population priors over (gamma, beta, A_eff):
    #   RSD-vgp: gamma ~ U[0,2]; beta ~ logU[1.8,6]; m~U[0.74,1], eps~U[0.03,0.11]
    #   EPAI-xp: gamma ~ 0.55*U[0,1.2] + 0.45*U[1.2,2.4]; beta ~ logU[2,7];
    #            m~U[0.78,0.96], eps~U[0.03,0.10]
    # The metric marginalizes the likelihood over each theory's prior by
    # deterministic quadrature and returns sum_i ln[ m_RSD(data_i)/m_EPAI(data_i) ]
    # over subjects (subjects are independent, so the joint log Bayes factor
    # is exactly the sum of the per-subject log Bayes factors).
    val = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    if data is None or len(data) == 0:
        return 0.0

    # ---------- map trials to unique stimulus types ----------
    a_lists = data['option_a_ratings'].tolist()
    b_lists = data['option_b_ratings'].tolist()
    keys = [(tuple(x), tuple(y)) for x, y in zip(a_lists, b_lists)]

    type_index = {}
    diff_rows = []
    for kk in keys:
        if kk not in type_index:
            type_index[kk] = len(diff_rows)
            diff_rows.append(np.asarray(kk[1], dtype=float) - np.asarray(kk[0], dtype=float))
    D = np.vstack(diff_rows)                      # [T, F]  b - a per type
    T = D.shape[0]
    tidx = np.array([type_index[kk] for kk in keys], dtype=int)

    resp_ser = pd.to_numeric(pd.Series(data['response'].values), errors='coerce')
    resp = resp_ser.to_numpy(dtype=float)
    if np.isnan(resp).any():
        vals = [str(r).strip().upper() for r in data['response'].values]
        resp = np.array([0.0 if v in ('0', 'A') else 1.0 for v in vals])

    uq, sidx = np.unique(np.asarray(data['subject_id'].values), return_inverse=True)
    n_sub = int(len(uq))

    K = np.zeros((n_sub, T))                     # A-choice counts per type
    N = np.zeros((n_sub, T))                     # total trials per type
    isA = (resp == 0).astype(float)
    np.add.at(K, (sidx, tidx), isA)
    np.add.at(N, (sidx, tidx), 1.0)

    # ---------- gamma quadrature: 96 midpoint bins on [0, 2.4] ----------
    ge = np.linspace(0.0, 2.4, 97)
    gm = 0.5 * (ge[:-1] + ge[1:])
    dg = float(ge[1] - ge[0])
    wR_g = np.where(gm <= 2.0, 0.5, 0.0) * dg                 # RSD: U[0,2]
    wE_g = np.where(gm <= 1.2, 0.55 / 1.2, 0.45 / 1.2) * dg    # EPAI mixture
    wR_g = wR_g / wR_g.sum()
    wE_g = wE_g / wE_g.sum()
    G = len(gm)

    # ---------- beta quadrature: 12 midpoint bins in log space ----------
    le = np.linspace(np.log(1.8), np.log(7.0), 13)
    bm = 0.5 * (le[:-1] + le[1:])
    beta = np.exp(bm)

    def _log_overlap(lo, hi):
        lo_c = np.clip(le[:-1], lo, hi)
        hi_c = np.clip(le[1:], lo, hi)
        w = np.maximum(hi_c - lo_c, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    wR_b = _log_overlap(np.log(1.8), np.log(6.0))   # logU[1.8, 6]
    wE_b = _log_overlap(np.log(2.0), np.log(7.0))   # logU[2, 7]
    B = len(beta)

    # ---------- evidence and sigmoid - 0.5 ----------
    Wg = np.power(val[None, :], gm[:, None])         # [G, F] v^gamma
    Ev = Wg @ D.T                                    # [G, T]
    x = beta[:, None, None] * Ev[None, :, :]         # [B, G, T]
    x = np.clip(x, -60.0, 60.0)
    u = 1.0 / (1.0 + np.exp(-x)) - 0.5               # sigmoid(beta*E) - 0.5
    C = B * G
    u_flat = u.reshape(C, T)                         # c = b*G + g

    w_cR = (wR_b[:, None] * wR_g[None, :]).ravel()
    w_cE = (wE_b[:, None] * wE_g[None, :]).ravel()

    # ---------- A_eff quadrature from each theory's (m, eps) uniforms ----------
    def _a_grid(m_lo, m_hi, e_lo, e_hi):
        mm = m_lo + (m_hi - m_lo) * (np.arange(7) + 0.5) / 7.0
        ee = e_lo + (e_hi - e_lo) * (np.arange(4) + 0.5) / 4.0
        A = (2.0 * mm - 1.0)[:, None] * (1.0 - ee)[None, :]
        return A.ravel()

    def _prep(A_vals, w_c):
        pu = 0.5 + A_vals[:, None, None] * u_flat[None, :, :]   # p(A)
        D1 = np.log(pu) - np.log1p(-pu)                        # log p - log(1-p)
        L0 = np.log1p(-pu)                                     # log(1-p)
        nA = A_vals.size
        D1m = D1.reshape(nA * C, T)
        L0m = L0.reshape(nA * C, T)
        logw = (np.full(nA, -np.log(nA))[:, None] + np.log(w_c)[None, :]).ravel()
        return D1m, L0m, logw

    A_R = _a_grid(0.74, 1.00, 0.03, 0.11)
    A_E = _a_grid(0.78, 0.96, 0.03, 0.10)
    D1R, L0R, lwR = _prep(A_R, w_cR)
    D1E, L0E, lwE = _prep(A_E, w_cE)

    def _logsumexp(z):
        m = np.max(z)
        return float(m + np.log(np.sum(np.exp(z - m))))

    total = 0.0
    for i in range(n_sub):
        # log likelihood of subject i's counts at every grid point:
        # sum_t [ K_t*log p_t + (N_t-K_t)*log(1-p_t) ]
        lr = _logsumexp(D1R @ K[i] + L0R @ N[i] + lwR)
        le_i = _logsumexp(D1E @ K[i] + L0E @ N[i] + lwE)
        total += lr - le_i
    return float(total)

### RESULTS
- Predicted under pi_7 (simulated): 5.3165 (var=0.3027)
- Predicted under pi_8 (simulated): -30.9335 (var=3.3944)
- Observed on real data: 6.0286 (var=0.5984)

## EXPERIMENT 2 (proposed by pi_8)

### DESIGN
**Validities (n_features=13):** [1.0, 1.0, 1.0, 0.74, 0.72, 0.68, 0.63, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 21: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  trial 24: A=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

**Rationale:** STRUCTURAL SETUP. EPAI-xp and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights. RSD-vgp's two-component gamma mixture is algebraically EXACTLY Uniform[0,2] (both components have density 0.5). EPAI-xp's mixture has density 0.4583 on [0,1.2] and 0.375 on [1.2,2.4] (mean gamma 1.14 vs RSD's 1.0). The theories are the SAME model with different parameter SUPPORTS; only three regions can dissociate them: (i) gamma in (2.0, 2.4] -- 15% of EPAI subjects, 0% of RSD; (ii) beta in [1.8, 2.0) -- 8.8% of RSD subjects (ln(2/1.8)/ln(6/1.8)), 0% of EPAI (EPAI's (6,7] beta region is saturation-blind and acknowledged as undetectable -- one-sided weapon); (iii) A_eff = (1-eps)(2m-1): RSD support [0.427, 0.970] (p_unanimous in [0.714, 0.985], ~12% below 0.752 and ~8% above 0.946) vs EPAI hard box [0.504, 0.892] (p_unanimous in [0.752, 0.946], ZERO mass outside). A fourth, previously undeployed lever: EPAI's CDF is pointwise BELOW RSD's on all of (0,2] (0.458t < 0.5t below 1.2; 0.55+0.375(t-1.2) < 0.5t above), so EPAI's gamma distribution STRICTLY STOCHASTICALLY DOMINATES: EPAI predicts a HIGHER inversion rate at EVERY flip point, plus an 18% density drop at the 1.2 kink.

ARCHITECTURE (13 experts): three cues at validity 1.00 (power weight exactly 1 for EVERY gamma -- gamma-invariant unit cues) and a spread singleton ladder 0.74/0.72/0.68/0.63/0.56/0.53 plus four 0.50 cues. Unanimous total weight S(gamma) >= 5.88 for all gamma in [0, 2.4], so the unanimous cells saturate every subject of both theories (beta*|E| >= 1.8*5.88 = 10.6), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002.

DISSOCIATION 1 -- ENGINEERED FLIP LADDER (headline; unlike prior mass-2-vs-uniform-block constructions, this uses a mass-1 unit cue against MIXED-VALIDITY PAIRS whose combined weight crosses exactly 1 at a designed flip point g*, giving a denser, precisely placed ladder). On each flip cell A is endorsed on one 1.00 cue (weight exactly 1 for all gamma) and B on a pair/block with W(g*) = 1; E(gamma) = W(gamma) - 1 is strictly decreasing with a single sign change at g*, so subjects with gamma < g* prefer the count-sparser unit-cue side and gamma > g* prefer the low-validity-denser side. Flip points: F1 {0.50,0.56} g*=1.09; F2 {0.50,0.63} g*=1.22; F3 {0.50,0.72} g*=1.43; F4 {0.63,0.72} g*=1.78; F5 {0.68,0.72} g*=1.95; F6 {0.50 x4} g*=2.00; F7 {0.72,0.74} g*=2.20. Predicted population inversion rates P(gamma > g*): RSD 0.455/0.385/0.285/0.110/0.025/0.000/0.000; EPAI 0.500/0.443/0.364/0.233/0.169/0.150/0.075. EPAI predicts MORE inverters at every rung (stochastic dominance, a one-sided ordering test across the whole ladder). HARD CELLS: on F6, E(gamma) = 4*0.5^gamma - 1 >= 0 for ALL gamma <= 2 (zero only at the measure-zero point gamma = 2), and on F7, E(2.0) = 0.72^2 + 0.74^2 - 1 = +0.066 > 0 with E strictly decreasing, so since m >= 0.74 > 0.5 makes p >= 0.5 whenever E >= 0, NO RSD subject can truly prefer the low-validity side on F6/F7 -- a hard floor at exactly 50%. EPAI-xp REQUIRES 15.0% of subjects below 0.5 on F6 (inversion depth: typical tail subject gamma~2.2 has E = -0.131, beta~3.7, p(unit side) ~ 0.42) and 7.5% on F7 (shallow, p ~ 0.47-0.49 -- acknowledged weak per subject, but population-detectable). Any subject credibly below 50% on F6/F7 categorically falsifies RSD-vgp; conversely a hard zero of inversions past g* = 2.0 where EPAI demands 15% (with N = 80, ~12 expected tail subjects) falsifies EPAI. Within-subject, inverting F6 implies inverting F5, F4, ... (monotone in gamma); per-subject gamma-hat from the 7 rungs (56 observations) traces the survival curve: RSD linear to a hard edge at 2.0 vs EPAI's smoother decline to 2.4 with 15% of gamma-hat mass in (2.0, 2.4]. The ladder also rules out pure counting: counting predicts the unit-cue (fewer-endorsement) side at ~100% on EVERY rung, flat, which neither theory produces.

DISSOCIATION 2 -- THE 1.2 KINK. F1 (1.09) and F2 (1.22) bracket EPAI's gamma-density drop (0.4583 -> 0.375 at 1.2, an 18% drop) while RSD's density is flat 0.5 throughout. The survival-curve slope (inversion-rate drop per unit flip-point interval) is 0.5 everywhere under RSD vs ~0.46 below 1.2 and ~0.375 above under EPAI: fit a piecewise-density survival curve with a break at 1.2. Honest caveat: the kink is a secondary, N-hungry signature (N >= 100 recommended); the tail cells F6/F7 and the dominance ordering are the primary weapons.

DISSOCIATION 3 -- BETA FLOOR VIA EXACT-E RATIO CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements (a shared endorsement on the same cue contributes w*(1-1) = 0), giving E = -1 and -2 EXACTLY for every subject at every gamma, with both bars nonempty (conflict-style). The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta) - 0.5)/(sigmoid(beta) - 0.5) is exactly A_eff-free and lapse-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8, 6]); EPAI: [1.003, 1.266] (beta in [2.0, 7.0]). The region r12 in (1.266, 1.322] (beta in [1.8, 2.0)) is RSD-EXCLUSIVE: 8.8% of RSD subjects, 0% of EPAI. EPAI's own exclusive beta region (6, 7] moves r12 only between 1.0025 and 1.005 -- below measurement resolution, honestly acknowledged as one-sided. Per-subject r12 from 8 observations per cell is noisy; the arbiter is hierarchical population fitting under each theory's beta prior.

DISSOCIATION 4 -- A_eff SUPPORT. The unanimous cells U/Um saturate every subject, reading out A_eff. RSD support for p_sparse on U: [0.714, 0.985] with ~12% of subjects below 0.752 and ~8% above 0.946 (m up to 1.0 with eps down to 0.03 permits near-deterministic subjects; m down to 0.74 permits weak ones); EPAI's hard box: [0.752, 0.946], exactly zero mass outside. Any subject with credibly p_U < 0.752 or > 0.946 falsifies EPAI-xp categorically; RSD requires ~20% of subjects in these forbidden zones. EPAI additionally predicts a NARROWER population spread of p_U (A_eff concentrated near 0.68-0.72) than RSD's wide continuum.

CONTROLS. C/Cm: a 2-2 endorsement-count tie (A endorsed on two 1.00 cues, B on two 0.50 cues): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.75-0.82) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2: identical non-trivial vectors -- both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (flip cells), sparse side (B-cells, U), or low-validity side (C). All informative trials carry distinct, non-trivially comparable rating vectors on both bars (locked-prompt requirement).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject. Pooled mid-box differences are modest and honestly acknowledged (approximate pooled p(unit-cue side): F1 ~0.51 RSD vs ~0.48 EPAI; F4 ~0.64 vs ~0.60; F6 ~0.73 vs ~0.69 -- the populations' central tendencies nearly coincide) because the decisive evidence is SUBJECT-LEVEL population structure. Requires N >= 80-100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on F6 or F7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at 0.5), while EPAI requires 15%/7.5% of subjects there; (b) inversion-rate survival curve over g* = 1.09/1.22/1.43/1.78/1.95/2.00/2.20: RSD 0.455/0.385/0.285/0.110/0.025/0/0 vs EPAI 0.500/0.443/0.364/0.233/0.169/0.150/0.075 -- EPAI higher at EVERY rung (stochastic dominance) with a hard zero past 2.0 under RSD; (c) per-subject gamma histogram: flat density 0.5 with hard edge at 2.0 (RSD) vs 18% density drop at 1.2 and 15% of mass in (2.0, 2.4] (EPAI); (d) per-subject p_U outside [0.752, 0.946]: ~20% (RSD) vs 0% (EPAI); (e) per-subject r12 in (1.266, 1.322]: ~9% (RSD) vs 0% (EPAI); (f) within-subject monotone flip pattern (inverting F6 implies inverting F5, F4, F3...); (g) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
ESTIMAND: the population fraction of subjects whose validity exponent gamma exceeds 2.0. The two theories are the SAME per-trial model (anti-endorsement, raw weights, p(A)=0.5+A_eff*(sigmoid(beta*E)-0.5), E=sum_j v_j^gamma*(b_j-a_j)) and differ ONLY in their population parameter supports. The single hardest structural difference is the gamma support: RSD-vgp's gamma is confined to [0,2] (its mixture is exactly Uniform[0,2]), so RSD predicts EXACTLY ZERO subjects with gamma>2; EPAI-xp places 45% of its mass on [1.2,2.4], i.e. 15% of subjects in (2.0,2.4]. The engineered flip ladder makes this observable: on rung F6 (one validity-1.00 cue vs four 0.50 cues) E(gamma)=4*0.5^gamma-1 >= 0 for every gamma <= 2, so no RSD subject can truly prefer the low-validity side, while every gamma>2 subject does; rungs F5/F7 and the B/C/U cells add graded gamma information. MEASUREMENT: for each subject I fit the SHARED core (which both theories accept per-trial, so nothing EPAI-specific is baked in) with diffuse priors over gamma in [0,2.6], A_eff in [0.35,0.95], beta in [1.8,7], using all 24 unique trial types (4 reps each). The per-subject output is the posterior probability P(gamma>2 | own 96 choices); the population estimate is the mean of these posteriors, smoothed by a weak empirical-Bayes prior (strength 15 subjects, mean 0.075). PREDICTIONS: under RSD-vgp the true tail mass is 0, so the estimate is pure estimation smear (~0.07-0.09); under EPAI-xp it is ~0.13-0.16 (15% genuine tail subjects plus smear). The gap (~0.05-0.08) is large relative to the between-subject spread of the smoothed statistic (~0.003 SD), because the shrinkage makes each subject's value a stable posterior mean rather than a noisy indicator. This is exactly the design's decisive signature (a): a hard zero of gamma>2 mass under RSD vs a required 15% under EPAI, read out with the model both theories share. The metric is well defined on the pooled DataFrame and on each single-subject slice (per-subject values lie in a tight band around the prior mean, with the subject's own tail posterior contributing), so between-subject variance is available and small, while the pooled scalar tracks the population tail mass that only EPAI-xp predicts to be positive.

Source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """Estimated population fraction of subjects whose validity exponent
    gamma lies above 2.0 (the hard support edge of RSD-vgp's gamma box),
    measured with the anti-endorsement raw-weight measurement model that
    BOTH theories share per-trial, and stabilized by a weak empirical-Bayes
    smoothing prior so the statistic is well-behaved on single subjects."""
    VAL = np.array([1.0, 1.0, 1.0, 0.74, 0.72, 0.68, 0.63,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    KAPPA = 15.0      # smoothing prior strength (in subjects)
    P0 = 0.075        # smoothing prior mean tail mass
    TAIL = 2.0        # gamma support edge of the competing theory

    if data is None or len(data) == 0:
        return P0

    # ---- tidy parse (list cells -> hashable tuples) ----
    ka = [tuple(int(v) for v in x) for x in data['option_a_ratings']]
    kb = [tuple(int(v) for v in x) for x in data['option_b_ratings']]
    resp = np.asarray(data['response'].values).astype(int)
    choseA = (resp == 0).astype(int)

    df = pd.DataFrame({'s': np.asarray(data['subject_id'].values),
                       'ka': ka, 'kb': kb, 'cA': choseA})
    cnt = df.groupby(['s', 'ka', 'kb']).agg(
        n=('cA', 'size'), k=('cA', 'sum')).reset_index()

    uniq = sorted(set(zip(cnt['ka'].values, cnt['kb'].values)))
    tidx = {t: i for i, t in enumerate(uniq)}
    subjects = list(pd.unique(cnt['s']))
    sidx = {s: i for i, s in enumerate(subjects)}
    S = len(subjects)
    T = len(uniq)
    if S == 0 or T == 0:
        return P0

    K = np.zeros((S, T))   # A-choices per (subject, trial type)
    N = np.zeros((S, T))   # trials per (subject, trial type)
    for row in cnt.itertuples(index=False):
        K[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.k)
        N[sidx[row.s], tidx[(row.ka, row.kb)]] = float(row.n)

    # ---- diffuse parameter grids (theory-neutral, covers both supports) ----
    gam = np.linspace(0.0, 2.6, 53)                       # validity exponent
    Agrid = np.linspace(0.35, 0.95, 7)                    # effective attenuation
    Bgrid = np.exp(np.linspace(np.log(1.8), np.log(7.0), 7))  # inverse temp
    Wg = VAL[None, :] ** gam[:, None]                     # (53, 13)

    # per-trial-type choice probabilities under the shared core:
    # p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j)
    LP = np.empty((T, 53, 7, 7))
    L1P = np.empty((T, 53, 7, 7))
    for t, (a, b) in enumerate(uniq):
        av = np.array(a, dtype=float)
        bv = np.array(b, dtype=float)
        E = Wg @ (bv - av)                                # (53,)
        x = Bgrid[None, :] * E[:, None]                   # (53, 7)
        ex = np.exp(-np.abs(x))
        sig = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
        P = 0.5 + Agrid[None, :, None] * (sig[:, None, :] - 0.5)  # (53,7,7)
        P = np.clip(P, 1e-6, 1.0 - 1e-6)
        LP[t] = np.log(P)
        L1P[t] = np.log1p(-P)

    tail_mask = gam > (TAIL + 1e-9)

    # ---- per-subject posterior P(gamma > 2 | own 96 trials) ----
    s_vals = np.empty(S)
    chunk = 64
    for i0 in range(0, S, chunk):
        Kc = K[i0:i0 + chunk]
        Nc = N[i0:i0 + chunk]
        logL = (np.tensordot(Kc, LP, axes=(1, 0)) +
                np.tensordot(Nc - Kc, L1P, axes=(1, 0)))   # (s,53,7,7)
        m = logL.max(axis=(1, 2, 3), keepdims=True)
        w = np.exp(logL - m)
        post = w.sum(axis=(2, 3))                          # marginal over A,beta
        tot = post.sum(axis=1)
        tot = np.where(tot <= 0.0, 1.0, tot)
        s_vals[i0:i0 + chunk] = post[:, tail_mask].sum(axis=1) / tot

    raw = float(np.mean(s_vals))
    # empirical-Bayes shrinkage toward the prior mean; with one subject the
    # posterior mean is prior-dominated (tight across subjects), with many
    # subjects the data dominate (population tail mass).
    return (KAPPA * P0 + S * raw) / (KAPPA + S)

### RESULTS
- Predicted under pi_7 (simulated): 0.0990 (var=0.0001)
- Predicted under pi_8 (simulated): 0.1396 (var=0.0003)
- Observed on real data: 0.2983 (var=0.0006)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dissociation score: on trials where Take-The-Best (top-validity
    # cascade) and Tallying (cue-count majority) make DIFFERENT modal
    # predictions, how often does the subject side with the TTB option?
    # TTB predicts > 0.5 on every such trial (constant sigmoid(beta)
    # confidence, including on tally-tie trials 7/8 where Tallying must
    # guess at exactly 50%). Tallying predicts < 0.5 on conflict trials
    # (majority option wins) and exactly 0.5 on tally ties. So the metric
    # is positive under TTB and negative under Tallying.
    V = np.array([0.95, 0.85, 0.65, 0.60, 0.50])
    order = np.argsort(-V, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    matches = []
    for i in range(len(a_col)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)

        # TTB winner: first discriminating cue in validity order.
        ttb = -1
        for j in order:
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb == -1:
            continue  # TTB guesses; not diagnostic

        # Tallying winner: strict cue-count majority.
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins > b_wins:
            tal = 0
        elif b_wins > a_wins:
            tal = 1
        else:
            tal = -1  # tally tie -> Tallying guesses

        # Keep only dissociation trials: the two theories' modal
        # predictions differ (tally tie, or tally majority against TTB).
        if tal == ttb:
            continue

        matches.append(1.0 if resp[i] == ttb else 0.0)

    if len(matches) == 0:
        return 0.0
    # Center at 0.5: positive => TTB-like, negative => Tallying-like.
    return float(np.mean(matches)) - 0.5
```

**Observed (real) value:** 0.1580 (var=0.0120)
**Predicted under pi_7:** 0.1660 (var=0.0056)
**Predicted under pi_8:** 0.1537 (var=0.0082)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the TALLY winner | Tallying and Take-The-Best disagree).

    On every trial we compute (a) the tally winner (option winning more
    strict feature-wise comparisons) and (b) the TTB winner (winner on
    the first discriminating cue in descending-validity order, cues here
    are already ordered by validity). We keep only CONFLICT trials where
    both heuristics have a definite winner and they point to opposite
    options, and return the proportion of choices falling on the tally
    winner's side.

    Under Tallying this proportion is high (sigmoid(2*beta) on the
    margin-2 grid conflicts, sigmoid(6*beta) on the extreme-conflict
    trials where one option wins 7 of 8 cues); under TTB it is low
    (1 - sigmoid(beta)). Works on the pooled DataFrame and on any
    single subject's slice (each subject has 7 unique conflict trials
    x 4 reps = 28 conflict observations).
    """
    validities = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
    cue_order = np.argsort(-validities, kind="stable")

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape != b.shape:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying needs a strict majority; skip tied tallies.
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1

        # Take-The-Best winner: first discriminating cue in validity order.
        ttb_winner = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        # Keep only genuine conflicts between the two heuristics.
        if ttb_winner == tally_winner:
            continue

        resp = int(row["response"])
        hits.append(1.0 if resp == tally_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.1814 (var=0.0118)
**Predicted under pi_7:** 0.1764 (var=0.0097)
**Predicted under pi_8:** 0.1571 (var=0.0041)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Depth/conflict invariance contrast for TTB vs PWEI.

    Three structurally identified trial classes (validity order is fixed:
    [0.90, 0.89, 0.88, 0.87, 0.70, 0.60]):

    HIGH  - the top cue (0.90) discriminates and every other discriminating
            cue agrees with it (design trials 1, 2, 9, 10; 24 obs/subject).
            PWEI evidence margin |D| is maximal (~1 or w_top).
    DEPTH5- cues 1-4 all tie; only the 0.70-validity cue discriminates
            (design trials 11, 12; 12 obs/subject). PWEI's weight for
            0.70^gamma is ~1e-5 of total, so PWEI predicts ~50% choices.
    REV   - the top cue discriminates one way, the three sub-top cues
            (0.89, 0.88, 0.87) discriminate the other way, tail cues tie
            (design trials 3, 4; 12 obs/subject). In PWEI's steep-gamma
            regime the sub-top block outweighs the top cue, so PWEI
            chooses AGAINST the TTB winner.

    For each class compute p = P(choose the TTB winner). TTB's structural
    invariant is a single constant confidence p* on every discriminating
    trial, so 2*p_HIGH - p_DEPTH5 - p_REV = 0 exactly, at ANY (beta, eps).
    PWEI predicts p_HIGH ~ 0.9, p_DEPTH5 ~ 0.5, p_REV ~ 0.1-0.37, giving a
    value ~0.9-1.2 across its entire parameter box.
    """
    VALS = np.array([0.90, 0.89, 0.88, 0.87, 0.70, 0.60])
    order = np.argsort(-VALS, kind="stable")  # validity-descending cue indices

    n_high, k_high = 0, 0
    n_d5, k_d5 = 0, 0
    n_rev, k_rev = 0, 0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)  # +1 favors A, -1 favors B, 0 tie, per cue

        # First discriminating cue in validity order (TTB's search).
        first = None
        for j in order:
            if s[j] != 0.0:
                first = j
                break
        if first is None:
            continue  # identical options: both theories say 50%

        ttb_winner_is_a = s[first] > 0
        chose_ttb = (int(resp) == 0) == ttb_winner_is_a

        if first == 4:
            # DEPTH5: only the 0.70-validity cue discriminates.
            n_d5 += 1
            k_d5 += int(chose_ttb)
        elif first == 0 and s[1] == -s[0] and s[2] == -s[0] and s[3] == -s[0] \
                and s[4] == 0.0 and s[5] == 0.0:
            # REV: top cue vs the three sub-top cues, tail ties.
            n_rev += 1
            k_rev += int(chose_ttb)
        elif first == 0:
            # HIGH candidate: all other discriminating cues agree with top.
            others = [s[j] for j in range(1, 6) if s[j] != 0.0]
            if all(o == s[0] for o in others):
                n_high += 1
                k_high += int(chose_ttb)
        # All other trial types (mixed conflicts, cue-2-led trials, ties)
        # are deliberately excluded to keep the classes pure.

    p_high = k_high / n_high if n_high > 0 else 0.5
    p_d5 = k_d5 / n_d5 if n_d5 > 0 else 0.5
    p_rev = k_rev / n_rev if n_rev > 0 else 0.5

    return 2.0 * p_high - p_d5 - p_rev
```

**Observed (real) value:** -0.7167 (var=0.0939)
**Predicted under pi_7:** -0.7067 (var=0.0731)
**Predicted under pi_8:** -0.7200 (var=0.0585)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # OLS slope of "chose the TTB (first-discriminating-cue) winner" on the
    # PWEI power-weighted evidence margin measured TOWARD that winner.
    # TTB: choice prob is one constant p* on every discriminating trial
    #      (any beta, eps)  =>  slope exactly 0.
    # PWEI: prob is a psychometric function of the weighted margin D
    #      =>  strongly positive slope (~0.5 over the design's margin range).
    val = np.array([0.95, 0.92, 0.91, 0.90, 0.89, 0.88], dtype=float)
    gamma = 40.0  # mid of PWEI's calibrated regime [35, 50]
    w = np.power(val, gamma)
    w = w / w.sum()
    cue_order = np.argsort(-val, kind="stable")

    xs = []
    ys = []
    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        s = np.sign(a - b)
        if not np.any(s != 0):
            # Identical options: pure guessing for both theories; excluded.
            continue
        # TTB winner: first discriminating cue in descending-validity order.
        winner = None
        for j in cue_order:
            if s[j] != 0:
                winner = 0 if s[j] > 0 else 1
                break
        if winner is None:
            continue
        # Power-weighted evidence for option A.
        D = float(np.sum(w * s))
        # Margin re-signed so that positive = evidence favors the TTB winner.
        x = D if winner == 0 else -D
        y = 1.0 if int(r) == winner else 0.0
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0
    sxx = float(np.sum((xs - xs.mean()) ** 2))
    if sxx <= 1e-12:
        return 0.0
    sxy = float(np.sum((xs - xs.mean()) * (ys - ys.mean())))
    return float(sxy / sxx)
```

**Observed (real) value:** -0.5230 (var=0.0540)
**Predicted under pi_7:** -0.5575 (var=0.0274)
**Predicted under pi_8:** -0.5635 (var=0.0198)

### Experiment 7
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Lower-cue evidence-usage slope.

    For every trial with a discriminating cue, determine the Take-The-Best
    winner (first cue in descending-validity order that differs). Pool trials
    into cells keyed by the inverted-polarity integrated evidence favoring
    that winner, E_w = sum_j w_j * (loser_j - winner_j), w_j proportional to
    validity. Then compute the OLS slope of P(choose the TTB winner) on E_w
    across cells.

    TTB: the winner's choice probability is the SAME constant p* on every
    discriminating trial (only the top discriminating cue is used), so the
    psychometric function over E_w is exactly flat -> slope = 0 for ANY
    (beta, epsilon).

    SPI: choice probability is a strictly increasing sigmoid of E_w
    (rho < 0.5 preserves monotonicity, lapse is affine), and the design's
    cells sweep E_w from -1 (unanimous endorsement) to +0.58 -> large
    positive slope.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.70, 0.65, 0.60], dtype=float)
    w = validities / validities.sum()
    cue_order = np.argsort(-validities, kind="stable")

    a_cells = data["option_a_ratings"].tolist()
    b_cells = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    n = len(data)
    E_list = []
    Y_list = []

    for i in range(n):
        a = np.asarray(a_cells[i], dtype=float).ravel()
        b = np.asarray(b_cells[i], dtype=float).ravel()
        if a.shape[0] != validities.shape[0] or b.shape[0] != validities.shape[0]:
            continue

        # TTB cascade: first discriminating cue in validity order.
        winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner < 0:
            continue  # no discriminating cue: uninformative for the contrast

        # Inverted-polarity integrated evidence favoring the TTB winner:
        # the loser's endorsements (1s) count FOR the winner (sparse option).
        if winner == 0:
            E = float(np.sum(w * (b - a)))
        else:
            E = float(np.sum(w * (a - b)))

        E_list.append(E)
        Y_list.append(1.0 if int(resp[i]) == winner else 0.0)

    if len(E_list) < 4:
        return 0.0

    E = np.asarray(E_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)

    # Pool trials into cells by (rounded) evidence value; mirrored twins and
    # repetitions of the same pair share the same E_w and are averaged.
    keys = np.round(E, 4)
    uniq = np.unique(keys)
    if uniq.shape[0] < 2:
        return 0.0

    cell_x = np.empty(uniq.shape[0], dtype=float)
    cell_y = np.empty(uniq.shape[0], dtype=float)
    for idx in range(uniq.shape[0]):
        mask = keys == uniq[idx]
        cell_x[idx] = E[mask].mean()
        cell_y[idx] = Y[mask].mean()

    x = cell_x - cell_x.mean()
    y = cell_y - cell_y.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0

    return float(np.sum(x * y) / denom)
```

**Observed (real) value:** 0.5944 (var=0.0376)
**Predicted under pi_7:** 0.5915 (var=0.0172)
**Predicted under pi_8:** 0.5704 (var=0.0179)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design constants: 8 cues with the experiment's validities.
    validities = np.array([0.95, 0.92, 0.62, 0.60, 0.62, 0.60, 0.62, 0.60])

    # Parse per-trial rating vectors (cells are lists -> project safely).
    a_lists = [list(r) for r in data["option_a_ratings"]]
    b_lists = [list(r) for r in data["option_b_ratings"]]
    A = np.asarray(a_lists, dtype=float)
    B = np.asarray(b_lists, dtype=float)
    resp = np.asarray(data["response"]).astype(int)

    n_feat = A.shape[1]
    if n_feat == len(validities):
        cue_order = np.argsort(-validities, kind="stable")
    else:
        cue_order = np.arange(n_feat)

    n_a = A.sum(axis=1)
    n_b = B.sum(axis=1)

    # TTB winner per trial: first discriminating cue in validity order.
    winner = np.full(A.shape[0], -1, dtype=int)
    for j in cue_order:
        open_rows = winner == -1
        if not open_rows.any():
            break
        wa = open_rows & (A[:, j] > B[:, j])
        wb = open_rows & (B[:, j] > A[:, j])
        winner[wa] = 0
        winner[wb] = 1

    abs_diff = np.abs(n_a - n_b)
    denser = np.where(n_a > n_b, 0, 1)  # option carrying MORE 1-ratings

    # Extreme-count trials only (|n_A - n_B| >= 5): the sign of SPI's
    # evidence is robust across its whole parameter box on these cells.
    valid = (winner >= 0) & (abs_diff >= 5)
    ttb_picks_denser = (winner == denser)

    chose_winner = (resp == winner)

    grp_d = valid & ttb_picks_denser      # TTB winner is the DENSER option
    grp_s = valid & (~ttb_picks_denser)    # TTB winner is the SPARSER option

    nd = int(grp_d.sum())
    ns = int(grp_s.sum())
    if nd == 0 or ns == 0:
        return 0.0

    p_denser = float(chose_winner[grp_d].mean())
    p_sparser = float(chose_winner[grp_s].mean())

    return p_denser - p_sparser
```

**Observed (real) value:** -0.6758 (var=0.0393)
**Predicted under pi_7:** -0.6850 (var=0.0317)
**Predicted under pi_8:** -0.6883 (var=0.0128)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Per-subject choice CONSISTENCY (extremity) toward the sparse option on
    # near-unanimous trials (one option all-zeros, the other endorsed on at
    # least n_features-1 cues), averaged over subjects.
    #
    # RFI-delta: polarity is a stable per-subject trait with only a small
    # attentional slip (delta <= 0.12) and small lapse (eps <= 0.08), so EVERY
    # subject -- majority (anti-endorsement) and minority (conventional)
    # alike -- is strongly and stably consistent on these trials:
    # per-subject p(sparse) is either ~[0.80, 0.92] or its mirror ~[0.08, 0.20],
    # i.e. |2p-1| ~ [0.6, 0.85] for essentially all subjects, with tight
    # between-subject spread.
    #
    # SPI: the wide lapse/polarity-mixing boxes (eps in [0.05, 0.35],
    # rho in [0, 0.35], beta in [0.5, 20]) mean per-subject attenuation varies
    # enormously; the average subject sits much closer to chance (mean
    # |2p-1| far below RFI's), and the population spread is wide.
    # The absolute value makes the statistic polarity-symmetric, so RFI's
    # small conventional minority does NOT inflate the between-subject
    # variance (both clusters have the same extremity).

    a_ratings = data['option_a_ratings'].tolist()
    b_ratings = data['option_b_ratings'].tolist()
    resp = data['response'].astype(float).values
    subj = data['subject_id'].values

    n = len(data)
    a_sum = np.empty(n, dtype=float)
    b_sum = np.empty(n, dtype=float)
    n_feat = np.empty(n, dtype=float)
    for i in range(n):
        av = list(a_ratings[i])
        bv = list(b_ratings[i])
        a_sum[i] = float(np.sum(av))
        b_sum[i] = float(np.sum(bv))
        n_feat[i] = float(len(av))

    # Near-unanimous trials: sparse side has zero endorsements, dense side
    # is endorsed on all or all-but-one cue (E in ~[0.8, 1.0] for any
    # weighting gamma in [0, 2]).
    thresh = n_feat - 1.0
    sparse_is_a = (a_sum == 0.0) & (b_sum >= thresh)
    sparse_is_b = (b_sum == 0.0) & (a_sum >= thresh)
    extreme = sparse_is_a | sparse_is_b

    # 1 if the subject chose the all-zeros (sparse) option, else 0.
    chose_sparse = np.where(sparse_is_a, 1.0 - resp, resp)

    vals = []
    for s in np.unique(subj[extreme]):
        m = extreme & (subj == s)
        if m.sum() == 0:
            continue
        p = float(np.mean(chose_sparse[m]))
        vals.append(abs(2.0 * p - 1.0))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.7044 (var=0.0382)
**Predicted under pi_7:** 0.7111 (var=0.0259)
**Predicted under pi_8:** 0.7178 (var=0.0239)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Polarity-agnostic preference EXTREMITY on high-contrast trials.
    #
    # Select trials where the two options' endorsement counts differ by >= 3
    # (design trials R3..R8 and mirrors: |E| in [~0.35, 1.0] for every gamma
    # in [0,2]; the count-sparser option is the inverted-evidence-favored
    # option on all of them). Per subject, compute p_hat = rate of choosing
    # the count-sparser option on those 48 trials, then take |p_hat - 0.5|.
    # The metric is invariant to a subject's polarity (a conventional-polarity
    # subject and a red-flag subject both yield large extremity), which
    # neutralizes RFI's ~5% minority cluster, and directly measures the
    # ATTENUATION away from a pure deterministic preference: SPI's large
    # lapse (eps in [0.05,0.35], mean ~0.20) plus polarity mixture
    # (rho in [0,0.35], mean ~0.175) pulls per-subject p toward 0.5
    # (predicted extremity ~0.22), whereas RFI-delta's small slip
    # (delta ~0.10) and tiny lapse (eps <= 0.08) keep every subject
    # near saturation (predicted extremity ~0.35, per-subject range
    # [0.75, 0.92] in p). Applied to the pooled DataFrame the function
    # returns the mean of the per-subject values, so the pooled point
    # estimate is exactly the population mean of the per-subject metric.
    sids = []
    chose_sparse = []
    for r in data.itertuples(index=False):
        a = np.asarray(r.option_a_ratings, dtype=float)
        b = np.asarray(r.option_b_ratings, dtype=float)
        sa = int(a.sum())
        sb = int(b.sum())
        # Keep only high-contrast trials (count difference >= 3).
        if abs(sa - sb) < 3:
            continue
        resp = int(r.response)  # 0 = chose A, 1 = chose B
        sparse_is_a = sa < sb
        picked_sparse = (resp == 0) if sparse_is_a else (resp == 1)
        sids.append(r.subject_id)
        chose_sparse.append(1.0 if picked_sparse else 0.0)
    if len(sids) == 0:
        return 0.0
    tmp = pd.DataFrame({"sid": sids, "cs": chose_sparse})
    per_subj = tmp.groupby("sid")["cs"].mean()
    extremity = (per_subj - 0.5).abs()
    return float(extremity.mean())
```

**Observed (real) value:** 0.3467 (var=0.0108)
**Predicted under pi_7:** 0.3521 (var=0.0065)
**Predicted under pi_8:** 0.3438 (var=0.0047)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.98, 0.95, 0.88, 0.85, 0.82, 0.80,
                  0.78, 0.76, 0.74, 0.72, 0.62, 0.62], dtype=float)
    WTIE = np.power(V, 0.75)

    df = data.reset_index(drop=True)

    # ---------- 1. classify trials into evidence cells ----------
    # Cell = the dense-minus-sparse endorsement pattern; its evidence under
    # any gamma is x(gamma) = sum_j v_j^gamma * dvec_j  (>= 0 on this design).
    subj_cells = {}
    for i in range(len(df)):
        a = np.asarray(df.at[i, "option_a_ratings"], dtype=float)
        b = np.asarray(df.at[i, "option_b_ratings"], dtype=float)
        if a.shape != b.shape or a.shape[0] != V.shape[0]:
            continue
        if np.array_equal(a, b):
            continue  # identical vectors: uninformative
        ca = float(np.sum(a))
        cb = float(np.sum(b))
        if ca < cb:
            sparse, dvec = 0, b - a
        elif cb < ca:
            sparse, dvec = 1, a - b
        else:
            wa = float(np.dot(WTIE, a))
            wb = float(np.dot(WTIE, b))
            if wa < wb:
                sparse, dvec = 0, b - a
            else:
                sparse, dvec = 1, a - b
        y = 1.0 if int(df.at[i, "response"]) == sparse else 0.0
        key = tuple(int(round(v)) for v in dvec)
        sid = df.at[i, "subject_id"]
        cells = subj_cells.setdefault(sid, {})
        ent = cells.setdefault(key, [0, 0.0])
        ent[0] += 1
        ent[1] += y

    if not subj_cells:
        return 0.0

    # ---------- 2. per-subject polarity orientation ----------
    # RFI-delta has a ~5.5% conventional-polarity minority; orienting each
    # subject by their overall sparse-choice rate maps that minority onto
    # the majority cluster (removing the huge mixture variance a signed
    # statistic would otherwise suffer). Under GSP every subject's overall
    # sparse rate stays well above 0.5 (min ~0.65), so orientation never
    # misfires there.
    for cells in subj_cells.values():
        tot_n = sum(e[0] for e in cells.values())
        tot_k = sum(e[1] for e in cells.values())
        if tot_n > 0 and tot_k < 0.5 * tot_n:
            for e in cells.values():
                e[1] = e[0] - e[1]

    # ---------- 3. reference mixtures ----------
    # Both theories reduce to the SAME per-subject family
    #     p_i = q*sigmoid(s*x_i(gamma)) + (1-q)*(1 - sigmoid(s*x_i(gamma)))
    # with theory-specific priors:
    #   RFI: s = 2*beta/sum(v^gamma), beta~U[3,8]  -> s in [0.5, 1.69];
    #        q = (1-eps)(1-delta)+eps/2 in [0.85, 0.92] (TIGHT); gamma in [0,1]
    #   GSP: s = beta_raw in [1.1, 3.2] (log-uniform);
    #        q = (1-eps)*m+eps/2 in [0.72, 0.98] (WIDE); gamma in [0, 1.5]
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    keys = sorted({k for cells in subj_cells.values() for k in cells})
    ncell = len(keys)
    if ncell == 0:
        return 0.0
    key_mat = np.array([list(k) for k in keys], dtype=float)  # (C, 12)

    GAMMAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    XG = np.array([key_mat @ np.power(V, g) for g in GAMMAS])  # (7, C)

    # RFI mixture components
    BETAS = np.array([3.0, 4.25, 5.5, 6.75, 8.0])
    Q_RFI = np.array([0.86, 0.885, 0.91])
    QW_RFI = np.array([0.25, 0.5, 0.25])
    rg, rs, rq, rw = [], [], [], []
    for gi in range(5):  # gamma in [0, 1]
        Sg = float(np.sum(np.power(V, GAMMAS[gi])))
        for be in BETAS:
            s = 2.0 * be / Sg
            for qi in range(3):
                rg.append(gi)
                rs.append(s)
                rq.append(Q_RFI[qi])
                rw.append(QW_RFI[qi] / 25.0)
    rg = np.array(rg); rs = np.array(rs); rq = np.array(rq); rw = np.array(rw)

    # GSP mixture components
    SG = np.exp(np.linspace(np.log(1.1), np.log(3.2), 7))
    Q_GSP = np.array([0.735, 0.79, 0.845, 0.90, 0.955])
    gg, gs, gq, gw = [], [], [], []
    w_unit = 1.0 / (7 * 7 * 5)
    for gi in range(7):  # gamma in [0, 1.5]
        for s in SG:
            for q in Q_GSP:
                gg.append(gi); gs.append(s); gq.append(q); gw.append(w_unit)
    gg = np.array(gg); gs = np.array(gs); gq = np.array(gq); gw = np.array(gw)

    def comp_logps(gidx, slopes, qs):
        z = slopes[:, None] * XG[gidx, :]
        sig = sigmoid(z)
        p = qs[:, None] * sig + (1.0 - qs[:, None]) * (1.0 - sig)
        p = np.clip(p, 0.01, 0.99)
        return np.log(p), np.log(1.0 - p)

    LP_r, L1_r = comp_logps(rg, rs, rq)
    LP_g, L1_g = comp_logps(gg, gs, gq)
    LW_r = np.log(rw)
    LW_g = np.log(gw)

    def logmix(LP, L1, LW, k_vec, n_vec):
        ll = (k_vec[None, :] * LP + (n_vec - k_vec)[None, :] * L1).sum(axis=1) + LW
        m = np.max(ll)
        return float(m + np.log(np.sum(np.exp(ll - m))))

    # ---------- 4. per-subject marginal log-likelihood ratio ----------
    vals = []
    for sid in sorted(subj_cells, key=str):
        cells = subj_cells[sid]
        k_vec = np.zeros(ncell)
        n_vec = np.zeros(ncell)
        for ci, key in enumerate(keys):
            ent = cells.get(key)
            if ent is not None:
                n_vec[ci] = float(ent[0])
                k_vec[ci] = float(ent[1])
        if n_vec.sum() <= 0:
            continue
        vals.append(logmix(LP_r, L1_r, LW_r, k_vec, n_vec)
                    - logmix(LP_g, L1_g, LW_g, k_vec, n_vec))

    if not vals:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -1.7633 (var=3.8102)
**Predicted under pi_7:** -1.3031 (var=2.4039)
**Predicted under pi_8:** -0.6950 (var=1.2398)

### Experiment 12
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ------------------------------------------------------------------
    # Theory-neutral psychometric slope test.
    #
    # Both theories are EXACTLY re-parameterizable, on every trial, as
    #   p(choose A) = 0.5 + A * ( sigmoid(s * E_norm(gamma)) - 0.5 ),
    #     E_norm(gamma) = sum_j v_j^gamma (b_j - a_j) / sum_j v_j^gamma,
    # where
    #   GSP  (raw weights):    s = beta_raw * sum(v^gamma) in [9.1, 38.4],
    #                          A = (1-eps)(2m-1) in [0.44, 0.98];
    #   RFI  (normalized wts): s = beta in [3, 8],
    #                          A = +-(1-eps)(1-2 delta), |A| in [0.70, 0.84].
    # The two slope supports are DISJOINT: every GSP subject's effective
    # normalized evidence slope lies above ~9, every RFI subject's below 8.
    # The attenuation (m, eps, delta) and the polarity sign are absorbed
    # by the free coefficient A in [-1, 1], which removes exactly the
    # level/attenuation heterogeneity that destroyed the level-based
    # statistics tried before.
    #
    # Per subject we fit (gamma, s, A) by binomial maximum likelihood on
    # ALL 96 trials (grid search; exact likelihood, no approximation),
    # and report the posterior probability that s > 9.5 under a
    # log-uniform prior on s in [1.2, 48] (profiled over gamma and A).
    #   GSP subjects -> posterior mass almost entirely above 9.5
    #                  (expected value ~0.75-0.85);
    #   RFI subjects -> posterior mass below 9.5
    #                  (expected value ~0.15-0.45).
    # The pooled metric is the mean of the per-subject posterior
    # probabilities; a single subject's slice returns that subject's
    # posterior probability.
    # ------------------------------------------------------------------

    VALS = np.array([0.99, 0.95, 0.88, 0.85, 0.82, 0.80, 0.78,
                     0.76, 0.74, 0.72, 0.50, 0.50], dtype=float)

    GAMMA_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])
    S_GRID = np.logspace(np.log10(1.2), np.log10(48.0), 40)
    A_GRID = np.linspace(-0.98, 0.98, 41)
    S_THRESH = 9.5

    n_g, n_s, n_a = len(GAMMA_GRID), len(S_GRID), len(A_GRID)

    # per-gamma cue weights and their totals
    W = np.power(VALS[None, :], GAMMA_GRID[:, None])      # (n_g, F)
    WSUM = W.sum(axis=1)                                  # (n_g,)
    ABOVE = S_GRID > S_THRESH                            # (n_s,)

    def _vec(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        return np.asarray(list(x), dtype=float)

    def _subject_value(sub):
        a_list = sub["option_a_ratings"].tolist()
        b_list = sub["option_b_ratings"].tolist()
        r = sub["response"].to_numpy()

        # aggregate identical trials: key -> [n_chose_A, n_total]
        agg = {}
        for i in range(len(sub)):
            a = _vec(a_list[i])
            b = _vec(b_list[i])
            key = (tuple(int(round(v)) for v in a),
                   tuple(int(round(v)) for v in b))
            y = 1 if int(r[i]) == 0 else 0   # response 0 = chose A
            if key in agg:
                agg[key][0] += y
                agg[key][1] += 1
            else:
                agg[key] = [y, 1]

        nll = np.zeros((n_g, n_s, n_a), dtype=float)
        for (ta, tb), (k, n) in agg.items():
            da = np.asarray(ta, dtype=float)
            db = np.asarray(tb, dtype=float)
            d = db - da                     # endorsement difference (B - A)
            if not np.any(d != 0.0):
                continue                    # tie trials: no information
            e = (W @ d) / WSUM              # (n_g,) normalized evidence for A
            for gi in range(n_g):
                x = S_GRID * e[gi]           # (n_s,)
                ax = np.abs(x)
                ex = np.exp(-ax)
                sg = np.where(x >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
                dev = sg - 0.5                               # (n_s,)
                p = 0.5 + A_GRID[None, :] * dev[:, None]     # (n_s, n_a)
                p = np.clip(p, 1e-9, 1.0 - 1e-9)
                nll[gi] -= k * np.log(p) + (n - k) * np.log1p(-p)

        # profile the negative log-likelihood over gamma and A
        prof = nll.min(axis=(0, 2))         # (n_s,)
        w = np.exp(-(prof - prof.min()))   # likelihood weights (log-uniform s prior)
        return float(w[ABOVE].sum() / w.sum())

    subj = data["subject_id"].to_numpy()
    vals = []
    for s in np.unique(subj):
        sub = data[subj == s]
        vals.append(_subject_value(sub))
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.9113 (var=0.0227)
**Predicted under pi_7:** 0.9161 (var=0.0235)
**Predicted under pi_8:** 0.8983 (var=0.0297)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd
from collections import defaultdict


def metric(data: pd.DataFrame) -> float:
    """Total log marginal-likelihood ratio (log Bayes factor) of RSD-vgp over GSP.

    Both theories share the identical choice core
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E(gamma) = sum_j v_j**gamma * (b_j - a_j),
    and differ ONLY in the population supports of (gamma, log beta):
        RSD-vgp: gamma ~ U[0,2], log beta ~ U[ln1.8, ln6]
        GSP    : gamma ~ U[0,1.5], log beta ~ U[ln1.1, ln3.2]
    For every subject the exact binomial likelihood of their choices over the
    distinct trial types is integrated (grid) against each theory's prior; the
    attenuation A_eff gets a shared neutral prior (the two theories' A_eff
    ranges nearly coincide, so it must not drive the comparison). Subjects are
    independent, so on the pooled DataFrame the value is the sum of the
    per-subject log Bayes factors (the joint log BF of the whole dataset); on a
    single-subject slice it is that subject's individual log Bayes factor.
    Positive values favor RSD-vgp, negative values favor GSP.
    """
    VAL = np.array([1.0, 0.99, 0.92, 0.85, 0.75, 0.65,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    df = data
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    a_list = df['option_a_ratings'].tolist()
    b_list = df['option_b_ratings'].tolist()
    resp = np.asarray(df['response'].values, dtype=int)
    subj = df['subject_id'].values

    # Hashable string keys for the trial types (avoids unhashable list cells).
    abmap = {}
    key_list = []
    for i in range(n_rows):
        a = a_list[i]
        b = b_list[i]
        ka = ''.join(['1' if int(x) else '0' for x in a])
        kb = ''.join(['1' if int(x) else '0' for x in b])
        key = ka + '#' + kb
        key_list.append(key)
        if key not in abmap:
            abmap[key] = (np.array([int(x) for x in a], dtype=float),
                          np.array([int(x) for x in b], dtype=float))

    tkeys = sorted(abmap.keys())
    tidx = {k: i for i, k in enumerate(tkeys)}
    T = len(tkeys)

    K = defaultdict(lambda: np.zeros(T))  # A-choices per trial type
    N = defaultdict(lambda: np.zeros(T))  # presentations per trial type
    for i in range(n_rows):
        s = subj[i]
        ti = tidx[key_list[i]]
        N[s][ti] += 1.0
        if resp[i] == 0:
            K[s][ti] += 1.0

    # Parameter grids (cover both theories' supports).
    gam = np.arange(0.02, 2.0001, 0.02)      # validity exponent gamma
    lbg = np.arange(0.02, 1.9401, 0.02)      # log raw inverse temperature
    Agr = np.arange(0.42, 0.9901, 0.03)     # effective attenuation A_eff
    G, L, NA = len(gam), len(lbg), len(Agr)

    # E_t(gamma) = sum_j v_j^gamma (b_j - a_j), shape (G, T)
    W = VAL[None, :] ** gam[:, None]
    E = np.empty((G, T))
    for key in tkeys:
        a, b = abmap[key]
        E[:, tidx[key]] = W @ (b - a)

    # s = sigmoid(beta * E) - 0.5, shape (L, G, T)
    beta = np.exp(lbg)
    u = beta[:, None, None] * E[None, :, :]
    au = np.abs(u)
    ex = np.exp(-au)
    sig = np.where(u >= 0.0, 1.0 / (1.0 + ex), ex / (1.0 + ex))
    s = sig - 0.5

    # p(A) = 0.5 + A_eff * s, shape (NA, L, G, T)
    P = 0.5 + Agr[:, None, None, None] * s[None, :, :, :]
    P = np.clip(P, 1e-9, 1.0 - 1e-9)
    logP = np.log(P)
    log1P = np.log1p(-P)
    del P, s, sig, u, au, ex, E, W

    # Log priors.
    logwA = -np.log(NA)  # shared neutral attenuation prior
    logpR_g = np.full(G, np.log(0.5))                          # RSD: U[0,2]
    logpG_g = np.where(gam <= 1.5, np.log(1.0 / 1.5), -np.inf)  # GSP: U[0,1.5]
    loR, hiR = np.log(1.8), np.log(6.0)
    loG, hiG = np.log(1.1), np.log(3.2)
    logpR_b = np.where((lbg >= loR) & (lbg <= hiR),
                       -np.log(hiR - loR), -np.inf)              # RSD: U[ln1.8, ln6]
    logpG_b = np.where((lbg >= loG) & (lbg <= hiG),
                       -np.log(hiG - loG), -np.inf)             # GSP: U[ln1.1, ln3.2]

    PR = logpR_b[None, :, None] + logpR_g[None, None, :] + logwA  # (L, G)
    PG = logpG_b[None, :, None] + logpG_g[None, None, :] + logwA  # (L, G)

    def lse(x):
        m = np.max(x)
        if not np.isfinite(m):
            return -np.inf
        return m + np.log(np.sum(np.exp(x - m)))

    total_llr = 0.0
    for sid in list(N.keys()):
        kk = K[sid]
        nn = N[sid]
        if nn.sum() <= 0:
            continue
        # binomial log-likelihood on the (NA, L, G) parameter grid
        LL = (np.einsum('t,ijkt->ijk', kk, logP)
              + np.einsum('t,ijkt->ijk', nn - kk, log1P))
        total_llr += lse(LL + PR[None, :, :]) - lse(LL + PG[None, :, :])

    return float(total_llr)
```

**Observed (real) value:** 52.5984 (var=6.4849)
**Predicted under pi_7:** 96.6930 (var=15.9176)
**Predicted under pi_8:** 67.3084 (var=8.4566)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ------------------------------------------------------------------
    # Per-subject BOUNDED log Bayes factor (GSP vs RSD-vgp), averaged
    # over subjects:
    #     score_i = tanh( log m_GSP(data_i) - log m_RSD(data_i) )
    #
    # Both theories share the EXACT per-trial likelihood
    #     p(A) = 0.5 + A_eff * ( sigmoid(beta * E) - 0.5 ),
    #     E    = sum_j v_j^gamma * (b_j - a_j)   (raw anti-endorsement),
    #     A_eff = (1 - eps) * (2m - 1),
    # so eps and m enter ONLY through A_eff and each subject's 96 choices
    # identify just (gamma, ln beta, A_eff). The theories differ ONLY in
    # the population prior over (gamma, ln beta):
    #     GSP : gamma ~ U[0, 1.5],  ln beta ~ U[ln 1.1, ln 3.2]
    #     RSD : gamma ~ U[0, 2.0],  ln beta ~ U[ln 1.8, ln 6.0]
    # (RSD's two-component gamma mixture is algebraically uniform.)
    # Each marginal is computed by grid quadrature over the theory's exact
    # box, using THIS experiment's true validities
    #     [1, 1, 1, .5, .5, .5, .5, .5, .5, .6, .6, .7]
    # (verified by reproducing the design's own mid-box predictions:
    # B1 = .77, F0 = .42, C2 = .74 at gamma=.75, beta=2, A_eff=.70).
    # A_eff receives a shared uniform prior on [0.42, 0.98], which covers
    # both theories' induced ranges and cancels in the ratio.
    # tanh() bounds each subject's contribution to (-1, 1), capping the
    # heavy log-BF tails that destroyed the between-subject variance
    # budget of the unbounded version.
    # ------------------------------------------------------------------

    V = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7])

    gam = np.linspace(0.0, 2.0, 81)                    # gamma grid (union)
    lnb = np.linspace(np.log(1.1), np.log(6.0), 61)    # ln beta grid (union)
    ave = np.linspace(0.42, 0.98, 12)                   # A_eff grid (shared)
    betas = np.exp(lnb)
    ng, nb, na = len(gam), len(lnb), len(ave)

    gsp_g = gam <= 1.5 + 1e-9
    gsp_b = lnb <= np.log(3.2) + 1e-9
    rsd_b = lnb >= np.log(1.8) - 1e-9
    all_g = np.ones(ng, dtype=bool)

    def subject_score(sub: pd.DataFrame) -> float:
        # ---- pool mirrored twins into canonical (X, Y) cells, X <= Y ----
        agg = {}
        for a, b, r in zip(sub["option_a_ratings"].tolist(),
                           sub["option_b_ratings"].tolist(),
                           sub["response"].tolist()):
            ta = tuple((int(v) for v in a))
            tb = tuple((int(v) for v in b))
            if ta <= tb:
                X, Y, choseX = ta, tb, (int(r) == 0)
            else:
                X, Y, choseX = tb, ta, (int(r) == 1)
            key = (X, Y)
            cnt = agg.get(key)
            if cnt is None:
                cnt = agg[key] = [0, 0]
            cnt[0] += 1
            if choseX:
                cnt[1] += 1

        D, kX, nT = [], [], []
        for (X, Y), cnt in agg.items():
            d = np.asarray(Y, dtype=float) - np.asarray(X, dtype=float)
            if not np.any(d != 0.0):
                continue            # exact tie: p = 0.5, cancels in the BF
            D.append(d)
            kX.append(float(cnt[1]))
            nT.append(float(cnt[0]))
        if not D:
            return 0.0
        D = np.vstack(D)                                # (C, F)
        kX = np.asarray(kX)
        kY = np.asarray(nT) - kX

        nf = D.shape[1]
        if nf <= len(V):
            v = V[:nf]
        else:
            v = np.concatenate([V, np.full(nf - len(V), 0.6)])

        W = np.power(v[None, :], gam[:, None])          # (ng, F)
        E = D @ W.T                                     # (C, ng)
        x = E[:, :, None] * betas[None, None, :]         # (C, ng, nb)
        sig = 0.5 * (1.0 + np.tanh(0.5 * x))             # stable logistic
        C = sig.shape[0]
        sig = sig.reshape(C, ng * nb)

        LL = np.empty((na, ng * nb))
        for ia in range(na):
            p = 0.5 + ave[ia] * (sig - 0.5)
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            ll = kX[:, None] * np.log(p) + kY[:, None] * np.log1p(-p)
            LL[ia] = ll.sum(axis=0)
        LL = LL.reshape(na, ng, nb)

        def logm(gmask, bmask):
            # grid-average of exp(LL) over the theory's box = marginal
            # likelihood under the uniform prior on that box
            sub_ = LL[:, gmask, :][:, :, bmask]
            m = float(sub_.max())
            return m + float(np.log(np.mean(np.exp(sub_ - m))))

        lg = logm(gsp_g, gsp_b)
        lr = logm(all_g, rsd_b)
        return float(np.tanh(lg - lr))

    data = data.reset_index(drop=True)
    if len(data) == 0:
        return 0.0
    if "subject_id" not in data.columns:
        return subject_score(data)
    scores = [subject_score(sub)
              for _, sub in data.groupby("subject_id", sort=False)]
    if not scores:
        return 0.0
    return float(np.mean(scores))
```

**Observed (real) value:** -0.4144 (var=0.4154)
**Predicted under pi_7:** -0.2520 (var=0.3854)
**Predicted under pi_8:** -0.3417 (var=0.3534)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The two theories share an identical per-trial core (anti-endorsement polarity, raw validity weights, p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5)) and differ only in population priors over (gamma, beta, A_eff). Across the 14 experiments, that shared core is clearly right: on Experiments 3-10 and 12 both theories track the observed values closely (TTB-vs-tallying contrasts, evidence-usage slopes, extremity/consistency anchors, psychometric-slope posteriors all match to within noise). The arbitration therefore hinges on the population structure, and there the picture is: (1) Experiment 1 (log Bayes factor RSD-prior vs EPAI-prior) is decisive \u2014 observed +6.03, essentially on top of pi_7's predicted +5.32, while pi_8 predicts -30.93. The real data contain essentially no subjects in EPAI's gamma in (2.0, 2.4] band as read out by that quadrature; pi_8 is categorically rejected by its own discriminative weapon. (2) But pi_7 is not clean either: Experiment 2 estimates the population fraction of gamma > 2 at 0.298 \u2014 three times pi_7's prediction (0.099, pure estimation smear under its hard gamma <= 2 cap) and double pi_8's (0.140). Neither theory's population can produce this. (3) The remaining hard cases show a consistent direction: Experiment 13 (obs 52.6 vs 96.7 / 67.3) and Experiment 14 (obs -0.414 vs -0.252 / -0.342) both show the real population produces *attenuated* log-Bayes-factor magnitudes and drifts 'GSP-ward' relative to both theories, while Experiment 11 (obs -1.76 vs -1.30 / -0.70) also sits beyond both predictions. The coherent reading: real subjects have the shared anti-endorsement raw-weight core with steep slopes and moderate asymptotes (confirmed by Exps 9, 10, 12), but carry *additional within-subject stochasticity* (trial-level attention gating / cue-weight jitter, plus a low-precision subpopulation) that neither clean population prior captures. This extra noise flattens per-subject identification of gamma (inflating Exp 2's diffuse-posterior tail mass to ~0.30 without genuine gamma > 2.4 subjects, consistent with Exp 1's positive LLR), and shrinks every log-BF magnitude (Exps 13, 14, 11). Tallying closeness: pi_7 wins Exps 1, 4, 6, 7, 11, 12 and ties 8-10; pi_8 wins only 2, 3, 5, 13, 14, and its wins on 13/14 are small while its loss on 1 is catastrophic. So pi_7 is the clear incumbent winner, but it is falsified in its hard gamma <= 2 cap by Experiment 2 and over-predicts Exp 13 by ~2x \u2014 it is good, not perfect. Since neither theory explains all the data (pi_8 fails Exp 1 categorically; pi_7 fails Exp 2 by 3x and misses 13/14), the right move is not to regenerate either model's parameters but to replace the loser (pi_8) with a brand-new theory that keeps the empirically validated core and adds the missing stochastic population structure, making it a genuine stronger competitor to pi_7.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_8 (EPAI-xp) with a new theory \u2014 Stochastic-Gate Anti-Endorsement Integration (SGAI) \u2014 that keeps the empirically confirmed core (endorsements read as evidence AGAINST the endorsed option; E = sum_j v_j^gamma (b_j - a_j) with RAW unnormalized validity weights; p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2) but revises the population and adds within-subject noise, targeting exactly the three failures: (i) TRIAL-LEVEL ATTENTION GATE: on each trial, with probability delta in [0.10, 0.25], the subject re-samples cue attention, drawing an effective per-trial gamma_t from a wide distribution around their mean gamma (or falling back to raw cue counting); this within-subject jitter flattens per-subject gamma identification, which reproduces Experiment 2's inflated tail-mass estimate (~0.25-0.32 vs observed 0.298) WITHOUT placing genuine mass above gamma = 2.2, so Experiment 1's RSD-vs-EPAI log Bayes factor stays mildly positive (~+3 to +7, matching observed +6.03 and avoiding EPAI's -31). (ii) POPULATION PRIORS: gamma from a mixture with mean ~1.0 and support [0, 2.2] \u2014 predominantly below 2 with only a 5-10% sliver in (2.0, 2.2] (no mass above 2.2, so Exp 1 stays safe); beta log-uniform on [1.6, 6.0] (floor slightly BELOW RSD's 1.8, since Exps 13/14 hint at lower-precision subjects than either theory admits); m in [0.76, 0.98] and epsilon in [0.03, 0.12], keeping mean A_eff ~0.65-0.70 to preserve the good fits on Exps 9 and 10. (iii) LOW-PRECISION SUBPOPULATION: 10-15% of subjects with beta near the floor and elevated epsilon (0.15-0.25), absorbing the shallow-slope / GSP-ward drift seen in Experiments 13 (predicted ~45-60 vs observed 52.6, fixing RSD's 96.7 over-prediction) and 14 (predicted ~-0.38 to -0.45 vs observed -0.414). Because the per-trial core is unchanged, Experiments 3-10 and 12 should remain well fit; the new degrees of freedom (delta, gamma jitter width, low-precision mixture weight) are each pinned by a currently-missed experiment (2, 13/14, 11 respectively), giving the new theory falsifiable signatures distinct from both pi_7 (hard gamma <= 2, noise-free) and pi_8 (gamma mass in (2.0, 2.4], beta floor 2.0): per-subject psychometric functions with shallower mid-slope than their saturated asymptote implies, elevated trial-to-trial inconsistency on repeated identical trials beyond binomial noise, and a positive-but-attenuated Exp 1 log Bayes factor."
}
```

## Usage

```json
{
  "input_tokens": 45656,
  "output_tokens": 6697,
  "reasoning_tokens": 4360,
  "provider": "Reka"
}
```
