# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_10") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_10" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_10
Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69 — the 'steep normalized slope x moderate asymptote' family. The population is a three-segment mixture with SEGMENT-SPECIFIC PRECISION, all segments now at moderate-to-high precision (no soft-beta segment remains, since weakly identified slopes were shown to leak Exp 10/14 posteriors below threshold and to donate RFI-ward on Exp 9): (1) a 40% low bulk on gamma in [0, 1.0] with beta log-uniform on [2.2, 4.0] — the validated shared-core bulk, concentrated below the ~0.73/1.0 flip points of Exp 16's ladder, with the beta floor raised so every low-bulk subject's normalized slope is sharply identified; (2) a 25% mid segment on gamma in [1.05, 1.35] with beta log-uniform on [2.2, 3.2] — a band where RSD's uniform density (0.5) beats SGAI's (0.4) and EPAI's (0.375), sitting fully inside GSP's gamma box with GSP-compatible beta, so these subjects donate RSD-ward on Exp 15 while pulling Exp 11 and Exp 9 toward their observed values; (3) a 35% heavy tail on gamma in [2.0, 2.10] with beta log-uniform on [2.2, 2.8] — an extreme edge-hug of the RSD gamma boundary (average distance <= 0.05) at sharpened precision, so tail subjects' gamma posteriors concentrate just above 2.0 (feeding Exp 14), their slopes stay above the s>9.5 threshold (feeding Exp 10), and their near-deterministic past-flip-point behavior pulls Exp 16 down toward 0.4335. No per-trial gamma jitter. Falsifiable signatures: ~35% of subjects carry gamma within 0.10 of 2.0 with per-subject posteriors extending smoothly past 2.0; Exps 13/15 small and positive; Exp 11 in the 50-100 range; Exp 14 at or above 0.21; Exp 16 in [0.42, 0.45]; Exp 10 at or above 0.85.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HT-AEI-r6 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: three-segment mixture with SEGMENT-SPECIFIC
    # beta. All segments now carry moderate-to-high precision (no
    # soft-beta segment: weakly identified slopes were shown to leak
    # Exp 10/14 posteriors below threshold and donate RFI-ward on Exp 9).
    # gamma_u ~ U[0, 1] is mapped through:
    #   u in [0.00, 0.40): gamma ~ U[0.0, 1.0], beta log-U[2.2, 4.0]
    #            (40%; the validated low bulk, UNTOUCHED in gamma support.
    #            Beta floor raised 1.8 -> 2.2 per the pre-specified Exp 10
    #            contingency: sharpens per-subject slope identification
    #            without touching any other mechanism.)
    #   u in [0.40, 0.65): gamma ~ U[1.05, 1.35], beta log-U[2.2, 3.2]
    #            (25%; ENLARGED from 18%. On [1.05, 1.3] RSD's uniform
    #            density 0.5 beats SGAI's 0.4 and EPAI's 0.375, so these
    #            subjects donate RSD-ward on Exp 15; the band sits fully
    #            inside GSP's gamma box with GSP-compatible beta, pulling
    #            Exp 11 down toward the observed 52.6 and Exp 9 GSP-ward
    #            toward -1.76; it is above the 1.0 flip point
    #            (Exp-16-neutral); gamma near 1.2 matches the
    #            validity-proportional metrics of Exps 2 and 5.)
    #   u in [0.65, 1.00]: gamma ~ U[2.0, 2.10], beta log-U[2.2, 2.8]
    #            (35%; TRIMMED from 42% and RE-SHARPENED. Extreme edge-hug
    #            of the RSD gamma boundary (average distance <= 0.05), with
    #            beta raised from [1.9, 2.4] to [2.2, 2.8]: sharper gamma
    #            identification concentrates tail posteriors just above
    #            2.0 (Exp 14 recovers), keeps slopes above the s>9.5
    #            threshold (Exp 10), makes tail subjects near-deterministic
    #            past the flip points (Exp 16 pulled down toward 0.4335),
    #            and removes RFI-fittable soft slopes (Exp 9 back toward
    #            -1.76). The fraction trim cuts the residual Exp 13 edge
    #            cost ~17% while the mid segment's positive donations
    #            hold the total near the observed +6.)
    u = float(parameters["gamma_u"])
    if u < 0.40:
        gamma = 1.0 * (u / 0.40)
        log_b = float(parameters["log_beta_low"])
    elif u < 0.65:
        gamma = 1.05 + 0.30 * ((u - 0.40) / 0.25)
        log_b = float(parameters["log_beta_mid"])
    else:
        gamma = 2.0 + 0.10 * ((u - 0.65) / 0.35)
        log_b = float(parameters["log_beta_tail"])

    beta = float(np.exp(log_b))
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
    # (1-m) with the conventional polarity. Continuous, no empty middle,
    # no mirror cluster. NO per-trial gamma jitter (delta = 0): the
    # stochastic attention gate is a confirmed dead lever.
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
**Validities (n_features=14):** [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.72, 0.63]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
  trial 10: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 11: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 13: A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 16: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 18: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 20: A=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 22: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 23: A=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]
  trial 24: A=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

**Rationale:** STRUCTURAL SETUP. RSD-vgp and HT-AEI-r6 share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, m in [0.74-0.75, 1], epsilon in [0.03, 0.10-0.11] (A_eff boxes nearly identical, so extremity is NOT a usable lever). The theories are the SAME model with different population SUPPORTS, and only three support regions can ever dissociate them: (i) GAMMA GAP (1.35, 2.0]: RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout), placing 32.5% of subjects in (1.35, 2.0]; HT-AEI places exactly ZERO there (its segments are [0,1], [1.05,1.35], [2.0,2.10]). (ii) GAMMA TAIL [2.0, 2.10]: HT-AEI's 35% heavy-tail segment lives entirely here; RSD's gamma is hard-capped at 2.0, so ZERO RSD subjects exist above 2.0. (iii) BETA FLOOR: RSD beta log-uniform [1.8, 6.0]; every HT-AEI segment has beta >= 2.2 (low [2.2,4.0], mid [2.2,3.2], tail [2.2,2.8]). Beta in [1.8, 2.2) holds 16.7% of RSD subjects (ln(2.2/1.8)/ln(6/1.8) = 0.1667) and 0% of HT-AEI subjects. HT's beta ceiling 4.0 vs RSD's 6.0 is acknowledged one-sided (saturation-limited). The design makes all three boundaries empirically decisive using EXACT algebra.

ARCHITECTURE (14 experts): three cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- gamma-invariant unit cues), nine cues at 0.50 (weights halve per unit gamma), and two filler cues (0.72, 0.63) used ONLY in exactly-canceling shared endorsements or unanimous/tied positions, so they never perturb any evidence value. Shared endorsements on identical cue indices cancel exactly (w*(1-1) = 0), keeping both bars nonempty on every informative trial (conflict-style; locked-prompt requirement satisfied).

DISSOCIATION 1 -- THE EXACT FLIP LADDER (headline). Rungs R3-R8: option A is endorsed on the two non-shared unit cues (endorsement mass EXACTLY 2 for every gamma); option B is endorsed on k cues at validity 0.50. The evidence is then EXACTLY E(gamma) = k*0.5^gamma - 2, strictly decreasing with a single sign change at flip point g* = log2(k/2): gamma < g* subjects prefer the count-sparser unit-cue side A; gamma > g* prefer the low-validity-dense side B. Flip points: R3 (k=3) 0.585; R4 (k=4) 1.000; R5 (k=5) 1.322; R6 (k=6) 1.585; R7 (k=7) 1.807; R8 (k=8) 2.000. Predicted TRUE inversion rates P(gamma > g*): RSD 0.707 / 0.500 / 0.339 / 0.207 / 0.097 / 0.000 vs HT-AEI 0.766 / 0.600 / 0.373 / 0.350 / 0.350 / 0.350. Three signatures: (a) THE PLATEAU: HT-AEI's inversion rate is FLAT at 0.35 across R5-R8 (all its inverters there are tail subjects, gamma >= 2.0) and drops to zero only at 2.10, whereas RSD's uniform density forces a strictly LINEAR decline 0.34 -> 0.21 -> 0.10 -> 0 hitting a hard zero AT 2.0. RSD cannot produce a flat plateau at any parameterization; HT-AEI cannot put mass in the gap. (b) HARD FLOOR AT R8: E(gamma) = 8*0.5^gamma - 2 >= 0 for ALL gamma <= 2.0 (zero only at the measure-zero point gamma = 2), and m >= 0.74 > 0.5 keeps p >= 0.5 whenever E >= 0, so NO RSD subject can truly prefer the dense side on R8. HT-AEI REQUIRES its 35% tail there (E in [-0.134, 0), p as low as ~0.44 at gamma = 2.10, beta = 2.8 -- shallow per subject, population-detectable). (c) CONDITIONAL COUPLING: under HT-AEI every R6 inverter has gamma >= 2.0 (the gap is empty), so P(inverts R7 | inverts R6) = 1.0 and P(R8 rate < 0.5 | inverts R6) ~ 1; under RSD, P(inverts R7 | inverts R6) = (2-1.807)/(2-1.585) = 0.465 and P(R8 < 0.5 | inverts R6) = 0 HARD. Similarly P(inverts R7) itself: 35% (HT) vs 9.7% (RSD). (d) THE GAP HISTOGRAM: per-subject gamma recovered from the six rungs (48 observations) lands in bins [1.322,1.585)/[1.585,1.807)/[1.807,2.0] with total mass ~34% under RSD vs ~2.3% under HT-AEI (the sliver (1.322,1.35]); and above 2.0: 0% (RSD, hard) vs 35% (HT). Within-subject monotonicity (inverting R6 implies inverting R5..R3) holds under both and is a data-quality check. (e) SECONDARY SHAPE: HT-AEI's survival slope is ~0.40 below g*=1.0, ~0.70 on [1.0,1.322] (mid-segment density 0.833 vs RSD's 0.5) vs RSD's constant 0.50 -- a slope crossover, plus a 1.67x density bump on [1.05,1.35]. Depth among R7 inverters also differs (HT inverters all have gamma in [2,2.1], p ~ 0.35-0.42; RSD inverters have gamma in (1.807,2], p ~ 0.36-0.50, mean ~0.44 vs ~0.38).

DISSOCIATION 2 -- BETA FLOOR VIA EXACT-E CELLS. B1/B2 use only the unit cues with exactly canceling shared endorsements: E = +1 and +2 EXACTLY for every subject at every gamma (B1: A endorsed on unit cue 1, B on unit cues 2-3; B2: B on unit cues 2-3, unit cue 1 shared). The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]); HT-AEI: r12 <= 1.219 for ALL subjects (beta >= 2.2). The region r12 in (1.219, 1.322] is RSD-EXCLUSIVE: 16.7% of RSD subjects, 0% of HT-AEI. Any subject credibly there categorically falsifies HT-AEI; a hard zero there falsifies RSD's beta floor. Corroborating one-sided lever: among high-gamma subjects (R7/R8 inverters), HT-AEI's tail caps beta at 2.8 (r12 >= 1.121) while RSD's independent supports put ~63% of them above beta = 2.8 (r12 < 1.121) -- population-level among the ~10-35 high-gamma subjects. HT's beta ceiling 4.0 (vs RSD 6.0) moves r12 only between 1.005 and 1.037 -- below per-subject resolution, honestly acknowledged as one-sided.

CONTROLS. U/Um (one bar all-0, one all-1) saturate every subject (|E| = S(gamma) >= 5.9 even at gamma = 2.1 with beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 -- required to normalize the beta probes. Uc/Ucm are conflict-style saturated twins (A endorsed on a single 0.50 cue, B on all 13 others; |E| >= 5.5): they verify the A_eff anchor is not an empty-bar artifact. C/Cm is a 5-5 endorsement-count tie (A endorsed on the two unit cues, B on two 0.50 cues, shared fillers): E = 2*0.5^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.75-0.85) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2 are identical non-trivial vectors: both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs), sparse side (B-cells, U, Uc), or low-validity side (C).

POOLED PREDICTIONS (approximate mid-box planning values, p(unit-cue side), RSD vs HT-AEI): R3 ~0.31/0.35 (crossover: HT's shallow low-segment inverters pull its R3 up); R4 ~0.52/0.46; R5 ~0.60/0.54; R6 ~0.69/0.61; R7 ~0.73/0.67; R8 ~0.75/0.70. Pooled differences are modest (0.04-0.08) and honestly acknowledged: the populations' central tendencies nearly coincide, and the decisive evidence is SUBJECT-LEVEL population structure. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6]; HT-AEI: the three-segment gamma mixture with segment-specific beta boxes), plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R8 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at exactly 0.5), while HT-AEI requires 35% of subjects there; (b) inversion-rate survival curve over g* = 0.585/1.000/1.322/1.585/1.807/2.000: RSD 0.707/0.500/0.339/0.207/0.097/0 (linear, hard edge at 2.0) vs HT-AEI 0.766/0.600/0.373/0.350/0.350/0.350 (flat 0.35 plateau on [1.32, 2.0], cliff only at 2.10); (c) per-subject gamma histogram: RSD flat density 0.5 with hard edge at 2.0 vs HT-AEI's ~34%-vs-2.3% gap contrast on (1.322, 2.0] and 35%-vs-0% mass above 2.0, with a 1.67x bump on [1.05,1.35] and depletion on [0,1]; (d) P(inverts R7 | inverts R6) = 1.0 and P(R8 < 0.5 | inverts R6) ~ 1 (HT) vs 0.465 and 0 (RSD); (e) per-subject r12 in (1.219, 1.322] (beta-hat in [1.8, 2.2)): ~16.7% (RSD) vs 0% hard (HT-AEI); (f) among R7/R8 inverters, r12 >= 1.121 for all (HT tail beta <= 2.8) vs ~63% below 1.121 (RSD); (g) within-subject monotone flip pattern (inverting R6 implies inverting R5..R3); (h) count-tie C preferring the low-validity side above 50% and identical-vector ties at exactly 50% (shared controls).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
WHY THIS METRIC. The two theories share the IDENTICAL per-trial likelihood p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5) with E = sum_j v_j^gamma*(b_j - a_j); they differ ONLY in the population distribution of (gamma, beta). My previous submission of this Bayesian-evidence metric crashed on a pure implementation bug (a scalar np.float64 was indexed with [None, None, :], raising 'IndexError: invalid index to scalar variable' before any statistics were computed), so the idea was never actually tested. This submission is the corrected, numerically hardened implementation: the A_eff prior weight is now a plain scalar added by broadcasting, list-valued rating cells are coerced through a robust parser before any hashing/grouping, all -inf prior cells are handled with log-sum-exp, and the grid (gamma step 0.01 over [0, 2.10]; 41 log-spaced betas; 15 A_eff points) is fine enough relative to the per-subject likelihood widths (sigma_gamma ~ 0.13-0.25 as implied by the ladder's Fisher information ~ 8*7.4 ~ 59 per subject; sigma_logbeta ~ 0.3-0.45). WHY IT DISCRIMINATES. The metric is the Neyman-Pearson-optimal scalar summary of each subject's 96 choices against the two fully specified populations, averaged over subjects. The three dissociating support regions all feed it: (i) RSD places 32.5% of subjects in gamma in (1.35, 2.0), where HT-AEI's prior density is exactly zero — with per-subject gamma likelihood width ~0.15 these subjects are 1.5-4 likelihood units away from ANY HT support point, earning log Bayes factors of roughly +0.5 to +2.4 each; (ii) HT-AEI's 35% tail at gamma in [2.0, 2.10] lies above RSD's hard cap — those subjects earn log Bayes factors of roughly -0.8 to -1.6 (bounded there, not more negative, because HT's narrow tail boxes in BOTH gamma (width 0.10) and beta (log-width 0.24) truncate its own likelihood mass, while RSD's broad boxes do not — this truncation is correctly captured by the grid integration); (iii) HT's mid segment (density 0.833 vs RSD's 0.5, with a 3.2x denser beta box) earns ~-0.3 per mid subject, and HT's low bulk is nearly neutral (+0.2 for RSD because HT's beta box truncates). Expected population values: mean log BF ~ +0.4 under RSD-vgp (mixture of +0.2 low, -0.3 mid, ~+1.1 gap) vs ~ -0.6 under HT-AEI (mixture of +0.2 low, -0.3 mid, -1.6 tail), a between-theory gap of ~1.0 with between-subject SD ~0.5 (RSD) and ~0.85 (HT), giving Welch t ~ 1.0/sqrt((0.25+0.72)/25) ~ 5 at N=25 — far beyond the 2.68 needed at alpha=0.01, with margin even if the effective gamma likelihood is twice as wide as estimated (t ~ 3). The metric is symmetric and falsifiable: it is NEGATIVE on HT-AEI-simulated data, so it is not trivially true for the advocated theory; it is simply the log-likelihood-ratio test between the two completely specified hypotheses, which is the most powerful test that exists for this comparison. PER-SUBJECT VALIDITY. The metric returns the mean of well-defined per-subject log Bayes factors (each computed from that subject's own 96 binomial outcomes), so the pooled point estimate equals the mean of the per-subject values and the between-subject variance is finite and small; identical-vector tie trials (23/24) contribute exactly equal constants to both marginals and cancel, mirrored twins and side bias cancel by symmetry of the likelihood, and the lapse/attenuation parameter is absorbed exactly into A_eff = (1-eps)(2m-1), so no nuisance model misspecification remains beyond the (shared, hence differencing-out) neutral A_eff prior.

Source:
import numpy as np
import pandas as pd

_CACHE = {}

_VALIDITIES = np.array([1.0, 1.0, 1.0,
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                        0.72, 0.63])


def _to_array(x):
    """Coerce a rating cell (list/tuple/ndarray/str) to a float ndarray."""
    if isinstance(x, str):
        x = [t for t in x.strip('[]() ').split(',') if t.strip() != '']
    return np.asarray(list(x), dtype=float)


def metric(data: pd.DataFrame) -> float:
    """
    Mean per-subject log Bayes factor, RSD-vgp vs HT-AEI-r6.

    Both theories share the exact per-trial likelihood
        p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
        E    = sum_j v_j^gamma * (b_j - a_j),
    so the ONLY thing that distinguishes them is the population
    distribution of (gamma, beta). This metric integrates the exact
    binomial likelihood of each subject's 96 choices over each theory's
    stated population prior:
      RSD-vgp : gamma ~ U[0, 2] exactly; beta log-U[1.8, 6.0].
      HT-AEI  : 40% gamma U[0,1]  & beta log-U[2.2, 4.0];
                25% gamma U[1.05,1.35] & beta log-U[2.2, 3.2];
                35% gamma U[2.0,2.1]  & beta log-U[2.2, 2.8].
    with a common neutral uniform prior on A_eff in [0.42, 0.97]
    (the shared range of (1-eps)(2m-1) under both theories).

    Returns the mean over subjects of log m_RSD - log m_HT.
    Positive = data favor RSD-vgp; negative = data favor HT-AEI-r6.
    On a single subject's slice it returns that subject's log Bayes
    factor, so the pooled value is exactly the mean of the per-subject
    values and the between-subject variance is well defined.
    """
    df = data.reset_index(drop=True).copy()

    def _key(x):
        return ''.join([str(int(round(v))) for v in _to_array(x)])

    df['stim'] = (df['option_a_ratings'].apply(_key) + '|' +
                  df['option_b_ratings'].apply(_key))

    stims = sorted(set(df['stim'].tolist()))
    ckey = tuple(stims)

    if ckey not in _CACHE:
        # ---------------- parameter grids ----------------
        g_grid = np.round(np.arange(0.0, 2.1001, 0.01), 10)      # 211 pts
        b_grid = np.exp(np.linspace(np.log(1.8), np.log(6.0), 41))
        A_grid = np.linspace(0.42, 0.97, 15)

        dg = float(g_grid[1] - g_grid[0])
        dlb = float(np.log(b_grid[1]) - np.log(b_grid[0]))
        dA = float(A_grid[1] - A_grid[0])

        # ---------------- population priors ----------------
        # gamma densities (per unit gamma)
        pr_rsd = np.where(g_grid <= 2.0 + 1e-9, 0.5, 0.0)
        pr_ht = (0.40 * np.where(g_grid <= 1.0 + 1e-9, 1.0, 0.0)
                 + 0.25 * np.where((g_grid >= 1.05 - 1e-9) &
                                   (g_grid <= 1.35 + 1e-9),
                                   1.0 / 0.30, 0.0)
                 + 0.35 * np.where((g_grid >= 2.0 - 1e-9) &
                                   (g_grid <= 2.10 + 1e-9), 10.0, 0.0))

        def _logu(lo, hi):
            # log-uniform density on the beta grid
            return np.where((b_grid >= lo - 1e-9) & (b_grid <= hi + 1e-9),
                            1.0 / np.log(hi / lo), 0.0)

        # joint (gamma, beta) grid log-masses for RSD
        dens_rsd = pr_rsd[:, None] * _logu(1.8, 6.0)[None, :]
        lw_rsd = np.full(dens_rsd.shape, -np.inf)
        msk = dens_rsd > 0
        lw_rsd[msk] = np.log(dens_rsd[msk]) + np.log(dg) + np.log(dlb)

        # joint (gamma, beta) grid log-masses for HT (segment-specific beta)
        bw_ht = np.zeros((len(g_grid), len(b_grid)))
        bw_ht[g_grid <= 1.0 + 1e-9] = _logu(2.2, 4.0)
        bw_ht[(g_grid >= 1.05 - 1e-9) & (g_grid <= 1.35 + 1e-9)] = _logu(2.2, 3.2)
        bw_ht[(g_grid >= 2.0 - 1e-9) & (g_grid <= 2.10 + 1e-9)] = _logu(2.2, 2.8)
        dens_ht = pr_ht[:, None] * bw_ht
        lw_ht = np.full(dens_ht.shape, -np.inf)
        msk = dens_ht > 0
        lw_ht[msk] = np.log(dens_ht[msk]) + np.log(dg) + np.log(dlb)

        # common neutral A_eff prior (uniform on the shared range)
        log_a_w = float(np.log(dA / (0.97 - 0.42)))

        # ---------------- per-stimulus likelihood tables ----------------
        first = df.drop_duplicates(subset='stim').set_index('stim')
        a_mat = np.vstack([_to_array(first.loc[s, 'option_a_ratings'])
                           for s in stims])
        b_mat = np.vstack([_to_array(first.loc[s, 'option_b_ratings'])
                           for s in stims])
        D = b_mat - a_mat                                  # (n_u, 14)
        Wg = np.power(_VALIDITIES[:, None], g_grid[None, :])  # (14, n_g)
        E = D @ Wg                                        # (n_u, n_g)

        X = E[:, :, None] * b_grid[None, None, :]          # (n_u, n_g, n_b)
        S = 1.0 / (1.0 + np.exp(-np.clip(X, -60.0, 60.0)))
        core = S - 0.5
        P = 0.5 + A_grid[None, None, None, :] * core[:, :, :, None]
        P = np.clip(P, 1e-9, 1.0 - 1e-9)
        logP = np.log(P)
        log1mP = np.log1p(-P)

        _CACHE[ckey] = (logP, log1mP, lw_rsd, lw_ht, log_a_w)
    else:
        logP, log1mP, lw_rsd, lw_ht, log_a_w = _CACHE[ckey]

    stim_idx = {s: i for i, s in enumerate(stims)}
    n_u = len(stims)

    vals = []
    for sid, sdf in df.groupby('subject_id', sort=True):
        nA = np.zeros(n_u)
        nB = np.zeros(n_u)
        for s, v in sdf.groupby('stim')['response']:
            v = np.asarray(v, dtype=float)
            i = stim_idx[s]
            nb = float(np.sum(v))          # response 1 = chose B
            nB[i] = nb
            nA[i] = float(len(v)) - nb    # response 0 = chose A

        # exact binomial log-likelihood on the (gamma, beta, A_eff) grid
        LL = (np.tensordot(nA, logP, axes=([0], [0])) +
              np.tensordot(nB, log1mP, axes=([0], [0])))   # (n_g, n_b, n_A)

        def _marg(lw):
            tot = LL + lw[:, :, None] + log_a_w
            mx = float(np.max(tot))
            return mx + float(np.log(np.sum(np.exp(tot - mx))))

        vals.append(_marg(lw_rsd) - _marg(lw_ht))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

### RESULTS
- Predicted under pi_7 (simulated): 0.3381 (var=0.3081)
- Predicted under pi_10 (simulated): -0.2872 (var=0.6994)
- Observed on real data: 0.8355 (var=2.6525)

## EXPERIMENT 2 (proposed by pi_10)

### DESIGN
**Validities (n_features=32):** [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.64, 0.76]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 22: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Rationale:** STRUCTURAL SETUP. HT-AEI-r6 and RSD-vgp share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, and nearly identical m/epsilon boxes (A_eff support is NOT a usable lever). The theories are the SAME model with different population SUPPORTS: (i) GAMMA. RSD's two-component mixture is algebraically EXACTLY Uniform[0,2] (density 0.5 throughout, hard edge at 2.0). HT-AEI is a three-segment density: 0.40 on [0,1] (40% low bulk), 0.833 on [1.05,1.35] (25% mid segment), 3.5 on [2.0,2.10] (35% heavy tail), with EMPTY gaps (1.0,1.05) and (1.35,2.0) and a hard cliff at 2.10. (ii) BETA. RSD: log-uniform [1.8,6] independent of gamma. HT: every segment has beta >= 2.2 (low [2.2,4.0], mid [2.2,3.2], tail [2.2,2.8]) -- beta in [1.8,2.2) holds 16.7% of RSD subjects and 0% of HT subjects, and the tail's beta is capped at 2.8. Unlike the prior HT-vs-RSD design ([16]: a coarse 6-rung ladder at 0.585/1.0/1.322/1.585/1.807/2.0 that could see the plateau and the 2.0 edge but NOT the internal structure of any segment), this design is a FINE-GRAINED GAMMA-DENSITY PROFILOMETER: 8 rungs spaced 0.04-0.26 apart that recover the entire piecewise density -- the mid-segment bump, the empty gap, the tail's internal uniform density, and the exact location of the 2.10 cliff.

ARCHITECTURE (32 experts): six cues at validity 1.00 (power weight EXACTLY 1 for every gamma -- a gamma-invariant reference block of weight exactly 6) and 24 cues at 0.50, plus two tuning cues (0.64, 0.76). On every rung trial A is endorsed on the six unit cues (endorsement mass exactly 6 for all gamma) and B on a block of low cues, so E(gamma) = W_block(gamma) - 6 is strictly decreasing with a single engineered sign change at flip point g*: gamma < g* subjects prefer A (the count-sparser side, 6 endorsements); gamma > g* prefer B. The W=6 architecture makes the NATURAL all-0.50 rung grid hit exactly the points this dissociation needs: 13 cues -> g*=1.1155 (inside HT's mid segment); 15 -> 1.3219 (top of mid segment); 18 -> 1.585 and 21 -> 1.807 (inside HT's empty gap); 24 -> 2.000 (the tail's lower edge). Two TUNED rungs probe the tail's interior: 23x0.50 + 0.64 -> g*=2.040; 23x0.50 + 0.76 -> g*=2.082. A beyond-cliff control: 24x0.50 + 0.76 -> g*=2.141, just past HT's 2.10 tail edge (E(2.10) = +0.16 > 0, so every subject of BOTH theories has p >= 0.5 there).

DISSOCIATION 1 -- DENSITY PROFILOMETRY (headline). Because both gamma distributions are mixtures of uniforms, the population inversion rate at a rung equals the survival function S(g*) = P(gamma > g*), and adjacent rungs estimate LOCAL GAMMA DENSITY. Predicted inversion rates, HT vs RSD: R1 (1.1155): 0.545 / 0.442; R2 (1.3219): 0.373 / 0.339; R3 (1.585): 0.350 / 0.207; R4 (1.807): 0.350 / 0.097; R5 (2.000): 0.350 / 0.000; R6 (2.040): 0.210 / 0.000; R7 (2.082): 0.063 / 0.000; R8 (2.141): 0.000 / 0.000. Local densities (survival drop per unit flip-point): [1.1155,1.3219]: 0.833 (HT) vs 0.5 (RSD) -- the mid-segment 1.67x BUMP; [1.3219,1.585]: 0.087 vs 0.5; [1.585,1.807] and [1.807,2.0]: 0.000 vs 0.5 -- the EMPTY GAP (a 3-point FLAT plateau at exactly 0.35 across R3/R4/R5 is a signature RSD's uniform density cannot produce at any parameterization); [2.0,2.04] and [2.04,2.082]: 3.5 vs 0 -- the tail's internal density, ~7x RSD's maximum; [2.082,2.141]: cliff to 0 exactly at HT's 2.10 edge vs RSD's edge already at 2.0. Per-subject crossing points (the rung interval where a subject's psychometric crosses 0.5) give the full histogram: HT predicts 45.5% below 1.1155, 17.2% in (1.1155,1.3219], 2.3% in (1.3219,1.585], 0% in (1.585,2.0], and 35% in (2.0,2.141] (14.0% / 14.7% / 6.3% across the three tail sub-intervals -- a uniform 3.5-density fill); RSD predicts 55.8% / 10.3% / 13.2% / 20.8% / 0% -- a 15x contrast on the gap mass (2.3% vs 34.0%) and a hard 35%-vs-0% contrast above 2.0.

DISSOCIATION 2 -- HARD TAIL CELLS AND THE CLIFF. On R5-R8, E(gamma) >= 0 for EVERY gamma <= 2.0 (R5: E = 24*0.5^gamma - 6 >= 0 with equality only at the measure-zero point gamma = 2; R6-R8 flip above 2.0), and m >= 0.74 > 0.5 keeps p >= 0.5 whenever E >= 0, so NO RSD subject can truly prefer B on any of R5-R8 -- a hard floor at exactly 50%. HT-AEI REQUIRES 35% of subjects below 0.5 on R5 (tail depth: E in [-0.40, 0], p in [0.39, 0.50], mean ~0.44), 21% on R6 (p ~ 0.44-0.50), and 6.3% on R7. The WITHIN-TAIL LINEAR DECLINE 35% -> 21% -> 6.3% -> 0% across [2.0, 2.141] is a quantitative fingerprint no RSD parameterization can approximate (its gamma is hard-capped at 2.0), and rung R8 locates the cliff: inversions at R8 falsify BOTH theories (HT's tail ends at 2.10; RSD's at 2.0), making the design adversarial in both directions. CONDITIONAL COUPLING: under HT, P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (every R4 inverter is a tail subject); under RSD it is 0 HARD (no RSD subject can be below 0.5 on R5). Under HT, P(invert R6 | invert R5) = 0.60; under RSD the conditioning event has probability 0. Within-subject monotonicity (inverting a higher rung implies inverting all lower rungs) holds under both and is a data-quality check.

DISSOCIATION 3 -- BETA SUPPORT VIA EXACT-E CELLS. B1/B2 use only unit cues with exactly canceling shared endorsements (shared endorsements on identical cue indices contribute w*(1-1) = 0), giving E = +1 and +2 EXACTLY for every subject at every gamma. The per-subject ratio r12 = (p_sparse,B2 - 0.5)/(p_sparse,B1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free, lapse-free, and gamma-free. RSD support: r12 in [1.005, 1.322] (beta in [1.8,6]); HT-AEI: r12 <= 1.219 for ALL subjects (beta >= 2.2 in every segment). The region r12 in (1.219, 1.322] is RSD-EXCLUSIVE: 16.7% of RSD subjects, 0% of HT -- any subject credibly there categorically falsifies HT-AEI; a hard zero there falsifies RSD's beta floor. SEGMENT-COUPLED BETA (new readout on this design's deep rungs): among R4/R5 inverters, HT-AEI's tail caps beta at 2.8 (r12 >= 1.121 for all of them), whereas RSD draws beta independently, so ~63% of its high-gamma subjects have beta > 2.8 (r12 < 1.121) -- a population-level split of the inverter group that HT's coupled supports forbid. HT's beta ceiling of 4.0 (vs RSD's 6.0) moves r12 only between 1.005 and 1.037, below per-subject resolution -- honestly acknowledged as one-sided and not relied upon.

POOLED PREDICTIONS (approximate mid-box planning values, p(A-side), HT vs RSD): R1 ~0.47/0.47 (the low/mid mass and inverter depths nearly cancel -- R1 is a shape probe, not a pooled probe; honestly acknowledged); R2 ~0.57/0.64; R3 ~0.61/0.72; R4 ~0.63/0.76; R5 ~0.70/0.80; R6 ~0.72/0.79; R7 ~0.73/0.82; R8 ~0.78/0.83; B1 ~0.80/0.80; U ~0.85/0.85; ties 0.50/0.50. Pooled differences of 0.05-0.13 on R2-R7 are corroborating; the decisive evidence is SUBJECT-LEVEL population structure.

CONTROLS. U/Um (all-0 vs all-1, mirrored) saturate every subject (|E| = S(gamma) >= 12.5 at gamma = 2.1, beta >= 1.8), reading out A_eff = 2*(p_sparse - 0.5) to within 0.0002 -- required to normalize the beta probes and rung psychometrics. Uc is a CONFLICT-STYLE saturated twin with the OPPOSITE sparse side (A endorsed on 31 cues, B on a single 0.50 cue; |E| >= 11): p_sparse(Uc) must equal p_sparse(U), which detects any 'pick the empty bar' heuristic while its reversed side assignment cancels side bias against U. T1 (identical non-trivial vectors) verifies both theories' exact-50% prediction and catches key bias. Every rung and beta cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (rungs) or sparse side (B-cells, U, Uc). The ladder itself rules out pure counting (counting predicts flat ~100% preference for the 6-endorsement side on every rung) and conventional pro-endorsement polarity (flat ~0%). All informative trials carry nonempty, distinct bars on both sides, satisfying the locked-prompt comparison requirement.

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; 8 observations per pooled mirrored cell per subject; 64 rung observations per subject. Requires N >= 100 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior (RSD: gamma exactly U[0,2], beta log-U[1.8,6] independent; HT-AEI: the three-segment gamma mixture with segment-coupled beta boxes), plus simulation-based model recovery.

DECISIVE QUANTITATIVE SIGNATURES: (a) any subject credibly below 50% on R5, R6, or R7 categorically falsifies RSD-vgp (E >= 0 over its entire gamma box; hard floor at exactly 0.5), while HT-AEI requires 35%/21%/6.3% of subjects there; (b) inversion-rate survival curve over g* = 1.1155/1.3219/1.585/1.807/2.000/2.040/2.082/2.141: HT 0.545/0.373/0.350/0.350/0.350/0.210/0.063/0 (flat 0.35 plateau on [1.35,2.0], linear 3.5-density decline inside the tail, cliff at 2.10) vs RSD 0.442/0.339/0.207/0.097/0/0/0/0 (constant 0.5 density, hard edge at 2.0); (c) per-subject crossing-point histogram: gap mass (1.3219,2.0] at 2.3% (HT) vs 34.0% (RSD), tail mass (2.0,2.141] at 35% (HT) vs 0% hard (RSD), mid-bump density 0.833 vs 0.5 on [1.1155,1.3219]; (d) P(p(R5) < 0.5 | p(R4) < 0.5) = 1.0 (HT) vs 0 hard (RSD); (e) per-subject r12 in (1.219, 1.322]: ~16.7% (RSD) vs 0% hard (HT-AEI); (f) among R4/R5 inverters, r12 >= 1.121 for all (HT tail beta <= 2.8) vs ~63% below 1.121 (RSD independent supports); (g) within-subject monotone flip pattern (inverting R5 implies inverting R4..R1); (h) inversions at R8 (g* = 2.141) falsify BOTH theories -- HT's tail edge is at 2.10, RSD's at 2.0; (i) U = Uc in sparse-side rate (empty-bar control) and identical-vector ties at exactly 50% (key-bias control).

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Both theories share the identical per-trial core p(B) = 0.5 - A_eff*(sigmoid(beta*E(gamma)) - 0.5) with E(gamma) = sum_j v_j^gamma*(b_j - a_j); they differ ONLY in the population distribution of gamma. HT-AEI-r6 puts 35% of subjects in a heavy tail on gamma in [2.0, 2.10] and leaves (1.35, 2.0) essentially EMPTY (2.3%), while RSD-vgp spreads 35% of subjects uniformly over [1.3, 2.0] with a hard cap at 2.0. The rung ladder was built so that per-subject gamma is recoverable from the 64 rung observations (flip points g* = 1.12/1.32/1.59/1.81/2.00/2.04/2.08/2.14; the deep rungs R5-R8 straddle the 2.0 boundary with 0.04 spacing, localizing a tail subject's gamma to within ~0.05-0.08).

The metric therefore fits the (shared, correctly specified) model per subject by grid MLE over gamma in [0, 2.5], beta in [1.6, 7.5], A_eff in [0.25, 0.97], converts the profile likelihood into a flat-prior posterior over gamma, and scores each subject as s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95). The first term captures mass above RSD's hard gamma ceiling (HT tail subjects score ~ +0.4 to +1.0; RSD can only place ~2.5% true mass there plus a thin estimation smear), and the second term captures mass in HT's empty gap (1.40, 1.95), where RSD concentrates ~27% of its population (those subjects invert rungs R2-R4 but not R5, scoring ~ -0.9) but HT has almost none (its mid segment ends at 1.35). The statistic is continuous, so boundary subjects contribute partial credit instead of all-or-nothing noise, keeping per-subject variance moderate (~0.15-0.25) while the between-theory mean gap is large: expected pooled value ~ +0.22 to +0.30 under HT-AEI-r6 versus ~ -0.19 to -0.24 under RSD-vgp, giving Welch t ~ 3.5-4 at N = 25 (p < 0.002). The two-sided construction also makes the metric adversarial in both directions: it rewards HT only for genuinely above-2.0 precision AND punishes it if the (1.4, 1.95) gap fills in, so it is a real test of the population structure, not a tautology. It works identically on a single subject's slice (returns that subject's score), so between-subject variance is well defined.

Source:
def metric(data: pd.DataFrame) -> float:
    # Crossing-point population statistic.
    # Per subject: fit the shared core model p(B) = 0.5 - A*(sigmoid(beta*E(gamma)) - 0.5),
    # E(gamma) = sum_j v_j^gamma * (b_j - a_j), by grid MLE over (gamma, beta, A),
    # then score s = P(gamma > 1.95) - P(1.40 < gamma <= 1.95) under the flat-prior
    # profile-likelihood posterior over gamma. Metric = mean of s over subjects.
    a_list = data["option_a_ratings"].to_numpy()
    b_list = data["option_b_ratings"].to_numpy()
    A_mat = np.vstack([np.asarray(r, dtype=float) for r in a_list])
    B_mat = np.vstack([np.asarray(r, dtype=float) for r in b_list])
    y = data["response"].to_numpy(dtype=float)
    subj = data["subject_id"].to_numpy()

    n_feat = A_mat.shape[1]
    if n_feat >= 8:
        # design validities: six 1.00 cues, twenty-four 0.50 cues, 0.64, 0.76
        V = np.array([1.0] * 6 + [0.5] * (n_feat - 8) + [0.64, 0.76], dtype=float)
    else:
        V = np.ones(n_feat, dtype=float)

    gam = np.linspace(0.0, 2.5, 126)
    bet = np.geomspace(1.6, 7.5, 8)
    amps = np.linspace(0.25, 0.97, 9)

    Wg = V[None, :] ** gam[:, None]  # (G, F) validity-power weights per gamma

    scores = []
    for s_id in pd.unique(data["subject_id"]):
        m = subj == s_id
        D = (B_mat[m] - A_mat[m]).astype(float)  # (n, F)
        yy = y[m]  # 1 = chose B
        E = D @ Wg.T  # (n, G) raw evidence for A over B
        X = E[:, :, None] * bet[None, None, :]  # (n, G, B)
        d = 0.5 * np.tanh(0.5 * X)  # sigmoid(X) - 0.5, stable, in [-0.5, 0.5]
        ll_prof = np.full(gam.shape[0], -np.inf)  # profile log-likelihood over gamma
        for a in amps:
            P = np.clip(0.5 - a * d, 1e-9, 1.0 - 1e-9)  # p(choose B)
            ll = np.sum(
                yy[:, None, None] * np.log(P)
                + (1.0 - yy)[:, None, None] * np.log(1.0 - P),
                axis=0,
            )  # (G, B)
            ll_prof = np.maximum(ll_prof, ll.max(axis=1))
        w = np.exp(ll_prof - ll_prof.max())
        w = w / w.sum()  # posterior over gamma (flat prior on [0, 2.5])
        p_hi = float(w[gam > 1.95].sum())          # tail mass above RSD's hard cap
        p_gap = float(w[(gam > 1.40) & (gam <= 1.95)].sum())  # HT's empty-gap mass
        scores.append(p_hi - p_gap)
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_7 (simulated): -0.2078 (var=0.1272)
- Predicted under pi_10 (simulated): 0.1782 (var=0.1651)
- Observed on real data: -0.2077 (var=0.1170)

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
**Predicted under pi_10:** 0.1513 (var=0.0077)

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
**Predicted under pi_10:** 0.1714 (var=0.0079)

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
**Predicted under pi_10:** -0.7750 (var=0.0581)

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
**Predicted under pi_10:** -0.5479 (var=0.0211)

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
**Predicted under pi_10:** 0.5701 (var=0.0179)

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
**Predicted under pi_10:** -0.6917 (var=0.0411)

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
**Predicted under pi_10:** 0.7067 (var=0.0235)

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
**Predicted under pi_10:** 0.3629 (var=0.0063)

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
**Predicted under pi_10:** -1.1568 (var=1.5167)

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
**Predicted under pi_10:** 0.8791 (var=0.0271)

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
**Predicted under pi_10:** 114.0736 (var=12.7298)

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
**Predicted under pi_10:** -0.2310 (var=0.5023)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 6.0286 (var=0.5984)
**Predicted under pi_7:** 5.3165 (var=0.3027)
**Predicted under pi_10:** -9.6424 (var=0.6044)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.2983 (var=0.0006)
**Predicted under pi_7:** 0.0990 (var=0.0001)
**Predicted under pi_10:** 0.1215 (var=0.0002)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # ==================================================================
    # Population log Bayes factor: log P(data | RSD-vgp) - log P(data | SGAI-r7).
    # Both theories share the identical per-trial choice kernel
    #     p(A) = 0.5 + A_eff * (sigmoid(beta * E) - 0.5),
    #     E = sum_j v_j^gamma * (b_j - a_j),  A_eff = (1-eps)(2m-1),
    # and differ ONLY in their population priors over (gamma, beta, eps, m)
    # plus SGAI's per-trial gamma jitter (prob ~0.11, +/-0.4 around the
    # habitual gamma, capped at [0, 2.2]).  So the marginal likelihood
    # under each theory's stated population prior isolates exactly the
    # claims that separate them:
    #   RSD-vgp: gamma ~ U[0,2] exactly; beta log-U[1.8,6]; eps U[.03,.11];
            #   m U[.74,1]; no jitter.
    #   SGAI-r7: 80% majority (gamma 20%[.8,1.3]/48%[1.3,1.7]/24%[1.7,2.0]/
    #   8%(2.0,2.2]; beta log-U[1.8,6]; eps U[.03,.10]; m U[.8,.98]) plus
    #   20% low-precision cluster (gamma U[0,1]; beta log-U[1.2,1.8];
    #   eps U[.15,.25]); per-trial gamma jitter.
    # The metric integrates the shared kernel over these priors on a grid
    # (SGAI's jitter is folded into the per-trial kernel as a 0.89/0.11
    # mixture over the +/-0.4 gamma window) and returns the total log
    # likelihood ratio.  On a pooled DataFrame this is the dataset log Bayes
    # factor (sum of independent per-subject contributions); on a single
    # subject's slice it is that subject's log-BF contribution.
    # ==================================================================
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.0

    v = np.array([1.0, 1.0, 1.0,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.6])
    nf = len(v)

    def _to_arr(x):
        if isinstance(x, np.ndarray):
            a = x.astype(float)
            if a.size == nf:
                return a
        try:
            a = np.asarray(list(x), dtype=float)
            if a.size == nf:
                return a
        except (TypeError, ValueError):
            pass
        s = str(x).strip().strip("[]()")
        return np.array([float(tok) for tok in s.replace(",", " ").split()])

    A = np.vstack([_to_arr(r) for r in df["option_a_ratings"]])
    B = np.vstack([_to_arr(r) for r in df["option_b_ratings"]])
    resp = df["response"].to_numpy(dtype=float)  # 0 = chose A, 1 = chose B

    # ---- unique trial types ----------------------------------------
    keys = [(tuple(A[i]), tuple(B[i])) for i in range(len(df))]
    index_of = {}
    type_idx = np.empty(len(keys), dtype=int)
    for i, k in enumerate(keys):
        if k not in index_of:
            index_of[k] = len(index_of)
        type_idx[i] = index_of[k]
    types = list(index_of.keys())
    Amat = np.array([np.array(t[0], dtype=float) for t in types])
    Bmat = np.array([np.array(t[1], dtype=float) for t in types])
    T = len(types)

    subj_codes = pd.factorize(df["subject_id"])[0]
    S = int(subj_codes.max()) + 1
    nA = np.zeros((S, T))
    nB = np.zeros((S, T))
    np.add.at(nA, (subj_codes, type_idx), (resp == 0).astype(float))
    np.add.at(nB, (subj_codes, type_idx), (resp == 1).astype(float))

    # ---- master parameter grids ------------------------------------
    gmaster = np.round(np.arange(0.0, 2.2 + 1e-9, 0.05), 6)       # 45 gammas
    bmaster = np.exp(np.linspace(np.log(1.2), np.log(6.0), 48))  # 48 betas

    # E[g, t] = sum_j v_j^g * (B_tj - A_tj)
    Vg = v[None, :] ** gmaster[:, None]
    E = Vg @ (Bmat - Amat).T

    def _sigmoid(x):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    # s[g, b, t] = sigmoid(b * E) - 0.5   (kernel without jitter)
    X = bmaster[None, :, None] * E[:, None, :]
    s_master = _sigmoid(X) - 0.5

    # SGAI per-trial gamma jitter folded into the kernel:
    # s_jit(g) = 0.89*s(g) + 0.11*mean_{gt in [g-0.4, g+0.4] cap [0,2.2]} s(gt)
    delta = 0.11
    s_jit = np.empty_like(s_master)
    for gi in range(len(gmaster)):
        g = gmaster[gi]
        lo = int(np.searchsorted(gmaster, max(0.0, g - 0.4) - 1e-9))
        hi = int(np.searchsorted(gmaster, min(2.2, g + 0.4) + 1e-9))
        if hi <= lo:
            hi = lo + 1
        s_jit[gi] = ((1.0 - delta) * s_master[gi]
                     + delta * s_master[lo:hi].mean(axis=0))

    def _logml(smat, g_idx, g_w, b_idx, b_w, alphas, a_w):
        # marginal log-likelihood for every subject under one population
        # component given by (gamma grid/weights, beta grid/weights,
        # A_eff grid/weights); smat supplies the (possibly jittered) kernel.
        s_sel = smat[np.ix_(g_idx, b_idx)]                 # (ng, nb, T)
        p = 0.5 + alphas[None, None, :, None] * s_sel[:, :, None, :]
        p = np.clip(p, 1e-12, 1.0 - 1e-12)                 # (ng, nb, na, T)
        LP = np.log(p).reshape(-1, T)
        LQ = np.log1p(-p).reshape(-1, T)
        ll = nA @ LP.T + nB @ LQ.T                         # (S, ncombo)
        w = (g_w[:, None, None] * b_w[None, :, None]
             * a_w[None, None, :]).reshape(-1)
        ll = ll + np.log(np.maximum(w, 1e-300))[None, :]
        m = np.max(ll, axis=1)
        return m + np.log(np.sum(np.exp(ll - m[:, None]), axis=1))

    # ---- RSD-vgp population ----------------------------------------
    g_idx = np.arange(0, 41)                    # gamma ~ U[0, 2]
    g_w = np.full(41, 1.0 / 41)
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.11, 4)
    mm = np.linspace(0.74, 1.00, 7)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_R = _logml(s_master, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 low-precision cluster (weight 0.2) -----------------
    g_idx = np.arange(0, 21)                    # gamma ~ U[0, 1]
    g_w = np.full(21, 1.0 / 21)
    b_idx = np.arange(0, 12)                    # beta log-U[1.2, 1.8]
    b_w = np.full(12, 1.0 / 12)
    eps = np.linspace(0.15, 0.25, 4)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_L = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    # ---- SGAI-r7 majority cluster (weight 0.8) ----------------------
    g_idx = np.concatenate([np.arange(16, 26), np.arange(26, 34),
                            np.arange(34, 40), np.arange(40, 45)])
    g_w = np.concatenate([np.full(10, 0.20 / 10), np.full(8, 0.48 / 8),
                          np.full(6, 0.24 / 6), np.full(5, 0.08 / 5)])
    b_idx = np.arange(12, 48)                   # beta log-U[1.8, 6]
    b_w = np.full(36, 1.0 / 36)
    eps = np.linspace(0.03, 0.10, 3)
    mm = np.linspace(0.80, 0.98, 5)
    al = ((1.0 - eps)[None, :] * (2.0 * mm[:, None] - 1.0)).reshape(-1)
    lml_M = _logml(s_jit, g_idx, g_w, b_idx, b_w, al,
                   np.full(al.size, 1.0 / al.size))

    lml_S = np.logaddexp(np.log(0.2) + lml_L, np.log(0.8) + lml_M)

    llr = lml_R - lml_S   # per-subject log Bayes factor (RSD vs SGAI)
    return float(np.sum(llr))
```

**Observed (real) value:** 12.9184 (var=1.0896)
**Predicted under pi_7:** 23.7874 (var=2.2777)
**Predicted under pi_10:** 7.5133 (var=1.5781)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Fixed design validities (15 experts, design-time constants).
    V = np.array([1.0, 1.0, 1.0, 0.8, 0.75, 0.65, 0.6,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().lstrip('[').rstrip(']')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            arr = arr.ravel()
        return arr

    # Raw (unnormalized) validity weights at a low-gamma and a high-gamma
    # regime. Under the shared anti-endorsement core, a trial's evidence is
    # E(gamma) = sum_j v_j^gamma (b_j - a_j); the two critical 'flip' rungs
    # of this design are exactly the trials where sign(E) changes between
    # these two regimes (flip points gamma* ~ 0.73 and ~ 1.0).
    w_lo = np.power(V, 0.5)
    w_hi = np.power(V, 1.5)

    # Bayesian (Beta) regularization of the sparse-side rate: 96 pseudo-
    # trials centered at 0.5. Per subject only 16 critical trials exist
    # (binomial SE ~ 0.12), so the raw per-subject rate is unstable; the
    # shrinkage stabilizes individual-subject estimates while leaving the
    # pooled estimate (16 x 25 = 400 critical trials) essentially
    # unchanged.
    KAPPA = 96.0
    M0 = 0.5

    k = 0  # sparse-side choices on flip trials
    n = 0  # flip trials seen

    a_col = data['option_a_ratings'].values
    b_col = data['option_b_ratings'].values
    r_col = data['response'].values

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != V.size or b.size != V.size:
            continue
        e_lo = float(np.sum(w_lo * (b - a)))
        e_hi = float(np.sum(w_hi * (b - a)))
        if e_lo == 0.0 or e_hi == 0.0:
            continue  # tie trial, uninformative
        if (e_lo > 0.0) == (e_hi > 0.0):
            continue  # not a flip trial (sign of evidence is gamma-invariant)
        # Flip trial: low-gamma subjects prefer one side, high-gamma
        # subjects the other. The low-gamma-preferred side is the one with
        # the smaller raw validity weight (the 'count-sparser' side).
        n += 1
        sparse_is_a = (e_lo > 0.0)
        chose_a = (int(float(r_raw)) == 0)
        if chose_a == sparse_is_a:
            k += 1

    if n + KAPPA <= 0:
        return float(M0)
    return float((k + KAPPA * M0) / (n + KAPPA))
```

**Observed (real) value:** 0.4335 (var=0.0024)
**Predicted under pi_7:** 0.5335 (var=0.0013)
**Predicted under pi_10:** 0.4754 (var=0.0011)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 18 experiments, pi_7 (RSD-vgp) is the clearly superior theory. Head-to-head, pi_7 is closer to the observed value on Experiments 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, and 15 (12 experiments), while pi_10 (HT-AEI-r6) is closer only on 6, 9, 17, and 18, and mostly by marginal amounts. More importantly, the losses are qualitatively different. pi_10's failures are catastrophic and structural: on Experiment 2 (its own fine-grained gamma profilometer) the observed crossing-point statistic is -0.2077, essentially exactly pi_7's prediction (-0.2078) and on the opposite side of pi_10's (+0.1782) \u2014 the real population shows RSD's uniform, gap-filled gamma density with no 0.35 plateau and no narrow [2.0, 2.10] tail cliff. On Experiment 15, pi_10 predicts -9.64 where the data show +6.03 (a sign flip driven by its segment-coupled beta caps), and on Experiment 1 the log Bayes factor favors RSD at +0.84, beyond even RSD's own simulated +0.34. The three-segment support with empty gaps and an edge-hugging tail is refuted. By contrast, pi_7's misses are graded and point in a consistent direction: it underpredicts the apparent above-2.0 gamma mass on Experiment 16 (0.099 vs 0.298 observed), overpredicts the RSD-vs-SGAI Bayes factor on Experiment 17 (23.8 vs 12.9), overpredicts the low-gamma-side flip rate on Experiment 18 (0.534 vs 0.434), and overpredicts the log BFs on Experiments 13 and 14. Notably, Experiments 2 and 16 superficially conflict (Exp 2 says the gamma density is essentially U[0,2] with little mass above 1.95; Exp 16's no-jitter fitting model assigns ~43% raw posterior mass above 2.0). The one mechanism that reconciles this pattern is not a bigger static tail \u2014 which Exp 2 forbids \u2014 but per-trial variability: stochastic attentional fluctuation in the effective validity exponent. Jitter flattens the flip-trial psychometric functions, which a fixed-gamma measurement model mis-attributes to gamma values above 2.0 (inflating Exp 16's tail estimate and Exp 18's flip rate), softens per-subject likelihood concentration (pulling Exps 13/14/17 BFs down toward observed), while leaving the population-level uniform-gap signature of Exp 2 and the bulk fits of Experiments 3-12 essentially intact. Since pi_10 is degenerate rather than merely misparameterized, and pi_7, though victorious, has a systematic residual on precisely the tail/jitter-sensitive experiments, the right move is to replace pi_10 with a new theory that keeps the validated shared core and RSD-like uniform bulk but adds the jitter/extended-support mechanism \u2014 a genuine competitor to pi_7 on its weak spots \u2014 rather than merely re-fitting either existing model.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_10 with a new theory, 'Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI)'. Per-trial core is unchanged from the validated shared kernel: endorsements are evidence AGAINST the carrying option, E = sum_j v_j^gamma (b_j - a_j) with RAW (unnormalized) validity-power weights, p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, A_eff = (1-eps)(2m-1) concentrated near 0.69, ties exactly 50%. The population departs from both predecessors in three calibrated ways. (1) GAMMA SUPPORT: a single continuous uniform distribution on [0, 2.2] \u2014 no empty gaps and no narrow tail segment (both directly refuted by Exps 2 and 16: the gap must be filled at constant density, and any cliff at 2.10 or 2.0 is falsified), with the slight extension past 2.0 providing the modest true tail mass that Exp 18's flip rate (0.4335 < RSD's 0.5335) and Exp 17's intermediate Bayes factor require, while keeping Exp 2's crossing statistic near its observed -0.21. (2) BETA: log-uniform on [1.8, 6.0], independent of gamma \u2014 no segment-coupled beta boxes (the direct cause of pi_10's Exp 15 catastrophe). (3) THE NEW MECHANISM: per-trial stochastic validity-attention jitter \u2014 with probability delta ~ 0.10-0.13 per trial, the effective gamma for that trial is redrawn uniformly from [gamma_habit - 0.35, gamma_habit + 0.35] clipped to [0, 2.4]. This is the key competitor to pi_7: it explains why fixed-gamma measurement models (Exp 16's tail estimator, Exp 18's flip rungs, Exp 17's BF) see apparent mass above 2.0 and softened per-subject likelihoods in real data, while population-level density probes (Exp 2) still recover an essentially uniform [0, 2] shape, and why Exps 13/14's Bayes factors are smaller in real data than pi_7's noise-free simulation predicts. Falsifiable signatures: near-uniform recovered gamma density with no plateau on rung ladders; apparent tail-mass estimates that INFLATE on shallow-slope designs (Exp 16 ~ 0.25-0.35) while crossing-point statistics stay near -0.2 (Exp 2); flip-trial rates pulled 0.05-0.10 below RSD's predictions on gamma* ~ 0.7-1.0 ladders (Exp 18 ~ 0.43-0.47); per-subject log BFs against fixed-gamma competitors attenuated ~30-50% relative to RSD's predictions (Exps 13, 17); steep-slope posterior mass still high (Exp 12 ~ 0.85-0.92) since jitter is modest. Keep delta small enough to preserve the excellent Exps 3-12 fits that both theories currently achieve through the shared core."
}
```

## Usage

```json
{
  "input_tokens": 65302,
  "output_tokens": 9468,
  "reasoning_tokens": 8017,
  "provider": "Sail Research"
}
```
