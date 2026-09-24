# Round 8 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_7` — SURVIVED ✓

**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Rationale:** RSD-vgp keeps the empirically validated GSP skeleton (raw-weight additive anti-endorsement integration, which landed Exps 2/4/6/8 nearly exactly) and changes exactly the two dimensions the residuals implicated, while deliberately deviating from one piece of the arbiter prescription with data-driven justification.

(1) PRECISION (accepted, core of the proposal): beta_raw is log-uniform over [1.8, 6.0]. Since a subject's effective normalized slope is beta_raw * sum(v^gamma) and sum(v^gamma) >= ~7.4 for any gamma in [0,2] on 8-12 cue validity vectors, EVERY subject's slope exceeds ~13, directly targeting Experiment 10's observed P(s>9.5) = 0.911 (GSP under-predicted at 0.785 because its low-beta corner sat at s ~ 9.1). The same steepening pushes Experiment 9's LLR more negative: the metric's GSP reference mixture caps raw slope at 3.2 and gamma at 1.5, so the ~50% of RSD subjects with beta > 3.2 (and ~25% with gamma > 1.5) are out-of-grid for the reference and fit it disproportionately worse than the flatter RFI reference — moving the LLR from GSP's -0.65 toward the observed -1.76. Steeper slopes also raise Experiment 5's evidence-usage slope (GSP 0.541 vs obs 0.594) and sharpen Experiment 3's depth signature toward the observed -0.717.

(2) VALIDITY GATING (accepted): gamma is dispersed over [0, 2] via a mixture (65% on [0, 1.3], 35% steep on [1.3, 2.0]), exceeding both RFI's cap of 1 and GSP's 1.5. This supplies the steep-gating subpopulation for validity-conflict cells and is one of the three falsifiable fronts against GSP.

(3) DEVIATION FROM THE ARBITER ON m/eps, with rationale: the arbiter prescribed tightening and raising m to [0.85, 1.0] (mass above 0.9) with eps in [0.02, 0.08], claiming GSP's box 'slightly under-shoots' Experiment 7. That claim is factually inverted: GSP OVER-shoots Experiment 7 (predicted 0.754 vs observed 0.704) and matches Experiment 8 exactly (0.3467 vs 0.3467). On near-unanimous trials the per-subject saturated extremity is exactly A_eff = (1-eps)(2m-1), so raising m would push Experiment 7 to ~0.80 (normalized error ~0.39, worse than GSP's 0.205) while gaining nothing elsewhere that beta does not already deliver. I therefore set E[A_eff] ~ 0.69 via m in [0.74, 1.0] and eps in [0.03, 0.11], which simultaneously: (i) moves Experiment 7 toward ~0.70-0.73 and Experiment 8 toward ~0.34-0.35 (note the real data satisfy Exp7 ~ 2*Exp8, i.e. saturation consistency, which this box reproduces); (ii) moves Experiment 6's dense-vs-sparse TTB contrast to ~ -0.69 (obs -0.676); and (iii) the extra (1-m) mass exactly compensates the steeper sigmoid in Experiment 2's conflict-hit rate, keeping P(tally winner) near the observed 0.18 instead of collapsing toward 1-m of a high-m population.

The central insight is a slope/asymptote decomposition: Experiments 5, 9, and 10 are slope-sensitive and demand steep normalized psychometric slopes; Experiments 2, 6, 7, 8 are asymptote-sensitive and demand moderate extremity. A single shallow-slope/high-extremity family (RFI) or steep-slope/high-extremity family (the arbiter's tight-m variant) cannot satisfy both; RSD-vgp's steep-beta, moderate-attenuation, wide-gamma family can. It remains falsifiable against GSP on (i) the slope posterior distribution (RSD: essentially all subjects above 9.5; GSP: ~0.78), (ii) the attenuation distribution (RSD's E[A_eff] ~ 0.69 vs GSP's ~0.70 with wider spread), and (iii) steep-gamma behavioral signatures on validity-conflict cells (gamma up to 2 vs GSP's 1.5 cap).

**Parameters:**
  - `gamma_u`: `[0, 1]`
  - `log_beta`: `[0.588, 1.792]`
  - `epsilon`: `[0.03, 0.11]`
  - `m`: `[0.74, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


### slot 2 — `pi_10` — KILLED ✗

**Description:** Heavy-Tail Anti-Endorsement Integration, revision 6 (HT-AEI-r6). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69 — the 'steep normalized slope x moderate asymptote' family. The population is a three-segment mixture with SEGMENT-SPECIFIC PRECISION, all segments now at moderate-to-high precision (no soft-beta segment remains, since weakly identified slopes were shown to leak Exp 10/14 posteriors below threshold and to donate RFI-ward on Exp 9): (1) a 40% low bulk on gamma in [0, 1.0] with beta log-uniform on [2.2, 4.0] — the validated shared-core bulk, concentrated below the ~0.73/1.0 flip points of Exp 16's ladder, with the beta floor raised so every low-bulk subject's normalized slope is sharply identified; (2) a 25% mid segment on gamma in [1.05, 1.35] with beta log-uniform on [2.2, 3.2] — a band where RSD's uniform density (0.5) beats SGAI's (0.4) and EPAI's (0.375), sitting fully inside GSP's gamma box with GSP-compatible beta, so these subjects donate RSD-ward on Exp 15 while pulling Exp 11 and Exp 9 toward their observed values; (3) a 35% heavy tail on gamma in [2.0, 2.10] with beta log-uniform on [2.2, 2.8] — an extreme edge-hug of the RSD gamma boundary (average distance <= 0.05) at sharpened precision, so tail subjects' gamma posteriors concentrate just above 2.0 (feeding Exp 14), their slopes stay above the s>9.5 threshold (feeding Exp 10), and their near-deterministic past-flip-point behavior pulls Exp 16 down toward 0.4335. No per-trial gamma jitter. Falsifiable signatures: ~35% of subjects carry gamma within 0.10 of 2.0 with per-subject posteriors extending smoothly past 2.0; Exps 13/15 small and positive; Exp 11 in the 50-100 range; Exp 14 at or above 0.21; Exp 16 in [0.42, 0.45]; Exp 10 at or above 0.85.

**Rationale:** This is a minimal-diff edit of the ACCEPTED iter-6 base (HT-AEI-r5) implementing the iter-6 critic's coordinated package verbatim; the kernel, A_eff concentration near 0.69, no-jitter rule, exact-tie behavior, and the 40% low bulk's gamma support [0,1] are all untouched — only the segment shares, the tail support width, and the segment-specific beta ranges change. (i) LOW-BULK BETA FLOOR RAISED 1.8 -> 2.2 (logU[2.2, 4.0]): this is the critic's pre-specified contingency for the Exp 10 breach (0.8133 vs observed 0.9113, guardrail >= 0.84). The low bulk sits at gamma <= 1 where slopes are steep anyway; raising the floor only sharpens per-subject slope identification, moving Exp 10 back toward pi_7's 0.916 while being neutral in kind for every other metric. (ii) TAIL RE-SHARPENED: share trimmed 42% -> 35%, support narrowed [2.0, 2.12] -> [2.0, 2.10] (average edge distance <= 0.05), beta raised from logU[1.9, 2.4] to logU[2.2, 2.8]. The iter-6 data showed the soft tail beta was net-negative: it bought Exp 13 edge-cheapness but cratered Exp 14 (0.2201 -> 0.1852, below even the accepted 0.21 floor) by spreading tail posteriors toward the diffuse baseline, eroded Exp 10, and pushed Exp 9 from -1.45 to -0.84 (away from observed -1.76) because soft-slope tail subjects become RFI-fittable. Sharper beta reverses all three: tail posteriors concentrate just above 2.0 (Exp 14 recovers toward 0.22-0.26 despite the smaller fraction), slopes clear the s>9.5 threshold (Exp 10), tail subjects become near-deterministic past the ~0.73/1.0 flip points pulling Exp 16 down from 0.4643 toward 0.4335, and the RFI-fittable tail mass shrinks (Exp 9 back toward -1.76). The Exp 13 risk from a sharper likelihood at the 0.05 edge distance is offset by the ~17% fraction trim plus the enlarged mid segment's positive donations; target zone [0, +8] around the observed +6.03, with the critic's compromise (tail beta logU[2.1, 2.6]) held in reserve. (iii) MID ENLARGED 18% -> 25% on [1.05, 1.35], beta logU[2.2, 3.2] unchanged: these subjects sit inside GSP's gamma box with GSP-compatible beta, pulling Exp 11 down from 108.7 toward the observed 52.6 and Exp 9 GSP-ward; they remain RSD-donating on Exp 15 (density 0.5 vs SGAI's 0.4), so Exp 15 (+21.8, overshooting +12.9) should drift down a few units, helped further by the tail trim. The critic's guardrail — if Exp 15 lands above ~+28, hold the mid at 20% — is noted as the fallback. (iv) Exp 1 (0.115 vs 0.158) is deliberately NOT chased: it has been mildly low across all accepted iterations and every past attempt to move it regressed something bigger. Expected landing zone per the critic's verification checklist: Exp 10 >= 0.85, Exp 16 in [0.42, 0.45], Exp 13 in [0, +8], Exp 15 in [+8, +25], Exp 14 >= 0.21, Exp 11 <= 100, Exp 9 <= -1.2, Exp 12 within +/-0.1 of -0.414, with Exps 2-8 remaining at their confirmed shared-core level. The falsifiable core is unchanged in kind: roughly 35% of subjects carry validity exponents within 0.10 of the RSD edge at 2.0 with smoothly extending posteriors, and the population's precision is segment-specific — now uniformly sharp, since the iter-6 data proved that soft precision anywhere in the population costs more (Exps 9, 10, 14) than it buys (Exp 13).

**Parameters:**
  - `gamma_u`: `[0, 1]`
  - `log_beta_low`: `[0.788, 1.386]`
  - `log_beta_mid`: `[0.788, 1.163]`
  - `log_beta_tail`: `[0.788, 1.030]`
  - `epsilon`: `[0.03, 0.10]`
  - `m`: `[0.75, 1.0]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
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
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## Replacement

### `pi_11` → slot 2 (via `new_theory`)

**Description:** Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI, clip-restoration pass). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity-power weights: E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical rating vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with A_eff = (1-eps)(2m-1) near 0.69 (steep normalized slope x moderate asymptote). The population: habitual gamma ~ U[0, 2.2] (single continuous uniform — no gaps, no cliffs, no narrow tail segment), beta log-uniform on [1.8, 6.0] independent of gamma. The novel mechanism is per-trial stochastic validity-attention jitter: with per-trial probability delta ~ 0.13-0.16, the effective gamma for that trial is redrawn uniformly from a SYMMETRIC window [gamma_habit - 0.50, gamma_habit + 0.50] clipped to [0, 2.4] — the arbiter's own prescribed geometry. The critical factorization this theory embodies: TRUE habitual mass above 2.2 is poison (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment and destroys Exp 18's negative offset), while per-trial EXCURSIONS reaching 2.4 from habits below 2.2 are safe and are the only channel that moves the apparent-tail estimator (Exp 14) above 0.17. There is NO per-trial log-beta jitter: five passes of loop evidence establish that any per-trial precision noise flattens per-subject gamma posteriors toward the estimator's EB prior mean and caps the tail estimate at ~0.12, while gamma-only jitter at delta ~0.14 simultaneously satisfies the BF softening (Exps 15/9/12) and the flip-rate anchor (Exp 16). Falsifiable signatures: apparent tail-mass estimates ~0.17-0.20 on shallow-slope designs; positive RSD-vs-EPAI log BF ~[4, 10]; flip rates in [0.43, 0.45]; crossing statistic in [-0.15, -0.10]; steep-slope posterior >= 0.88; no plateau in recovered gamma rung ladders.

**Rationale:** This is a minimal-diff edit on the accepted JU-AEI base implementing the iter-7 critic's single surgical correction, plus the two consolidations its trajectory analysis validated. (1) RESTORE THE EXCURSION CLIP TO [0, 2.4] WHILE HOLDING THE HABITUAL SUPPORT AT U[0, 2.2]. The loop's seven-pass record now cleanly factorizes the two tail channels: TRUE habitual mass above 2.2 (iter 6) cost ~11-21 points on Exp 13 and all of Exp 18's negative offset, while per-trial EXCURSIONS reaching 2.4 from habits below 2.2 (iter 1) delivered the best simultaneous values ever achieved on the four resistant diagnostics — Exp 14 = 0.186, Exp 13 = +6.94, Exp 16 = 0.4397, Exp 18 = -0.149. The iter-7 pass's controlled comparison (only substantive changes: support 2.4->2.2 AND clip 2.4->2.2) saw Exp 14 fall 0.174 -> 0.115 and produced the loop's worst overall loss, implicating the 2.2 clip as the error — a deviation from the arbiter's own JU-AEI spec, which explicitly prescribes the redraw window 'clipped to [0, 2.4]'. This edit restores exactly that geometry: support [0, 2.2], clip [0, 2.4]. (2) REMOVE THE PER-TRIAL LOG-BETA JITTER ENTIRELY. Five passes of evidence are unambiguous: any per-trial precision noise, at any width, with or without box clipping, flattens per-subject gamma posteriors toward the Exp 14 estimator's EB prior mean and caps the tail estimate at ~0.12; the iter-6 pass proved that gamma-only jitter at delta ~0.14 simultaneously satisfies the BF softening (Exp 15 = 14.05 vs real 12.92) and the flip-rate anchor (Exp 16 = 0.4397 vs real 0.4335) — the first and only pass to do both. (3) SYMMETRIC +/-0.50 WINDOW, DELTA [0.13, 0.16]: the validated Pareto band — below ~0.10 the Exps 15/9/12 softening evaporates, above ~0.18 the asymptote anchors (Exps 7/8/10) and Exp 13's sign break. The window is wider than iter 1's +/-0.35, so at the same delta the apparent-tail channel should hold or slightly exceed iter 1's 0.186 on Exp 14. Expected landing zones from the loop's own trajectory: Exp 14 ~ 0.17-0.19, Exp 13 positive ~[4, 10], Exp 15 ~ 12-16, Exp 16 ~ 0.43-0.45, Exp 18 ~ -0.10 to -0.15, Exp 10 >= 0.88, Exps 7/8 within 0.02 of 0.704/0.347. All other code — the shared anti-endorsement raw-weight kernel, graded polarity, lapse, exact tie handling, the uniform [0, 2.2] habitual support, and the policy — is re-emitted from the accepted base with only the jitter block rewritten. Exp 11 (family overshoot; best reference value pi_8 = 67.3 vs real 52.6) and Exp 17 (between-subject variance 2.65, the noisiest diagnostic) remain accepted shared-family limitations, per seven passes of consistent evidence.

**Parameters:**
  - `gamma_u`: `[0, 1]`
  - `log_beta`: `[0.588, 1.792]`
  - `epsilon`: `[0.03, 0.11]`
  - `m`: `[0.74, 1.0]`
  - `delta`: `[0.13, 0.16]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Jittered Uniform-Support Anti-Endorsement Integration (JU-AEI),
    # clip-restoration pass: habitual support held at U[0, 2.2], the
    # per-trial gamma-excursion clip restored to the arbiter's own [0, 2.4],
    # gamma-only jitter (NO per-trial log-beta noise), symmetric +/-0.50
    # window, delta in [0.13, 0.16].
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"JU-AEI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- (1) GAMMA SUPPORT: single continuous uniform on [0, 2.2] ----
    # Held exactly at [0, 2.2]: TRUE habitual mass above 2.2 is poison
    # (it donates likelihood to EPAI's exclusive [2.2, 2.4] segment on
    # Exp 13 and inflates Exp 18's p_hi term). Apparent tail mass is
    # manufactured purely by the per-trial excursions below.
    gamma_habit = 2.2 * float(parameters["gamma_u"])

    # ---- (2) PRECISION: log-uniform raw inverse temperature, ----
    # ---- independent of gamma (no segment-coupled beta boxes)  ----
    # NO per-trial log-beta jitter: five loop passes establish that any
    # per-trial precision noise flattens per-subject gamma posteriors
    # toward the Exp 14 estimator's EB prior mean (0.075), capping the
    # tail estimate at ~0.12, and pushes Exp 16's flip rate toward 0.5.
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # ---- (3) PER-TRIAL STOCHASTIC VALIDITY-ATTENTION JITTER ----
    # Gamma-only. With per-trial probability delta in [0.13, 0.16] the
    # effective gamma is redrawn uniformly from a SYMMETRIC window
    # [gamma_habit - 0.50, gamma_habit + 0.50], clipped to [0, 2.4]
    # (the arbiter's own prescribed geometry). The 2.4 up-clip is the
    # single change most implicated in the previous pass's rejection:
    # excursions reaching 2.4 from habits in [1.7, 2.2] are what the
    # fixed-gamma per-subject MLE reads as apparent tail mass (the only
    # channel that has ever moved Exp 14 above 0.17), while habitual
    # RESIDENCE above 2.2 (removed here by keeping the support at 2.2)
    # is what flipped Exp 13's sign in the iter-6 pass.
    delta = float(parameters["delta"])
    if np.random.random() < delta:
        lo = max(0.0, gamma_habit - 0.50)
        hi = min(2.4, gamma_habit + 0.50)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma_habit

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma_t)

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
    # middle, no mirror cluster. UNCHANGED (Exps 7/8 anchors are exact).
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```
