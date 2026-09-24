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
Noisy Validity-weighted evidence integration with capacity-interference and one-reason lapses (NVW). People do not use a single fixed heuristic; they pool all the expert ratings they can, but each cue's contribution is weighted by its validity raised to a steepness exponent gamma (gamma = 0 recovers Tallying, gamma -> infinity recovers Take-The-Best, so both classic heuristics are limiting cases). Integration is capacity-limited: beyond a working-memory span of about four cues, every additional cue that must be pooled multiplies the reliability (effective inverse temperature) of the integrated comparison by an interference factor delta < 1, so choice consistency is design-dependent and falls sharply as feature count grows. Each cue is additionally attended only with probability alpha (lapses of attention). On a minority of trials — with a probability omega that is modulated by how strongly the most valid discriminating cue's validity dominates the validity of the opposing side's cues — the agent abandons integration and makes a one-reason decision based solely on that top cue. The final choice is a softmax (at the capacity-scaled inverse temperature) over the attended, validity-weighted evidence difference, mixed with an independent lapse to uniform guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # NVW: Noisy Validity-weighted integration with capacity-interference
    # and one-reason lapses.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"NVW expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n}."
        )

    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    delta = float(parameters["delta"])
    epsilon = float(parameters["epsilon"])
    omega_base = float(parameters["omega_base"])
    omega_dom = float(parameters["omega_dom"])

    # (1) Validity-steepness weights: gamma = 0 -> Tallying (equal weights),
    # gamma -> infinity -> Take-The-Best (top cue only).
    w = np.power(v, gamma)

    # (3) Attention: each cue independently attended with probability alpha.
    att = np.random.random(n) < alpha
    a_win = (a > b) & att
    b_win = (b > a) & att
    disc = a_win | b_win

    # (3) Capacity / interference: beyond a working-memory span of ~4 cues,
    # each additional cue that must be pooled multiplies the reliability
    # (effective inverse temperature) of the integrated evidence by delta.
    # This makes decision noise grow with feature count.
    excess = max(0, n - 4)
    beta_eff = beta * (delta ** excess)

    def _sig(z):
        # Numerically stable logistic.
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))

    if not np.any(disc):
        # No attended discriminating cue: pure guess.
        p_a = 0.5
    else:
        # (1) Integration: softmax over the weighted evidence difference.
        d = float(np.sum(w[a_win]) - np.sum(w[b_win]))
        p_int = float(_sig(beta_eff * d))

        # (2) One-reason mode: the single most valid attended
        # discriminating cue decides (TTB-like), with confidence scaled
        # by that cue's weight and the effective sensitivity.
        idx = np.where(disc)[0]
        top = int(idx[np.argmax(v[idx])])
        s = 1.0 if a[top] > b[top] else -1.0
        p_1r = float(_sig(beta_eff * w[top] * s))

        # (2) Validity-dominance coupling for the one-reason lapse rate:
        # dominance of the top cue over the opposing side's cue validities.
        opp = b_win if s > 0 else a_win
        if np.any(opp):
            dom = float(v[top] - np.mean(v[opp]))
        else:
            dom = float(v[top] - 0.5)
        omega_t = float(np.clip(omega_base + omega_dom * dom, 0.0, 0.95))

        p_a = (1.0 - omega_t) * p_int + omega_t * p_1r

    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent lapse toward uniform guessing.
    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs

`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.62, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 12: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 13: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 17: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 19: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 20: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 22: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. With validities fixed per feature, both theories reduce to a fixed position-indexed weight profile applied to the cue-allegiance pattern s_j = sign(A_j - B_j): NVW uses w_j ∝ v_j^gamma (validity-VALUE based), while SWC uses w_j = (1-lambda)/n + lambda[(1-rho)vnorm_j + rho*(j/(n-1))^eta] (position-RECENCY based, with recency necessarily dominating: lam*rho >= 0.64 versus a validity coefficient <= 0.20 and flat <= 0.20 across SWC's entire box). I therefore present validities in STRICTLY DECREASING order [0.95, 0.90, 0.85, 0.62, 0.56, 0.52] — SWC's own assumed presentation format, i.e., its home turf — with a sharp validity gap between positions 3 and 4 creating a 'high-validity early block' {0.95, 0.90, 0.85} and a 'low-validity late block' {0.62, 0.56, 0.52}. This makes the two weight profiles strongly ANTI-CORRELATED: NVW's weights fall with position, SWC's rise (convexly) with position. n_features = 6 also places two cues beyond NVW's span-4, engaging its capacity-interference signature (beta_eff = beta*delta^2 in [0.16, 0.99], i.e., NVW itself predicts compressed confidence), while SWC's beta is design-independent.

DISSOCIATION AXES (all predicted boxes verified at the corners of BOTH parameter spaces; NVW's d-sign is invariant over gamma in [0.7, 1.3] because v^gamma is monotone in v, and SWC's recency dominance holds over its whole box):

(1) SIGN-FLIP FAMILY (S1, S2, S3, E1, E3, T3, T6, T8 + mirrors; 16 of 24 trials). The early high-validity block favors one option and the late low-validity block the other. NVW predicts the early-block (validity) winner; SWC predicts the late-block (recency) winner — PARAMETER-FREE OPPOSITE SIGNS. Headline boxes for P(A): S1 (A wins only expert 6 vs expert 1): NVW 0.39-0.51, SWC 0.60-0.85. S2 (A wins experts 5-6 vs 1-2): NVW 0.28-0.48, SWC 0.64-0.91. S3 (A wins experts 4-6 vs 1-3): NVW 0.25-0.47, SWC 0.63-0.91. E1 — the most damning trial for SWC: B wins the FOUR MOST VALID experts and A only the two least valid, yet SWC still predicts A (P(A) in [0.55, 0.88]) because recency weight on the last two positions outweighs everything; NVW predicts B (P(A) in [0.10, 0.41]). E3: A wins the two top experts vs B's single bottom expert — SWC predicts B (P(A) in [0.18, 0.41]) because one late cue outweighs two top cues; NVW predicts A (0.55-0.80). T6/T8 extend the flip to mixed patterns (boxes: NVW 0.52-0.71 vs SWC 0.09-0.37; NVW 0.44-0.50 vs SWC 0.62-0.89).

(2) VALIDITY-TIE / RECENCY-ONLY (T1, T1m). A wins experts 1 and 6, B wins experts 3 and 4: the validity-weighted evidence is exactly balanced (|d| <= 0.04 for all gamma), so NVW predicts chance (P(A) in [0.50, 0.55], the tiny upward tilt being its one-reason lapse toward the top cue), while SWC's recency gradient (expert 6 vs experts 3-4) predicts P(A) in [0.58, 0.82]. Non-overlapping boxes: observed P(A) <= 0.55 falsifies SWC's entire parameter space.

(3) FIVE-VS-ONE (E2, E2m). The five most valid experts versus the single least valid expert: NVW predicts the validity winner with P in [0.62, 0.97]; SWC's flat + vnorm components nearly cancel its recency component, so its box STRADDLES chance (P(validity winner) in [0.42, 0.58]). Observed P > 0.6 for the five-expert side is impossible for SWC at any admissible parameter combination.

(4) CUE-MIGRATION PAIR (S3 vs S4). Moving expert 3 (v = 0.85) from the early side to the late side flips NVW's sign (d: -1.00 to +0.70; predicted P(A) crosses 0.5 between adjacent trial types) while leaving SWC's prediction on the SAME side (d approx 1.0 to 1.3, both strongly pro-late-block). A sign-crossing between S3 and S4 is a qualitative signature only NVW produces.

(5) AGREEMENT ANCHORS (A1/A1m total splits; S4/S4m). Both theories agree; they establish that subjects can perform the task and calibrate overall consistency. Note the confidence dissociation: because of capacity interference, NVW's maximum predicted probability on any non-total trial is approx 0.8-0.9, whereas SWC predicts up to 0.93-0.97 on recency-unbalanced trials — the joint profile 'validity-ordered choices at moderate consistency' is uniquely NVW; 'recency-ordered choices at high consistency' is uniquely SWC.

Every conflict type appears in A/B-mirrored form to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4 repetitions in randomized order = exactly 96 trials. Summary of decisive, parameter-free signatures: (i) on 16 sign-flip trials the theories predict opposite choices with non-overlapping probability boxes; (ii) on T1 NVW predicts exactly chance where SWC's box excludes chance; (iii) on E2 SWC's box is confined to near-chance where NVW predicts high confidence; (iv) the S3->S4 sign crossing is NVW-only; (v) regressing P(A) on validity-weighted evidence (sum v_j*s_j) versus recency-weighted evidence (sum (j/5)^3*s_j) — anti-correlated by construction — identifies which profile subjects use, with opposite regression signs under the two theories.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
CORE CONTRAST. Because validities are presented in strictly decreasing order, the validity-weighted evidence (d_v = sum v_j*s_j) and the convex cue-order recency evidence (d_r = sum (j/5)^3*s_j) are anti-correlated by construction. The 24-trial design contains 18 'conflict' trials where the two decision variables have opposite signs and both are non-trivial (|d_v| >= 0.3, |d_r| >= 0.5): e.g., trial 13 (A wins experts 1-2, B wins expert 6: d_v = +1.35, d_r = -0.99), trial 19 (A wins experts 1,3 vs B's 5,6: d_v = +0.80, d_r = -1.45), trial 3/5/9/11 and all their A/B mirrors. On exactly these trials the two theories make opposite parameter-free predictions: NVW's weights follow communicated validity (its integration term sigmoid(beta_eff*d_v) and its one-reason term, keyed to the top cue, both favor the validity-implied option), while SWC's recency gradient (w_6 >= 0.64 vs w_1 <= 0.19 across its entire box) favors the late-block option, i.e., the anti-validity side.

METRIC. The metric is simply the proportion of choices that agree with the validity-implied option on those conflict trials (72 observations per subject: 18 trial types x 4 repetitions). It is a plain behavioral summary — 'do subjects follow the communicated validities when validity and recency disagree?' — not a re-implementation of either model.

PREDICTED VALUES. NVW: per-trial adherence sigmoid(beta_eff*|d_v|) with beta_eff in [0.16, 0.99] gives per-trial values 0.54-0.80, aggregate ~0.60-0.65 (worst-case corner ~0.54). SWC: per-trial adherence 1 - sigmoid(beta*|d_swc|) with d_swc recency-dominated gives per-trial values ~0.10-0.45 on the 2-vs-1 and 3-vs-2 cue trials (aggregate ~0.28-0.35; even its most validity-friendly corner stays ~0.42). The gap in means is ~0.3.

WHY IT SURVIVES THE N=25 TEST. Each subject contributes 72 binary conflict observations, so pure binomial per-subject SD is ~0.055-0.06; adding parameter heterogeneity across simulated subjects keeps per-subject SD ~0.08 (variance ~0.006). With a mean gap of ~0.3, Welch's t at N=25 is roughly 0.3 / sqrt(2*0.006/25) ≈ 9-14, far below the alpha=0.01 threshold, and even the joint worst-case corners of both parameter boxes (noisiest NVW vs most validity-friendly SWC) still leave a gap of ~0.12 with t ≈ 7. Mirror trials cancel side/position bias, and the |d_v| threshold removes the validity-tie trial (d_v ≈ 0) where 'validity winner' is arbitrary. The metric works identically on the pooled DataFrame and on single-subject slices, so between-subject variance is well defined.

Source:
def metric(data: pd.DataFrame) -> float:
    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.
    #
    # For every trial we compute two decision variables from the presented
    # cue patterns:
    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)
    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency
    #                                              evidence for A)
    # A trial is a CONFLICT trial when the two point in opposite directions
    # and both are non-trivial in magnitude. On those trials the metric is
    # the proportion of choices that follow the VALIDITY-implied option.
    #
    # NVW (validity-weighted integration) predicts adherence clearly above
    # chance (~0.6); SWC (recency-dominated subjective weights) predicts
    # adherence well below chance (~0.3), because on these trials its
    # recency gradient favors the opposite option.

    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)
    n = int(validities.shape[0])
    # Convex recency profile over presentation positions (eta ~ 3).
    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0

    a_col = data["option_a_ratings"]
    b_col = data["option_b_ratings"]
    r_col = data["response"]

    hits = []
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != n or b.shape[0] != n:
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue

        d_v = float(np.dot(validities, s))
        d_r = float(np.dot(recency, s))

        # Need a real conflict with non-trivial evidence on both axes.
        if abs(d_v) < 0.30 or abs(d_r) < 0.50:
            continue
        if np.sign(d_v) == np.sign(d_r):
            continue

        validity_winner_is_a = (d_v > 0.0)
        chose_a = (int(resp) == 0)
        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)

    if len(hits) == 0:
        return 0.0
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_3 (simulated): 0.5988 (var=0.0039)
- Predicted under pi_4 (simulated): 0.2816 (var=0.0052)
- Observed on real data: 0.2313 (var=0.0046)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=7):** [0.6, 0.96, 0.6, 0.6, 0.6, 0.6, 0.9]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 13: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 18: A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 19: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 20: A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 21: A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate SWC (convex cue-order recency, deterministic attention, no capacity limit) from NVW (validity^gamma weighting, capacity interference, span ~4). Prior design [4] used strictly decreasing validities, making recency and validity ANTI-correlated everywhere. This design takes the opposite, novel route: a NON-MONOTONE validity profile [0.60, 0.96, 0.60, 0.60, 0.60, 0.60, 0.90] with two high-validity peaks at positions 2 and 7 and five equal low-validity cues. This decorrelates validity from recency (rather than anti-correlating it), so each axis can be isolated in pure form, and it lets the CONVEXITY of SWC's recency gradient (eta in [2.5,4.5]: the last cue's weight is ~2x the second-to-last's and ~10x the third-to-last's) fight NVW's validity MASS head-to-head. n_features = 7 also puts three cues beyond NVW's span-4, engaging its capacity interference (beta_eff = beta*delta^3 <= 0.30), while SWC's beta is design-independent. All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - EQUAL-VALIDITY RECENCY LADDER (T4/T5/T6 + mirrors): A wins expert 1 (60%), B wins exactly one equally-valid (60%) expert at position 4, 5, or 6. NVW's evidence is EXACTLY zero for every parameter value (equal validities => equal weights v^gamma, symmetric attention/lapses), so NVW predicts exactly 50/50 on all six trials - parameter-free. SWC's recency gradient predicts a monotone ladder below 0.5 for the early-cue option: P(A) boxes [0.42,0.49] (vs p4), [0.30,0.43] (vs p5), [0.17,0.31] (vs p6). Observed P(later expert's product) > 0.60 on T6 is impossible for NVW at ANY admissible parameter combination, and required by SWC. This is the single cleanest trial family: pure recency with validity perfectly controlled.

AXIS 2 - HEAD-TO-HEAD DUELS (D1/D2 + mirrors): D1: A wins expert 2 (96%), B wins expert 7 (90%). NVW follows validity: P(A) in [0.50, 0.53]. SWC's convex recency makes the LAST cue dominate: P(A) in [0.14, 0.38]. Opposite signs, non-overlapping boxes - the subject prefers the product endorsed by the slightly LESS accurate but LATER expert iff SWC is true. D2 sharpens it: 96% expert 2 vs 60% expert 6 - a huge validity gap that NVW respects (P(A) in [0.50,0.54]) but SWC's recency overrides (P(A) in [0.31,0.46]). These flips survive even if NVW's capacity interference were removed (delta=1): NVW would just become more confident in the validity winner; the sign disagreement is structural.

AXIS 3 - CONVEXITY vs VALIDITY MASS (C1, L1-L4, E2 + mirrors): C1: A wins expert 7 (90%) alone; B wins experts 2+4 (96%+60%). NVW: validity mass favors B, P(A) in [0.43,0.50]; SWC: the last cue's convex weight beats both, P(A) in [0.59,0.80]. The L-ladder measures the crossover count: A wins expert 7 vs B wins k of the mid cues {3,4,5,6}. NVW's validity arithmetic says the 90% cue beats exactly ONE 60% cue (d=+0.23 at k=1, d=-0.47 at k=2), so NVW crosses 0.5 between L1 and L2. SWC's convex gradient says the last cue outweighs three mid cues (recency sum at k=3 is 0.40-0.60 < 1.0), crossing only at k=4 - a measured crossover-count dissociation of 2-3 opposing cues. E2 is the block-level flip: A wins the 3 late cues (incl. only the 90%), B wins 4 early cues (incl. the 96%): NVW says B (P(A) in [0.42,0.50]), SWC says A (P(A) in [0.71,0.94]).

AXIS 4 - CAPACITY CLIFF / CONFIDENCE (E1 + mirror): A wins experts 4-7, B wins 1-3. Both theories favor A in direction, but NVW's beta_eff <= 0.30 caps its P(A) at ~[0.50,0.56] on EVERY trial in this design (max |beta_eff*d| ~ 0.55), while SWC predicts [0.75,0.95]. Observed accuracy > 0.70 on E1 falsifies NVW's entire parameter space; near-chance performance everywhere with a residual validity-ordering falsifies SWC.

Aggregate signature: regressing P(A) on recency-weighted evidence sum((j/6)^3.5 * s_j) versus validity-weighted evidence sum(v^gamma * s_j) - decorrelated here by construction (expert 2: high validity, negligible recency; expert 6: low validity, high recency) - yields opposite-loading profiles under the two theories. Every trial type appears in A/B-mirrored form to absorb side/position bias. 24 unique pairs, K = 96//24 = 4 repetitions in randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the fraction of choices consistent with a convex cue-order recency gradient (position weights (j/(n-1))^eta), computed only on trials where (a) the gradient's preferred option is invariant across SWC's entire eta range [2.5, 4.5] and (b) its mid-range evidence magnitude is at least 0.25. In this design that admits 20 of the 24 trial types (the equal-validity recency ladder T3-T6, the validity-vs-recency duels T7-T12, the late-cue-mass trials T13-T18, and the block-level flips T21-T24), while excluding only the degenerate cue-1-vs-cue-4 trials and the eta-ambiguous 1-late-vs-4-mid trials. Mirroring in the design makes the recency-favored option A on exactly half the included trials, so side bias cancels. Predicted separation: under SWC, the per-trial probability of choosing the late-cue-endorsed product ranges from ~0.55 (worst-case corner beta=0.8, eta=2.5) to ~0.9, averaging ~0.70-0.75, so the metric lands ~0.65-0.85. Under NVW, capacity interference forces beta_eff = beta*delta^3 <= ~0.30 at n_features=7, so its logistic outputs stay within ~0.5 +/- 0.07 of chance on every trial in this design, and on the conflict trials its residual validity-ordering is anti-recency; its expected rate is ~0.49-0.51 with pure binomial noise. The gap (~0.2+ in means) is large relative to per-subject binomial noise over ~80 scored trials (sd ~0.05), so with N=25 the Welch t-test separates the theories by many sigma. The metric works identically on a single subject's slice (same trial filtering), so between-subject variance is well-defined.

Source:
def metric(data: pd.DataFrame) -> float:
    # Recency-consistency rate: fraction of choices aligned with a convex
    # cue-order recency gradient, restricted to trials where that gradient
    # is strong and its direction is unambiguous across SWC's eta range.
    #
    # SWC (advocated): late cues dominate via convex recency -> rate well
    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).
    # NVW (competing): validity-weighted integration with capacity
    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices
    #   near chance on nearly every trial, and anti-recency or neutral on
    #   the conflict trials -> rate ~0.50.
    df = data.reset_index(drop=True)
    if len(df) == 0:
        return 0.5

    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]
    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]
    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()

    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:
        return 0.5
    n = int(a_arrs[0].shape[0])
    if n < 2:
        return 0.5

    pos = np.arange(n, dtype=float) / (n - 1.0)
    w_lo = pos ** 2.5   # SWC's eta lower corner
    w_mid = pos ** 3.5  # SWC's eta midpoint
    w_hi = pos ** 4.5   # SWC's eta upper corner

    consistent = []
    for a, b, r in zip(a_arrs, b_arrs, resp):
        if a.shape[0] != n or b.shape[0] != n:
            continue
        if not np.isfinite(r):
            continue
        s = np.sign(a - b)
        if not np.any(s != 0):
            continue
        e_lo = float(np.dot(w_lo, s))
        e_mid = float(np.dot(w_mid, s))
        e_hi = float(np.dot(w_hi, s))
        # Require the recency gradient's direction to be invariant across
        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs
        # 4 mid cues, where the sign flips with eta).
        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):
            continue
        # Require non-trivial gradient magnitude (drops near-degenerate
        # trials where even SWC predicts ~0.52).
        if abs(e_mid) < 0.25:
            continue
        fav_a = e_mid > 0.0
        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B
        consistent.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(consistent) == 0:
        return 0.5
    return float(np.mean(consistent))

### RESULTS
- Predicted under pi_3 (simulated): 0.4847 (var=0.0034)
- Predicted under pi_4 (simulated): 0.7261 (var=0.0028)
- Observed on real data: 0.3739 (var=0.0026)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Design validities (fixed by experiment): strictly decreasing,
    # so the TTB cue order is simply feature 0,1,2,3,4.
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_vec(x):
        if isinstance(x, str):
            return np.array(
                [int(v) for v in x.strip("[]()").replace(",", " ").split()]
            )
        return np.asarray(x, dtype=float)

    hits = []
    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    for a_cell, b_cell, resp in zip(a_col, b_col, r_col):
        a = _to_vec(a_cell)
        b = _to_vec(b_cell)
        if a.size == 0 or a.shape != b.shape:
            continue

        # --- TTB winner: first discriminating cue in validity order ---
        ttb_winner = None
        for j in cue_order:
            if j >= a.size:
                continue
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue  # no discriminating cue: TTB guesses, uninformative

        # --- Tallying winner: strict feature-wise win counts ---
        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally ties -> Tallying guesses at 0.5, dilutes contrast
        tally_winner = 0 if a_wins > b_wins else 1

        if tally_winner == ttb_winner:
            continue  # heuristics agree: uninformative

        # Pure conflict trial: TTB's option vs the tally majority's option.
        hits.append(1.0 if int(resp) == ttb_winner else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2080 (var=0.0060)
**Predicted under pi_3:** 0.2377 (var=0.0050)
**Predicted under pi_4:** 0.2197 (var=0.0050)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-vs-TTB choice-allegiance contrast.

    For every trial we determine (i) the TALLYING winner (option with more
    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on
    the highest-validity discriminating cue, validities fixed by design as
    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is

        mean( response == tally winner ) - mean( response == TTB winner )

    On the 4 trial types where both heuristics pick the same option
    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and
    cancel.  On the 12 conflict trial types they are exact complements, so
    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly
    positive under Tallying, strongly negative under TTB.
    """
    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    tally_match = []
    ttb_match = []

    for ca, cb, cr in zip(a_col, b_col, r_col):
        a = np.asarray(ca, dtype=float).ravel()
        b = np.asarray(cb, dtype=float).ravel()
        r = int(cr)

        # Tallying winner: more strict feature-wise wins (ties on a feature
        # contribute to neither side).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = -1  # undiscriminating tally -> excluded

        # TTB winner: first discriminating cue in descending-validity order.
        ttb_winner = -1
        for j in cue_order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break

        if tally_winner >= 0:
            tally_match.append(1.0 if r == tally_winner else 0.0)
        if ttb_winner >= 0:
            ttb_match.append(1.0 if r == ttb_winner else 0.0)

    if len(tally_match) == 0 or len(ttb_match) == 0:
        return 0.0

    return float(np.mean(tally_match) - np.mean(ttb_match))
```

**Observed (real) value:** 0.1583 (var=0.0053)
**Predicted under pi_3:** 0.1658 (var=0.0093)
**Predicted under pi_4:** 0.2079 (var=0.0051)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def _to_vec(x):
    """Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(list(x), dtype=float)
    s = str(x).strip().replace("[", " ").replace("]", " ").replace(",", " ")
    return np.asarray(s.split(), dtype=float)


def metric(data: pd.DataFrame) -> float:
    df = data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows == 0:
        return 0.0

    A = np.vstack([_to_vec(x) for x in df["option_a_ratings"]])
    B = np.vstack([_to_vec(x) for x in df["option_b_ratings"]])
    n_feat = A.shape[1]

    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.
    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)
    if v.shape[0] != n_feat:
        v = np.linspace(0.95, 0.60, n_feat)

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = (np.asarray(df["response"], dtype=float) == 0.0).astype(float)

    a_win = A > B
    b_win = B > A

    # Signed tally margin (Tallying's ONLY explanatory variable).
    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)

    # Validity-weighted evidence difference (NVW's driving quantity).
    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)

    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.
    X = np.column_stack([np.ones(n_rows), margin, d])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])
```

**Observed (real) value:** -0.5756 (var=0.0186)
**Predicted under pi_3:** 0.2763 (var=0.0165)
**Predicted under pi_4:** -0.4120 (var=0.0294)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Anchor-allegiance: proportion of trials with a large tally margin
    (|a_wins - b_wins| >= 6) on which the subject chose the option that
    wins on more features (the tally winner).

    Under Tallying, these are the unambiguous anchors (8-0 and 7-1 splits:
    design trials 7, 8, 21, 22, 23, 24 and their 4 repetitions = 24 trials
    per subject), where the softmax over tallies predicts near-deterministic
    choice for any non-trivial beta. Under NVW, the capacity-interference
    factor delta^4 crushes the effective inverse temperature to
    beta_eff <= 11 * 0.30^4 ~ 0.09, so even the maximum validity-weighted
    evidence difference (~6) yields P ~ sigmoid(0.5) <= 0.64; attention
    lapses and the uniform-mixing epsilon only pull this further toward
    0.5. The two theories therefore make sharply separated predictions on
    this proportion, and the A/B-mirrored anchors cancel side bias.
    """
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    n_anchor = 0
    n_hit = 0

    for ra, rb, resp in zip(a_col, b_col, r_col):
        va = np.asarray(ra, dtype=float)
        vb = np.asarray(rb, dtype=float)
        if va.shape != vb.shape:
            continue

        a_wins = float(np.sum(va > vb))
        b_wins = float(np.sum(vb > va))
        margin = a_wins - b_wins

        if abs(margin) >= 6:
            n_anchor += 1
            winner_is_a = margin > 0
            chose_a = (int(resp) == 0)
            if chose_a == winner_is_a:
                n_hit += 1

    if n_anchor == 0:
        return 0.5
    return float(n_hit) / float(n_anchor)

```

**Observed (real) value:** 0.8433 (var=0.0152)
**Predicted under pi_3:** 0.5300 (var=0.0122)
**Predicted under pi_4:** 0.8133 (var=0.0065)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Scoring both theories against all six experiments (absolute error of the point estimate from the observed value):\n\n- Exp 1 (validity-vs-recency conflict, decreasing validities): observed validity-adherence 0.231. pi_3 predicted 0.599 (error 0.37, a SIGN error \u2014 it predicts above-chance validity-following where subjects are strongly anti-validity); pi_4 predicted 0.282 (error 0.05). pi_4 wins decisively.\n- Exp 2 (non-monotone validities, recency-consistency): observed 0.374. pi_3 predicted 0.485 (error 0.11); pi_4 predicted 0.726 (error 0.35, a SIGN error \u2014 its recency gradient predicts well-above-chance late-cue preference where subjects are below chance). pi_3 wins this one.\n- Exp 3 (TTB-vs-tally): observed 0.208. pi_3 0.238 (err 0.03), pi_4 0.220 (err 0.012). Both adequate; pi_4 marginally better.\n- Exp 4 (tally-minus-TTB contrast): observed 0.158. pi_3 0.166 (err 0.008), pi_4 0.208 (err 0.05). Both adequate; pi_3 marginally better.\n- Exp 5 (within-margin validity partial coefficient): observed -0.576. pi_3 predicted +0.276 (error 0.85, a catastrophic SIGN error \u2014 its core validity-weighted integration is flatly contradicted); pi_4 predicted -0.412 (error 0.16). pi_4 wins decisively.\n- Exp 6 (anchor consistency at n=8): observed 0.843. pi_3 predicted 0.530 (error 0.31 \u2014 its capacity-interference mechanism delta^4 crushing beta_eff is falsified; subjects are highly consistent far beyond span 4); pi_4 predicted 0.813 (error 0.03). pi_4 wins decisively.\n\nAggregate: pi_4 (SWC) wins 4/6 with total error ~0.66; pi_3 (NVW) wins 2/6 with total error ~1.68, including three catastrophic failures (Exps 1, 5, 6) that falsify BOTH of NVW's central commitments \u2014 validity-weighted integration (subjects are anti-validity, not pro-validity, within margin) and capacity interference (no consistency cliff at n=8). NVW is degenerate: it even fails the experiment its own proponent designed to showcase it.\n\nHowever, SWC is not yet task-invariant either. Its convex recency gradient sign-fails Experiment 2: with validities decorrelated from position, it predicts 0.65-0.85 recency-consistency but subjects produced 0.37 \u2014 a direction its entire parameter box cannot produce. A fixed position gradient (recency or primacy) cannot explain both Exp 1 (pro-late-cue with decreasing validities) and Exp 2 (anti-late-cue with non-monotone validities). What CAN unify all six experiments is a cue-weighting that is a DECREASING function of communicated validity: subjects behave as if they mistrust or invert the communicated validities (e.g., treating the number as an error rate). Checking this 'inverted-validity counting' against every experiment: (i) Exp 1 \u2014 weights increasing in position (since validities decrease) reproduce the 0.23 validity-adherence; (ii) Exp 2 \u2014 on the decisive trial families (single late 0.90/0.96 cue vs early/mid 0.60 cues) the anti-validity gradient favors the early option, reproducing below-chance recency-consistency (~0.40 predicted vs 0.37 observed, with the duels and block trials partially offsetting); (iii) Exps 3/4 \u2014 anti-validity pushes choices away from the top-cue (TTB) option toward the tally majority, matching the tally lean; (iv) Exp 5 \u2014 a strongly negative within-margin validity coefficient; (v) Exp 6 \u2014 on 8-0/7-1 anchors all weights fall on one side, so high consistency with no capacity cliff. One mechanism, six experiments, correct signs everywhere. Since neither existing theory explains all the data but this inverted-validity signature does, the correct action is to propose a new theory, replacing the degenerate pi_3 (NVW) so that it becomes a strong, task-invariant competitor to the incumbent pi_4.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 (NVW, degenerate) with a new theory: INVERTED-VALIDITY COUNTING (IVC) \u2014 'distrust-weighted tallying'. Core claim: subjects do weight cues by the communicated validities, but with an inverted mapping \u2014 they act as if the stated validity is the expert's error rate (or they systematically distrust the most highly validated experts). Decision rule: attend to all discriminating cues deterministically; assign each cue j a subjective weight w_j = (1-tau)/n + tau * (1 - v_j)^kappa / Z, where tau in [0.3, 1.0] is the trust-in-inversion mixing proportion, kappa in [0.5, 3.0] is the inversion steepness (kappa = 0 with tau = 0 recovers plain Tallying; large kappa approaches a 'worst-expert-first' one-reason rule); integrate additively d = sum_j w_j * sign(A_j - B_j); choose A with sigmoid(beta * d), beta in [1, 12] design-independent (NO capacity interference \u2014 Exp 6's 0.84 anchor consistency at n=8 forbids it), plus a small uniform lapse epsilon in [0.01, 0.10]. Optionally include a weak secondary position gradient with a mixing weight strictly bounded below 0.15 so it can absorb residual order bias without ever dominating the validity inversion (this is what let SWC overfit Exp 1 and then sign-fail Exp 2). Predictions to verify at box corners: Exp 1 validity-adherence ~0.15-0.30 (observed 0.23); Exp 2 recency-consistency ~0.35-0.45 (observed 0.37 \u2014 the late 0.90/0.96 cues get the smallest inverted weights, so the early low-validity cues win the decisive families); Exp 3 TTB-adherence ~0.15-0.30 (observed 0.21); Exp 4 tally-minus-TTB ~+0.1-0.25 (observed 0.16); Exp 5 within-margin validity coefficient strongly negative, ~-0.4 to -0.8 (observed -0.58); Exp 6 anchor adherence ~0.8-0.95 (observed 0.84). This keeps SWC's strengths (additive integration, deterministic attention, design-independent noise, no capacity limit) while replacing its position-recency gradient \u2014 which sign-failed Exp 2 \u2014 with the validity-inversion gradient that all six experiments jointly support."
}
```

## Usage

```json
{
  "input_tokens": 16842,
  "output_tokens": 16143,
  "reasoning_tokens": 14540,
  "provider": "Sail Research"
}
```
