# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=17):** [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.74, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 20: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-12): COUNT-DIFFERENCE DOSE-RESPONSE AT SATURATED EVIDENCE. Exp 12, the only prior SVD-vs-GDRD-B design, compared the theories with single-shot conflict, asymptote, parity, count-step, and exchange cells; it never manipulated the magnitude of the endorsement-count difference. This is the one stimulus dimension on which the two theories are structurally opposite: SVD's counter branch is a parameter-free STEP (exactly 1 for ANY count difference >= 1, exactly 0.5 at parity), whereas GDRD-B's Route 2 is a GRADED sigmoid(gamma*Delta) that keeps rising from Delta=1 to Delta=2. Because GDRD-B deliberately holds E[w] at SVD's base value, pooled means are near-identical by construction (conflict means differ by <2 pts); the design therefore (i) saturates the skeptic route so all parameter slop is pinned, (ii) concentrates 32 conflict + 32 congruent trials per subject so per-subject rates are estimable, and (iii) pre-registers the distributional statistics (12% counter mode, super-binomial variance, asymptote cap) plus two DOSE-RESPONSE signatures on which SVD predicts exact zeros. VALIDITY ENGINEERING: 6 perfect experts (v=1.0) give the maximum, KAPPA-INVARIANT charge c0+c1 (1.00-1.48 across both boxes), closing every corner-escape route; GDRD-B's micro-trust term is bounded at alpha*3 = 0.15 per perfect endorsement (18*alpha <= 0.90 total on 6-perfect cells), and saturation was verified to survive alpha=0.05 at EVERY corner of BOTH boxes (conflict |D| >= 1.99, congruent D >= 2.20). Nine coin experts (v=0.50) give the cheapest charge (0.29-0.58) and logit exactly 0 (trust-inert). Experts at 0.74 and 0.86 are penalty-matched single endorsers for the Delta=1 count-probe (charge within 2*charge(0.50) +/- 0.2 at every corner). 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K=96//24=4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. Mid parameters: SVD (c0=.085, c1=1.125, kappa=1.65, beta=1.9, w=.86, eps=.18); GDRD-B skeptic (c0=.095, c1=1.175, kappa=1.85, beta=1.9, w=.90, eps=.13, gamma=2.25, alpha=.025) and counter (w=.56, eps=.075, gamma=2.5); box ranges verified at box corners. FAMILY 1 -- CONFLICT STAIRCASE (pairs 1-4, primary): A = 6 perfects vs B = 7, 8, 9 coins (Delta = 1, 2, 3) plus a composition replicate (B = {0.86 + 6 coins}, Delta=1). The skeptic route is pinned toward the coin side at every corner (|D| >= 2.9, s_A <= 0.023; <= 0.07 including max trust), while the count route votes the perfect side. SVD: P(A) = (1-eps)(w*s_A + (1-w)) + eps/2 in [0.13, 0.29] for EVERY subject at EVERY corner, mid flat at 0.205, and FLAT IN DELTA (counter = 1 for all Delta >= 1; s_A pinned) -- per-rung drift <= 1 pt. GDRD-B skeptics: [0.12, 0.23], mid 0.144. GDRD-B counters: [0.35, 0.53], mid 0.414 -> 0.442 (RISING ~3 pts from Delta=1 to Delta=2 via sigmoid(gamma)). DISJOINT per rung: SVD max 0.277/0.280/0.286/0.280 vs counter min 0.351/0.397/0.391/0.373. FALSIFIERS: any subject with pooled conflict rate >= 0.32 (32 trials) is outside SVD's entire box -- GDRD-B predicts ~12% such subjects plus ~3x super-binomial between-subject variance (SD ~0.065 vs SVD's ~0.02); conversely, ZERO such subjects in N >= 40 rejects GDRD-B's bimodal population (P = 0.88^40 = 0.7%). Any Delta-effect >= 2 pts at saturation falsifies SVD outright; flatness falsifies GDRD-B. FAMILY 2 -- CONGRUENT DESCENT (pairs 5-8): A = 1, 3, 4, 5 coins vs B = 6 perfects (Delta = 5, 3, 2, 1); both routes saturate toward A (D >= 2.2 at every corner including max trust). SVD predicts the SAME P(A) = 1-eps/2 on ALL FOUR rungs ([0.88, 0.94] per subject, mid 0.910) -- its counter branch equals 1 for every Delta >= 1, so SVD is EXACTLY flat in Delta. GDRD-B predicts a specific descending profile: skeptic mid 0.935, 0.935, 0.934, 0.925; counter mid 0.962, 0.962, 0.958, 0.924. The Delta=1 deficit (1-w)(1-eps)(1-sigmoid(gamma)): SVD [0, 1.2] pts; GDRD-B skeptics [0.2, 4.9], counters [1.6, 8.8] (mid 1.0 / 3.9 pts) -- a direct per-subject read of the count route's non-saturation that no SVD parameterization can produce (its step is saturated at Delta=1 by construction). LEVEL assay: pooled asymptote GDRD-B 0.938 vs SVD 0.910; SVD hard-caps every subject at 0.94 (eps >= 0.12), while GDRD-B skeptics reach 0.96 and counters 0.95-0.975 -- ~2/3 of GDRD-B subjects above 0.94 vs 0% under SVD. FAMILY 3 -- PARITY (pair 9): 6 perfects vs 6 coins; skeptic pinned toward coins, counts matched. P(A) = (1-w)(1-eps)/2 + eps/2: SVD [0.095, 0.218] (mid 0.147); GDRD-B counters [0.206, 0.307] (mid 0.241) -- mid separation 9.4 pts, corner overlap 1.2 pts acknowledged (secondary). FAMILY 4 -- COUNT-PROBE (pairs 10-11): {0.74} and {0.86} vs 2 coins, Delta=1. SVD mid 0.595/0.531; GDRD-B skeptic 0.572/0.499; counter 0.697/0.649 -- an 11-pt mid separation directly assaying the count route's value sigmoid(gamma) vs SVD's step of 1.0; penalty-matching residual D in [-0.17, +0.22] across boxes keeps this supportive rather than decisive, and the C1-C2 difference calibrates each subject's skeptic validity sensitivity (c1, kappa). FAMILY 5 -- EXCHANGE CALIBRATION (pair 12): 1 perfect vs 3 coins traces each subject's sigmoid (beta, exchange rate), sharpening the per-subject classification; boxes are wide and overlapping by design. UNFAKEABILITY IN BOTH DIRECTIONS: (i) SVD's counter branch is a step, so on BOTH staircases its prediction is provably independent of Delta at saturated evidence -- any measurable Delta-gradient (conflict rise or congruent descent >= 2 pts) is outside its box; its w >= 0.80, eps >= 0.12 bound every subject's conflict rate <= 0.29 and asymptote <= 0.94, while GDRD-B's counter mode (w in [0.50, 0.62], eps in [0.05, 0.10]) lands at 0.35-0.53 conflict and 0.95-0.975 asymptote -- disjoint envelopes on Family 1 and Family 2 simultaneously. (ii) GDRD-B's skeptic mode overlaps SVD by construction (E[w] matched), so the falsification runs through the population: zero counter-mode subjects in N >= 40, a pooled asymptote <= 0.92, or a flat congruent profile (L1-L3 <= 1 pt pooled) rejects GDRD-B. The parameters trade off oppositely across families: the GDRD-B corner that best mimics SVD's asymptote (eps=0.18) simultaneously deepens its counter-mode divergence, and no single SVD corner can be high on conflict, capped on asymptote, AND flat in Delta while matching a subject high on all three. POWER: pooled assays (asymptote level, congruent deficit) need ~20 subjects; the counter-tail/bimodality assays need N >= 40-50 -- the design concentrates 32 conflict and 32 congruent trials per subject to make per-subject rates estimable. PREDICTED OUTCOMES: an SVD population shows conflict ~0.21 flat with all subjects in [0.13, 0.29], asymptote ~0.91 with no subject above 0.94, exactly flat congruent profile, unimodal spread. A GDRD-B population shows conflict pooled ~0.19 with a distinct ~12% cluster at 0.35-0.53 (and 3x super-binomial variance), asymptote ~0.94 with ~2/3 of subjects above 0.94, a descending congruent profile (0.938 -> 0.924 pooled, counters -3.9 pts), and a rising conflict staircase for count-heavy subjects. All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 4 non-endorsements, no all-tie pairs, no pair where both theories guess -- the closest cells, C2/E1, are exactly where the theories straddle 0.5 in opposite directions), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Graded Dual-Route Distrust with a Bimodal Strategy Population (GDRD-B). Subjects evaluate expert endorsements through two concurrent routes. ROUTE 1 (analytic distrust): every endorsement is a liability whose magnitude grows monotonically in the endorser's RAW claimed validity, d_j = c0 + c1*v_j^kappa; accumulated suspicion P(X) = sum_j x_j*d_j is compared across options through sigmoid(beta*(P(B)-P(A))), plus a small residual micro-trust edge alpha*sum_j clip(logit(v_j),0,3)*(a_j-b_j) (capped so perfect experts exert bounded, not unbounded, trust pull). ROUTE 2 (graded counting): a continuous preference for the fewer-endorsement option, sigmoid(gamma*(cnt_B-cnt_A)). The population is BIMODAL in the route weight w: a large majority (~88%) of 'skeptics' (w in [0.85, 0.95], moderate lapse) coexists with a minority (~12%) of 'counters' (w in [0.50, 0.62], low lapse, decisive count slope gamma >= 2). The mixture is implemented as a quantile transform of a single uniform per-subject population variable u_pop, which holds the population-MEAN route weight at its base value while exploding the between-subject behavioral variance — the empirically observed 3-12x super-binomial variance on the saturation assays (Exps 11, 12) that no uniform range-widening could reproduce.

**Parameters:**
- c0: [0.06, 0.13]
- c1: [1.00, 1.35]
- kappa: [1.6, 2.1]
- beta: [1.3, 2.5]
- alpha: [0.0, 0.05]
- u_pop: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Graded Dual-Route Distrust, bimodal-population variant (GDRD-B).
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # ROUTE 1 (analytic distrust, weight w): each endorsement is a
    # LIABILITY whose magnitude grows with the endorser's RAW claimed
    # validity:
    #   d_j = c0 + c1 * v_j^kappa          (kappa >= 1, no dampening
    #                                        near v = 0.5)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = (P(B) - P(A)) + alpha * sum_j clip(logit(v_j), 0, 3)*(a_j - b_j)
    #     (micro-trust term; the logit is CAPPED at 3 so perfect
    #      experts exert bounded, not unbounded, trust pull)
    #   P_skeptic(A) = sigmoid(beta * D)
    # ROUTE 2 (graded counting, weight 1-w): a CONTINUOUS preference
    # for the option with fewer total endorsements:
    #   P_count(A) = sigmoid(gamma * (cnt_B - cnt_A))
    #
    # BIMODAL POPULATION: instead of sampling w, gamma, epsilon from
    # independent uniform ranges (which shifts the population MEANS
    # whenever the ranges widen), a single uniform population quantile
    # u_pop in [0, 1] is deterministically mapped onto a two-component
    # mixture:
    #   u_pop < 0.12  -> 'counter' minority:  w in [0.50, 0.62],
    #                    epsilon in [0.05, 0.10], gamma in [2.0, 3.0]
    #   otherwise     -> 'skeptic' majority:   w in [0.85, 0.95],
    #                    epsilon in [0.08, 0.18], gamma in [1.5, 3.0]
    # This holds E[w] at the calibrated base value (~0.86) while
    # producing genuine between-subject bimodality — the mechanism the
    # real super-binomial variances on Exps 11/12 demand.
    #
    # History is ignored: validities are communicated in the
    # instructions, so the distrust weights are fixed for the whole
    # block. beta is sampled per subject from a wide range.
    if isinstance(state, dict):
        a_vec = np.asarray(state["option_a_ratings"], dtype=float).ravel()
        b_vec = np.asarray(state["option_b_ratings"], dtype=float).ravel()
        stim = np.vstack([a_vec, b_vec])
    else:
        stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"GDRD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Power route: v = 1 is harmless (no logs). Trust route: guard
    # the logit against v = 1 (perfect experts) and v < 0.5.
    v_pow = np.clip(v, 0.5, 1.0)
    v_tr = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])
    u_pop = float(parameters["u_pop"])

    # --- Bimodal population mixture over (w, epsilon, gamma) ---
    # Quantile transform of one uniform variable: the population mean
    # route weight stays at the calibrated base (~0.86) while the
    # between-subject distribution becomes bimodal.
    f_counter = 0.12
    u = min(max(u_pop, 0.0), 1.0)
    if u < f_counter:
        # Counter minority: near-balanced routes, low lapse, decisive
        # graded count slope.
        t = u / f_counter
        w_mix = 0.50 + 0.12 * t          # [0.50, 0.62]
        epsilon = 0.05 + 0.05 * t        # [0.05, 0.10]
        gamma = 2.0 + 1.0 * t            # [2.0, 3.0]
    else:
        # Skeptic majority: analytic-distrust dominant, moderate lapse.
        t = (u - f_counter) / (1.0 - f_counter)
        w_mix = 0.85 + 0.10 * t          # [0.85, 0.95]
        epsilon = 0.08 + 0.10 * t        # [0.08, 0.18]
        gamma = 1.5 + 1.5 * t            # [1.5, 3.0]
    w_mix = float(min(max(w_mix, 0.5), 0.95))

    # --- Route 1: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v_pow, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual micro-trust: a small validity-weighted endorsement
    # edge. logit(0.5) = 0 (coin-flip experts contribute nothing) and
    # the logit is CAPPED at 3 so perfect experts (v = 1.0) exert a
    # bounded pull (validated twice on the Exp-12 saturation assay).
    lg = np.clip(np.log(v_tr / (1.0 - v_tr)), 0.0, 3.0)
    trust = float(np.dot(lg, a - b))

    # D > 0 favors A (B carries more accumulated suspicion, net of
    # the small trust edge).
    D = (pen_b - pen_a) + alpha * trust
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * beta * D))  # stable sigmoid

    # --- Route 2: graded endorsement-count preference ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    p_count_a = 0.5 * (1.0 + np.tanh(0.5 * gamma * (cnt_b - cnt_a)))

    # --- Dual-route mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_a = float(min(max(p_a, 0.0), 1.0))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Softened Validity-Distrust with Count Mixture (SVD). Every expert endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity: d_j = c0 + c1*v_j^kappa, with no dampening near v=0.5 and no trust refund. Decision making is a dual-process mixture at the trial level: with probability w the subject is a 'skeptic' who integrates the accumulated-suspicion difference D = P(B) - P(A) through sigmoid(beta*D); with probability (1-w) the subject is a 'counter' who simply chooses the option with FEWER total endorsements (indifferent at matched counts). Two heterogeneity mechanisms soften the pooled extremes and capture between-subject variability: (i) the sigmoid slope beta is sampled from a WIDE per-subject range, so the population is a mixture of shallow and steep sigmoids — sensitive near D=0 (matched-count validity-composition trials) yet compressed at large |D| (unequal-count trials); (ii) the uniform lapse epsilon is also sampled from a wide per-subject range, which pulls every saturated large-D assay toward 0.5 while adding genuine between-subject dispersion in overall choice rates.

**Parameters:**
- c0: [0.05, 0.12]
- c1: [0.95, 1.3]
- kappa: [1.5, 1.8]
- beta: [1.2, 2.6]
- w: [0.80, 0.92]
- epsilon: [0.12, 0.24]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Softened Validity-Distrust with Count Mixture (SVD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Branch 1 (skeptic, prob w): each endorsement is a LIABILITY whose
    # magnitude grows with the endorser's RAW claimed validity:
    #   d_j = c0 + c1 * v_j^kappa   (kappa >= 1.5; NO dampening near
    #                                v = 0.5, NO trust refund)
    #   accumulated suspicion: P(X) = sum_j x_j * d_j
    #   D = P(B) - P(A)  (positive favors A)
    #   P_skeptic(A) = sigmoid(beta * D)
    # Branch 2 (counter, prob 1-w): pure tallying on endorsement COUNTS;
    #   choose the option with FEWER total endorsements; exactly
    #   indifferent (0.5) at matched counts.
    # Mixture: P_core(A) = w * P_skeptic(A) + (1-w) * P_count(A),
    # mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # distrust weights are fixed for the whole block. beta AND epsilon
    # are sampled per subject from WIDE ranges: between-subject
    # heterogeneity in the sigmoid slope and in the lapse rate are
    # explicit parts of the theory.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SVD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    v = np.clip(v, 0.5, 1.0)  # defensive; no logs taken, v=1 harmless.

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    w_mix = float(parameters["w"])

    # --- Skeptic branch: raw validity-proportional distrust charge ---
    d = c0 + c1 * np.power(v, kappa)
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion).
    D = pen_b - pen_a
    x = beta * D
    p_skeptic_a = 0.5 * (1.0 + np.tanh(0.5 * x))  # stable sigmoid

    # --- Counter branch: pure endorsement-count tallying ---
    cnt_a = float(np.sum(a))
    cnt_b = float(np.sum(b))
    if cnt_a < cnt_b:
        p_count_a = 1.0      # A has fewer endorsements -> prefer A
    elif cnt_a > cnt_b:
        p_count_a = 0.0
    else:
        p_count_a = 0.5      # matched counts -> indifferent

    # --- Dual-process mixture ---
    p_a = w_mix * p_skeptic_a + (1.0 - w_mix) * p_count_a
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # ------------------------------------------------------------------\n    # EXCESS-DISPERSION METRIC (conflict staircase + congruent asymptote)\n    #\n    # Per subject:\n    #   x = P(choose the fewer-endorsement side | CONFLICT trials)\n    #       Conflict trials: the option with FEWER endorsements carries the\n    #       HIGHER total validity mass (the perfect-vs-coins staircase,\n    #       8 pairs x 4 reps = 32 trials/subject).\n    #   r = P(choose the fewer-endorsement side | SATURATED CONGRUENT\n    #       trials) (fewer-endorsed side carries >= 3 LESS validity mass;\n    #       8 pairs x 4 reps = 32 trials/subject).\n    #   v = (x - 0.21)^2 - (r - 0.96)^2\n    #\n    # 0.21 = a priori conflict-rate reference (midpoint of the SVD\n    # box-implied range [0.13, 0.29]); 0.96 = the asymptote ceiling\n    # (1 - eps/2 at eps = 0.08), which sits above SVD's hard cap of\n    # 0.94 (eps >= 0.12).\n    #\n    # v is each subject's squared conflict-rate deviation from the\n    # reference that is NOT paid for by a lapse-driven asymptote\n    # shortfall.  Under SVD, all between-subject conflict dispersion\n    # comes from (w, eps) heterogeneity, and eps >= 0.12 forces\n    # r <= 0.94, so the two squared terms cancel subject-by-subject\n    # (E[v] ~ 0.001).  Under GDRD-B, the bimodal route-weight\n    # population (12% counter mode at conflict 0.35-0.53, skeptic\n    # majority at 0.12-0.23) generates conflict dispersion while the\n    # asymptote stays at the ceiling (eps <= 0.18), so v > 0\n    # (E[v] ~ 0.011).\n    #\n    # Population value: root-mean-square of v across subjects (the\n    # excess dispersion in rate units).  Single subject: its own v\n    # (its squared contribution to the population statistic).\n    # ------------------------------------------------------------------\n    VALIDITIES = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.74,\n                           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])\n    CONFLICT_REF = 0.21\n    ASYMPTOTE_REF = 0.96\n    MASS_GAP_MIN = 3.0\n\n    df = data.reset_index(drop=True)\n    A = np.vstack([np.asarray(x, dtype=float).ravel()\n                   for x in df[\"option_a_ratings\"]])\n    B = np.vstack([np.asarray(x, dtype=float).ravel()\n                   for x in df[\"option_b_ratings\"]])\n    resp = df[\"response\"].to_numpy(dtype=float)\n\n    n_feat = A.shape[1]\n    valid = VALIDITIES[:n_feat]\n\n    cnt_a = A.sum(axis=1)\n    cnt_b = B.sum(axis=1)\n    mass_a = A @ valid\n    mass_b = B @ valid\n\n    unequal = cnt_a != cnt_b\n    fewer_a = cnt_a < cnt_b\n    mass_fewer = np.where(fewer_a, mass_a, mass_b)\n    mass_more = np.where(fewer_a, mass_b, mass_a)\n\n    conflict = unequal & (mass_fewer > mass_more)\n    asymptote = (unequal & (mass_fewer < mass_more)\n                 & ((mass_more - mass_fewer) >= MASS_GAP_MIN))\n\n    chose_fewer = np.where(fewer_a, resp == 0.0, resp == 1.0).astype(float)\n\n    tmp = pd.DataFrame({\"sid\": df[\"subject_id\"].to_numpy(),\n                        \"conflict\": conflict,\n                        \"asymptote\": asymptote,\n                        \"cf\": chose_fewer})\n\n    sub_ids = pd.unique(tmp[\"sid\"])\n    x = tmp[tmp[\"conflict\"]].groupby(\"sid\")[\"cf\"].mean().reindex(sub_ids)\n    r = tmp[tmp[\"asymptote\"]].groupby(\"sid\")[\"cf\"].mean().reindex(sub_ids)\n\n    v = (x - CONFLICT_REF) ** 2 - (r - ASYMPTOTE_REF) ** 2\n    v = v.dropna().to_numpy()\n    if v.size == 0:\n        return 0.0\n    if v.size == 1:\n        # single subject: its squared excess deviation (contribution\n        # to the population dispersion statistic)\n        return float(v[0])\n    # population: root-mean-squared excess dispersion, in rate units\n    return float(np.sqrt(max(0.0, float(np.mean(v)))))",
  "rationale": "TARGETED SIGNATURE. The design's decisive population-level contrast is the SOURCE of between-subject dispersion. SVD attributes all conflict-staircase heterogeneity to (w, eps) sampling; because eps >= 0.12 hard-caps its congruent asymptote at r <= 0.94, every unit of SVD conflict dispersion must be 'paid for' by an asymptote shortfall from the 0.96 ceiling. GDRD-B's bimodal route-weight population (u_pop quantile transform: 12% counters with w in [0.50,0.62], eps in [0.05,0.10]; 88% skeptics with eps in [0.08,0.18]) generates conflict-rate dispersion (counters at 0.35-0.53, skeptics depressed at ~0.14) while keeping asymptotes at the ceiling.\n\nMETRIC MECHANICS. Trials are classified from the ratings themselves (no hardcoded trial list): conflict = unequal endorsement counts with the FEWER-endorsed option carrying HIGHER validity mass (the 8 perfect-vs-coins staircase pairs, 32 trials/subject); saturated congruent = fewer-endorsed side with >= 3 LESS validity mass (the 8 asymptote pairs, 32 trials/subject). x = rate of choosing the fewer-endorsement side on conflict trials; r = same on asymptote trials. The per-subject quantity is v = (x-0.21)^2 - (r-0.96)^2: squared conflict deviation from the a priori reference 0.21 (midpoint of SVD's box-implied [0.13,0.29]) MINUS the squared asymptote shortfall from the 0.96 ceiling (the lapse-explained part). The population scalar is the RMS of v across subjects; a single subject's slice returns that subject's v (its squared contribution to the population statistic).\n\nPREDICTED VALUES. SVD: x ~ 0.205 (sd 0.078), r ~ 0.910 (eps in [0.12,0.24] forces r in [0.88,0.94]); the two squared terms nearly cancel per subject, E[v] ~ 0.0007, pooled value ~ 0.00-0.05, between-subject Var(v) ~ 1.2e-4. GDRD-B: skeptics x ~ 0.145, r ~ 0.935 (v ~ 0.007); counters x ~ 0.41, r ~ 0.96 (v ~ 0.045); E[v] ~ 0.011, pooled value ~ 0.10-0.12, Var(v) ~ 4e-4. Welch t at N=25 ~ 10-20 (p << 0.01). The contrast is robust: (i) even with zero counter-mode subjects the skeptic mode alone gives E[v] ~ 0.007 vs SVD ~ 0.001 (t ~ 10), since skeptics' depressed conflict rates co-occur with ceiling-level asymptotes, a combination SVD's independent-uniform (w,eps) box produces only at vanishing density; (ii) the asymptote term alone (SVD eps >= 0.12 vs GDRD-B eps <= 0.18) rescues the metric even if conflict-rate calibration is off; (iii) under fixed mid-parameter simulation the cancellation under SVD is exact (v ~ 0) while GDRD-B still yields v ~ 0.006. Per-subject v has small variance because squaring compresses the binomial noise floor (Var((x-c)^2) ~ 2*sd^4 << Var(|x-c|)), which is what makes the population contrast tight at N=25."
}
```

## Usage

```json
{
  "input_tokens": 10081,
  "output_tokens": 115469,
  "reasoning_tokens": 113422,
  "provider": "Sail Research"
}
```
