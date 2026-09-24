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
**Validities (n_features=6):** [0.99, 0.99, 0.5, 0.5, 0.5, 0.99]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 10: A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 12: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 13: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 16: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 17: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 19: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 20: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 21: A=[0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 23: A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 24: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 25: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 26: A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 27: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 28: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 29: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 30: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. RTDT and PBDT share the distrust-inversion kernel, the floored convex primacy gate, and the multiplicative bracket w_j ∝ g_j·[t_j·(1-τ)/n + τ·inv_j]; they differ in exactly one quantitatively decisive respect: the SHAPE OF THE TRUST RAMP on the flat-share (trusted) channel. RTDT's ramp is GEOMETRIC from a crushed floor to a SUPER-UNITY ceiling: t_j = t0·(t1/t0)^((j/(n-1))^κ) with t0 ∈ [0.03, 0.12] and t1 ∈ [1.0, 1.6] — an early trusted expert's flat share is multiplied by as little as 0.03 (annihilated) while a late trusted expert's is AMPLIFIED up to 1.6x ABOVE the flat share. PBDT's ramp is a capped power law: t_j = t0 + (t1-t0)·(j/(n-1))^κ with t0 ∈ [0.25, 0.35] and t1 ≤ 1.0 — the early trusted expert keeps 25-35% of the flat share exactly where the gate is maximal, and the late trusted expert can never exceed the flat share. Consequence (raw weights = g_j·t_j·(1-τ)/n, verified at every corner of both boxes): the position-1 trusted weight is DISJOINT — RTDT [0.0006, 0.0064] vs PBDT [0.0079, 0.0200]; the combined positions-1+2 trusted weight is DISJOINT — RTDT ≤ 0.0095 < 0.0116 ≤ PBDT; the trusted-channel position ORDERING is opposite — PBDT requires ŵ(99@1) > ŵ(99@2) > ŵ(99@6) at every corner (ratios ≥ 1.31 and ≥ 1.27), while RTDT allows ŵ(99@2)/ŵ(99@1) up to 4.7 and ŵ(99@6)/ŵ(99@2) up to 6.7; and the late-trusted weight ŵ(99@6): RTDT [0.0010, 0.0060] vs PBDT [0.0017, 0.0038] — RTDT's super-unity t1 corner exceeds PBDT's maximum by 1.6x. The estimated ratio ŵ(99@1)/ŵ(99@6) must land in [2.70, 9.34] under PBDT and can fall to 0.17 under RTDT.

LAYOUT (n=6): v = [0.99, 0.99, 0.55, 0.55, 0.55, 0.99]. Trusted 99% experts sit at the three gate anchors — position 1 (amplified gate (6/5)^η ≈ 2.3-3.8), position 2 (gate EXACTLY 1.0), position 6 (gate floor δ) — giving maximal position contrast at tied validity. The three 55% experts occupy positions 3-5 (gate tail). Because all kernel cues share one validity, inv_j/Z_v = 1/3 EXACTLY for each, parameter-free: γ and c drop out of every prediction, isolating the ramp/gate architecture (the only place the boxes differ). The 0.99 validity kills kernel leakage ((0.0095/0.4475)^γ ≤ 6×10^-5), so the 99% experts' weights are pure flat-share × ramp × gate. The kernel pair 55@4 vs 55@5 provides a finely calibrated counterweight Δ45 = (g_4-g_5)·τ/3: RTDT [0.0055, 0.0127], PBDT [0.0042, 0.0078] — exactly the scale of the trusted weights, so trusted endorsements can single-handedly flip it or fail to, depending on the theory.

AXIS 1 — TRUSTED-POSITION DUELS (D16, D12, D26 + mirrors; 55s tied, so choice is decided by the trusted channel alone). PBDT's strictly-decreasing trusted profile requires the EARLIER product at every corner on all three duels: D16 P(early) ∈ [0.51, 0.58], D12 ∈ [0.50, 0.58], D26 ∈ [0.50, 0.54]. RTDT's crushed-t0/super-unity-t1 ramp predicts near-chance or LATE-tilted choices: D16 ∈ [0.49, 0.52], D12 ∈ [0.48, 0.53], D26 ∈ [0.486, 0.52] with the box extending BELOW chance at t1=1.6 corners. Falsification caps: mirror-corrected P(early) ≥ 0.55 on D16 (or ≥ 0.54 pooled over the duel family) is impossible for RTDT at any admissible parameter combination; P(early) ≤ 0.50 is impossible for PBDT. A reliable preference for the LATER 99%-expert's product on any duel kills PBDT's entire space.

AXIS 2 — EARLY-TRUSTED RESCUE BATTERY (R0, R1, R2, R6, R12 + mirrors; the headline). Base rung R0: A wins 55@4 vs B wins 55@5 — both theories pick A (gate gap Δ45). The R-cells add 99% endorsement(s) to B's side, testing whether the trusted channel closes Δ45. R12 (helper = 99@1 AND 99@2): PBDT's combined early-trusted weight (≥ 0.0116, disjoint from RTDT's ≤ 0.0095) overcomes the gap at EVERY corner — d < 0 throughout, P(A) ≤ 0.485; RTDT straddles with central P(A) 0.51-0.54. DISJOINT-ISH BOXES: observed P(A) ≥ 0.52 on R12 is unreachable for PBDT; ≤ 0.48 unreachable for RTDT. R1 (helper = 99@1 alone): PBDT flips to B at every corner (P(B) ≥ 0.50, up to 0.56); RTDT stays with A at 27/28 corners (single-corner exception at t0=0.12, η=7.5, δ=0.12, τ=0.91). R2 (helper = 99@2, gate exactly 1): RTDT's ŵ(99@2) ≤ 0.0048 NEVER closes its Δ45 ≥ 0.0055 — P(A) ≥ 0.50 at every RTDT corner (min raw margin +0.0007); PBDT straddles (ŵ(99@2) ∈ [0.0037, 0.0077] vs Δ45 ∈ [0.0042, 0.0078]). The within-subject R1-vs-R2 double difference isolates ŵ(99@1)-ŵ(99@2): robustly positive under PBDT (its trusted profile is strictly decreasing), sign-ambiguous under RTDT. R6 (helper = 99@6, the LATE trusted expert): PBDT's ŵ(99@6) ≤ 0.0038 never closes its Δ45 ≥ 0.0042 — P(A) ≥ 0.51 at every corner, so observed P(B) ≥ 0.52 on R6 is IMPOSSIBLE for PBDT; RTDT's super-unity corner (δ=0.25, t1=1.6: ŵ(99@6)=0.0060 > Δ45=0.0047) UNIQUELY produces the late-trusted rescue. These cells are also 1-vs-2 and 1-vs-3 count trials on which both theories pick the single-cue product — excluding tallying.

AXIS 3 — SWING BATTERY (S1, S2, S6 vs R0/R-cells + mirrors). Adding a 99%@p endorsement to the kernel-winning side (S-cells) versus to the kernel-losing side (R-cells) yields within-subject double differences equal to 2·ŵ(99@p), a direct behavioral estimate of the trusted-channel position profile (ŵ(99@1), ŵ(99@2), ŵ(99@6)). PBDT requires the estimated profile to be strictly decreasing with ŵ(99@1)/ŵ(99@6) ≥ 2.70; RTDT allows an increasing profile with the ratio down to 0.17. An estimated ratio ≤ 2.6 excludes PBDT's entire box; ≥ 3.6 excludes RTDT's. Absolute swing@1: PBDT's d-shift ∈ [0.024, 0.095] vs RTDT's ∈ [0.002, 0.038] — a 2-10x dissociation in dose-response.

AXIS 4 — SEESAW CELLS (X16, X26 + mirrors; RTDT-robust, two-sided adversarial cells). X16: A wins {99@1, 55@5} vs B wins {99@6, 55@4}: the kernel gap and the late trusted endorsement both favor B, and RTDT's early-trusted weight (≤ 0.0064) never overcomes their sum — d < 0 at EVERY RTDT corner (verified corner-by-corner, worst margin -0.0009); PBDT straddles (d ∈ [-0.003, +0.014]) because ITS early-trusted weight (up to 0.020) sometimes does. X26 replicates with the position-2 trusted expert (RTDT d < 0 at every corner; PBDT straddles). Observed P(A) ≥ 0.53 on X16 is very hard for RTDT; a robust B-preference pattern {X16: B, X26: B, D16: chance-or-late} is RTDT-only. X-cells are also anti-TTB (the top 99% expert's product loses under both theories).

AXIS 5 — SCREENS AND ANCHORS. DINV (the three 55s' product vs the three 99s' product, 3-3 count tie): both theories confidently prefer the 55-product (d_norm ≥ 0.78, P ≥ 0.80) — verifies subjects read and distrust-invert the stated validities; excludes normative weighting, TTB, and tallying (exact 0.5). VETO (all three 99s vs the single 55@5): both theories prefer the 55-product — the product endorsed by ALL THREE best experts loses; anti-TTB, anti-normative, anti-tallying (3 cues beat 1). TOT: comprehension anchor (both ≥ 0.90).

HONESTY ABOUT EFFECT SIZES. Both boxes pin τ ≥ 0.91, capping the flat share at 1.5% per cue, so trusted-channel single-cell proportions are confined to [0.43, 0.66] — the same regime as the accepted trusted-axis designs. Statistical power comes from the within-subject structure: every cell type appears in A/B-mirrored form with K = 3 repetitions (6 trials per cell per subject), and the decisive readouts are family-level contrasts (duel-family mean, R1-R2 and S-R double differences, the fitted swing profile). The kernel channel (Δ45, Δ34) and the screens carry the larger, theory-agreeing effects that verify task engagement.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on RTDT reference evidence (geometric ramp t0=0.06, t1=1.3, κ=0.8, η=6.5, δ=0.18, τ=0.93) versus PBDT reference evidence (t0=0.30, t1=0.975, κ=1.0, η=7.2, δ=0.20, τ=0.92) loads with opposite profiles: the two evidence axes differ almost exclusively on the early-trusted cells (D16/D12/R1/R12/S1/X16), where PBDT's position-1 term is 2-10x RTDT's, and on the late-trusted cells (R6/D26/X-family), where RTDT's super-unity t1 term can exceed PBDT's by 1.6x. Single-number readouts: (i) duel-family early-trusted preference (PBDT ≥ 0.51 everywhere, up to 0.58; RTDT ≤ 0.52); (ii) R12 sign (PBDT < 0.5 at every corner; RTDT central > 0.5); (iii) R2 sign (RTDT ≥ 0.50 at every corner); (iv) R6 late-rescue cap (P(B) ≥ 0.52 impossible for PBDT); (v) the fitted ŵ(99@1)/ŵ(99@6) swing ratio (disjoint requirement: ≥ 2.70 vs can-be < 1). 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** RECIPROCAL-TRUST DISTRUST TALLYING (RTDT), gate-steepened retune. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate now at the steeper/low-floored corner (eta in [5.5, 7.5], delta in [0.12, 0.25]) so the first one or two experts carry the dominant share of attention while the late tail retains only a small floor residue. Stated validities are distrust-inverted (treated as error rates) via a moderately steep kernel inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma backed off to [2.7, 3.3] and tau in [0.91, 0.95] — the calibration the iter-2 experiment verified moves the anti-validity slope, validity adherence, and recency composites toward their observed values without overshooting. The reciprocal trust ramp is retained at its moderate band: t_j = t0*(t1/t0)^((j/(n-1))^kappa) with t0 in [0.03, 0.12] (early trusted claims heavily discounted but not annihilated), super-unity t1 in [1.0, 1.6] amplifying late high-validity experts above the flat share, and kappa in [0.6, 1.0] so trust recovers early in the tail. Weights combine multiplicatively, w_j proportional to g_j*[t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with the softened noise box (beta [2.5, 4.0], epsilon [0.12, 0.20]) that iter-2 verified improves the tally-allegiance contrast while the anchors retain headroom.

**Parameters:**
- tau: [0.91, 0.95]
- eta: [5.5, 7.5]
- delta: [0.12, 0.25]
- gamma: [2.7, 3.3]
- c: [0.05, 0.25]
- t0: [0.03, 0.12]
- t1: [1.0, 1.6]
- kappa: [0.6, 1.0]
- beta: [2.5, 4.0]
- epsilon: [0.12, 0.20]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # RTDT: Reciprocal-Trust Distrust Tallying, GATE-STEEPENED retune.
    # Minimal-diff edit of the ACCEPTED iter-1 base per the critic's
    # iter-2 directive. The iter-2 experiment decomposed cleanly:
    #   WINNER (kept verbatim): the kernel backoff -- gamma [3.0,4.2] ->
    #     [2.7, 3.3], tau [0.92,0.96] -> [0.91, 0.95]. This moved
    #     Exp3 (-0.62 -> -0.557, obs -0.576), Exp5 (0.196 -> 0.209),
    #     Exp9 (0.328 -> 0.343, obs 0.341), Exp10 (-0.228 -> -0.190)
    #     and Exp2 (0.304 -> 0.289) all toward observed.
    #   LOSER (reverted): the trust-ramp extremes (t0 crushed to
    #     [0,0.05], t1 widened to [1.0,2.0], kappa lowered to [0.5,0.9])
    #     broke Exp18 (-0.97 -> -3.80) and flattened the g*t primacy
    #     product on Exps 11-15 without moving their intended targets
    #     (Exp14 rescue went UP; Exp8 FEDI untouched). Reverted to the
    #     moderate band: t0 [0.03, 0.12], t1 [1.0, 1.6], kappa [0.6, 1.0].
    #   NEW LEVER: the primacy gate steepened -- eta [4.5,6.5] ->
    #     [5.5, 7.5], delta [0.15,0.30] -> [0.12, 0.25]. The Exp14
    #     rescue trials pivot on the early 50%-validity expert vs late
    #     50% helpers, so a steeper gate pushes rescue toward the lapse
    #     floor (~0.13 observed); simultaneously Exps 11/12/13/15 all
    #     UNDERSHOOT their primacy composites, and steepening the gate
    #     and reverting the ramp act in the same direction on g*t.
    #   Noise box kept softened: beta [2.5, 4.0], epsilon [0.12, 0.20]
    #     (iter-2 verified Exp2 improved to ~1.8 SD with anchor headroom).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RTDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share (BACKED OFF)
    eta = float(parameters["eta"])        # primacy-gate convexity (STEEPENED)
    delta = float(parameters["delta"])    # primacy-gate floor (LOW)
    gamma = float(parameters["gamma"])    # distrust-inversion steepness (BACKED OFF)
    c = float(parameters["c"])            # partial centering of error rates
    t0 = float(parameters["t0"])          # early-trust floor (MODERATE, reverted)
    t1 = float(parameters["t1"])          # late-trust residue (SUPER-UNITY, reverted)
    kappa = float(parameters["kappa"])    # reciprocal-ramp exponent (reverted)
    beta = float(parameters["beta"])      # choice consistency (SOFTENED)
    eps = float(parameters["epsilon"])    # uniform lapse (SOFTENED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.7, 3.3] with tau in [0.91, 0.95] is the verified
    # calibration: steep enough to carry the anti-validity gradient in
    # validity-position-confounded layouts, flat enough not to
    # over-anti-follow high-validity cues (Exp3/5/8/10 all verified).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) RECIPROCAL TRUST RAMP at the moderate band, computed in
        # log space for numerical stability:
        #   log t_j = log t0 + (j/(n-1))^kappa * (log t1 - log t0).
        # t0 in [0.03, 0.12] discounts early trusted claims without
        # annihilating them (the iter-2 t0~0 corner exploded Exp18);
        # t1 in [1.0, 1.6] with kappa in [0.6, 1.0] amplifies late
        # trusted experts above the flat share moderately.
        t0_eff = max(t0, 1e-4)  # guard the geometric form against t0 = 0
        t1_eff = max(t1, 1e-4)
        frac = np.power(pos / (n - 1.0), kappa)  # in [0, 1]
        t = np.exp(np.log(t0_eff) + frac * (np.log(t1_eff) - np.log(t0_eff)))
        # (3) STEEP FLOORED CONVEX PRIMACY GATE, steepened per the
        # critic's iter-2 directive: eta in [5.5, 7.5], delta in
        # [0.12, 0.25]. The steeper gate pushes Exp14's rescue rate
        # toward the lapse floor and strengthens the primacy composites
        # on Exps 11/12/13/15, which all currently undershoot.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. The trusted channel is modulated by the reciprocal
    # ramp -- heavily discounted early, moderately amplified late --
    # while the distrust kernel (tau >= 0.91) dominates the bracket
    # elsewhere.
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # (6) Numerically stable logistic choice rule -- beta in [2.5, 4.0]
    # with a uniform lapse epsilon in [0.12, 0.20]: the softened noise
    # box the iter-2 experiment verified (Exp2 improved toward 0.158
    # while Exp4 anchors retain headroom above 0.82).
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** POSITION-BRIDGED DISTRUST TALLYING (PBDT), un-bundled-gate edition. Subjects scan expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, with the gate restored to the steep end of its box (eta high, delta low) so late low-validity helper cues are crushed relative to early ones — the structural lever separating primacy-rescue trials from validity-position-confounded layouts. Stated validities are distrust-inverted (treated as error rates) via inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z, kept at the steep bridged range [2.6, 2.9] with a high distrust share tau, so the anti-validity gradient in confounded layouts is carried by the distrust kernel alone rather than by the gate. Trust in high-validity experts ramps with position from a small-but-nonzero floor t_j = t0 + (t1-t0)*(j/(n-1))^kappa, with a slightly concave-leaning ramp (kappa below 1) so late trusted experts recover nearly the full flat share without raising the early floor. Weights combine multiplicatively, w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon, with noise set between the two previous iterations so slope/contrast metrics that were over-attenuated recover while anchor allegiances keep their margin.

**Parameters:**
- tau: [0.91, 0.93]
- eta: [6.5, 8.0]
- delta: [0.15, 0.25]
- gamma: [2.6, 2.9]
- c: [0.05, 0.25]
- t0: [0.25, 0.35]
- t1: [0.95, 1.00]
- kappa: [0.8, 1.5]
- beta: [3.6, 4.4]
- epsilon: [0.11, 0.14]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PBDT: Position-Bridged Distrust Tallying, UN-BUNDLED-GATE edition
    # (iter-3). Minimal-diff edit of the ACCEPTED iter-2 base per the
    # critic's iter-2 directive, which diagnosed that the iter-2 move (b)
    # had bundled an efficacious half (distrust strengthening: the Exp 3
    # win) with a counterproductive half (gate flattening: the Exp 14
    # and Exp 13 regressions). This iteration UN-BUNDLES them:
    #   (A) REVERT THE GATE FLATTENING, KEEP THE DISTRUST STRENGTHENING:
    #       eta [5.5,6.0] -> [6.5, 8.0] (steep end), delta [0.25,0.30]
    #       -> [0.15, 0.25] (low floor), while gamma stays [2.6, 2.9]
    #       and tau stays [0.91, 0.93] exactly as in iter-2. The steep
    #       gate crushes the late 50%-validity helpers on Exp 14's
    #       trusted-early side (0.350 -> toward 0.132) and restores the
    #       M/C position families on Exp 13 (1.35 -> toward 1.84);
    #       the anti-validity gradient in confounded layouts (Exp 3)
    #       is carried by the distrust kernel, which iter 2 verified.
    #   (B) PARTIALLY ROLL BACK THE NOISE: beta [3.2,4.0] -> [3.6, 4.4]
    #       and epsilon [0.13,0.16] -> [0.11, 0.14] -- between the
    #       iter-1 and iter-2 settings. Lifts the over-attenuated slope
    #       metrics (Exps 1, 11, 12, 13, 15) at a small cost on Exps
    #       4/7, which currently have margin. Exp 2 is held as a
    #       monitored constraint (structural, noise-insensitive).
    #   (C) WIDEN THE RAMP CONVEXITY DOWNWARD: kappa [1.0,1.5] ->
    #       [0.8, 1.5], a concave-leaning ramp so the late trusted
    #       expert recovers more of the flat share (Exp 8 FEDI
    #       0.294 -> toward 0.396) without touching t0, tau, or the
    #       gate. t0/t1 unchanged from iter-2.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PBDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket (HIGH, kept)
    eta = float(parameters["eta"])        # convexity of the primacy gate (STEEP, reverted)
    delta = float(parameters["delta"])    # floor of the primacy gate (LOW, reverted)
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (kept)
    c = float(parameters["c"])           # partial centering of error rates
    t0 = float(parameters["t0"])          # trust floor at position 1 (LOW, kept)
    t1 = float(parameters["t1"])          # trust residue at the last position (HIGH, kept)
    kappa = float(parameters["kappa"])   # convexity of the trust ramp (WIDENED DOWN)
    beta = float(parameters["beta"])     # choice consistency (PARTIALLY RESTORED)
    eps = float(parameters["epsilon"])   # uniform lapse (PARTIALLY RESTORED)

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma in [2.6, 2.9] and tau in [0.91, 0.93] are preserved verbatim
    # from iter-2 -- the verified carriers of the anti-validity gradient
    # in validity-position-confounded layouts (Exp 3).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and position alone carries structure.
        w_inv = np.full(n, 1.0 / n)

    if n > 1:
        pos = np.arange(n, dtype=float)
        # (2) TRUST RAMP with widened-down convexity: low floor t0,
        # high t1, and kappa now allowed below 1.0 so the ramp is
        # concave-leaning -- the late trusted expert recovers more of
        # the flat share earlier in the tail (the Exp 8 FEDI lever)
        # without raising the early floor that Exp 14 needs kept low.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) PRIMACY GATE REVERTED TO THE STEEP END: eta in [6.5, 8.0]
        # with delta in [0.15, 0.25]. The steep, low-floored gate
        # concentrates attention on the first one or two experts and
        # crushes the late tail -- the structural fix for Exp 14's
        # late-helper over-weighting and Exp 13's M/C position
        # families. The anti-validity regression in confounded layouts
        # is carried by the (preserved) steep distrust kernel instead.
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket
    # cue-by-cue. tau in [0.91, 0.93] keeps the distrust channel
    # dominant; the trusted channel is carried by the ramp (small
    # early, near-full and now earlier-recovering late).
    bracket = t * (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (5) Deterministic attention: every discriminating cue is attended;
    # no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule -- beta in [3.6, 4.4],
    # between the iter-1 and iter-2 calibrations, so the slope metrics
    # over-attenuated by iter-2's noise (Exps 1, 11, 12, 13, 15)
    # recover while Exps 4/7 keep their remaining margin.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Uniform lapse, epsilon in [0.11, 0.14]: partially rolled back
    # from iter-2's [0.13, 0.16] per the critic's split-the-difference
    # directive. Exp 2 is NOT chased with further noise -- it is held
    # as a monitored structural constraint of the distrust family.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"Box-averaged, precision-weighted theory discriminant (RTDT vs PBDT).\n\n    For every trial we compute each theory's predicted P(choose A), averaged\n    over an ensemble of parameter settings that spans that theory's box\n    (centre + a factorial corner grid over the decisive axes: tau, t0, the\n    noise pair (beta, epsilon), and the gate pair (eta, delta), plus kappa/t1\n    tails).  The per-trial contrast dp = P_PBDT - P_RTDT is then weighted by\n    its Bernoulli precision 1/(pbar*(1-pbar)) -- the linearised log-likelihood\n    ratio -- and summed against the observed signed choice.  Trials where the\n    two theories make visibly different predictions (the early-trusted rescue\n    cell R12, the trusted-position duels D16/D12/D26, the swing cells\n    R1/S1/X16, and the high-signal screens TOT/DINV where the theories'\n    beta/epsilon boxes differ) carry the load; theory-agreeing trials get\n    ~zero weight.  Positive values = PBDT-like behaviour (following the early\n    99%-expert, higher determinism); negative = RTDT-like.\n    \"\"\"\n\n    def _vec(x):\n        if isinstance(x, np.ndarray):\n            return x.astype(float)\n        if isinstance(x, str):\n            toks = x.strip().strip('[](){}').replace(',', ' ').split()\n            return np.array([float(t) for t in toks], dtype=float)\n        return np.asarray(list(x), dtype=float)\n\n    A = np.vstack([_vec(r) for r in data['option_a_ratings'].values])\n    B = np.vstack([_vec(r) for r in data['option_b_ratings'].values])\n    resp = np.asarray(data['response'].values, dtype=float)\n    s = 1.0 - 2.0 * resp  # +1 if chose A, -1 if chose B\n    T, n = A.shape\n    if T == 0 or n == 0:\n        return 0.0\n\n    # Design validities (0.50 vs 0.55 for the kernel cues is immaterial:\n    # the three kernel cues share one validity either way, and the 99%\n    # cues' distrust leakage is ~1e-5).\n    if n == 6:\n        v = np.array([0.99, 0.99, 0.55, 0.55, 0.55, 0.99], dtype=float)\n    else:\n        v = np.linspace(0.95, 0.55, n)\n\n    def _predict(tau, eta, delta, gamma, c, t0, t1, kappa, beta, eps, ramp):\n        pos = np.arange(n, dtype=float)\n        dpos = float(max(n - 1, 1))\n        err = 1.0 - v\n        centered = np.maximum(err - c * float(err.min()), 0.0)\n        inv = np.power(centered, gamma)\n        Z = float(inv.sum())\n        w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)\n        if ramp == 'geo':  # RTDT geometric reciprocal-trust ramp\n            lt0 = np.log(max(t0, 1e-4))\n            lt1 = np.log(max(t1, 1e-4))\n            t = np.exp(lt0 + np.power(pos / dpos, kappa) * (lt1 - lt0))\n        else:              # PBDT capped power-law ramp\n            t = t0 + (t1 - t0) * np.power(pos / dpos, kappa)\n        g = delta + (1.0 - delta) * np.power((n - pos) / dpos, eta)\n        bracket = t * (1.0 - tau) / n + tau * w_inv\n        raw = g * bracket\n        w = raw / float(raw.sum())\n        d = np.sign(A - B) @ w\n        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))\n        return (1.0 - eps) * p + 0.5 * eps\n\n    # ---- RTDT ensemble (geometric ramp, box [t0 .03-.12, t1 1.0-1.6]) ----\n    rb = dict(gamma=3.0, c=0.15, t1=1.3, kappa=0.8)\n    draws_R = [dict(tau=0.93, eta=6.5, delta=0.18, t0=0.06, beta=3.2, eps=0.16, **rb)]\n    for tau in (0.91, 0.95):\n        for t0 in (0.03, 0.12):\n            for beta, eps in ((2.5, 0.20), (2.5, 0.12), (4.0, 0.20), (4.0, 0.12)):\n                for eta, delta in ((5.5, 0.25), (7.5, 0.12)):\n                    draws_R.append(dict(tau=tau, eta=eta, delta=delta, t0=t0,\n                                        beta=beta, eps=eps, **rb))\n    draws_R.append(dict(tau=0.93, eta=6.5, delta=0.18, t0=0.075, beta=3.25, eps=0.16,\n                        gamma=3.0, c=0.15, t1=1.6, kappa=0.6))\n    draws_R.append(dict(tau=0.93, eta=6.5, delta=0.18, t0=0.075, beta=3.25, eps=0.16,\n                        gamma=3.0, c=0.15, t1=1.0, kappa=1.0))\n\n    # ---- PBDT ensemble (power ramp, box [t0 .25-.35, t1 .95-1.0]) ----\n    pb = dict(gamma=2.75, c=0.15, t1=0.975, kappa=1.15)\n    draws_P = [dict(tau=0.92, eta=7.2, delta=0.20, t0=0.30, beta=4.0, eps=0.125, **pb)]\n    for tau in (0.91, 0.93):\n        for t0 in (0.25, 0.35):\n            for beta, eps in ((3.6, 0.14), (3.6, 0.11), (4.4, 0.14), (4.4, 0.11)):\n                for eta, delta in ((6.5, 0.25), (8.0, 0.15)):\n                    draws_P.append(dict(tau=tau, eta=eta, delta=delta, t0=t0,\n                                        beta=beta, eps=eps, **pb))\n    draws_P.append(dict(tau=0.92, eta=7.2, delta=0.20, t0=0.30, beta=4.0, eps=0.125,\n                        gamma=2.75, c=0.15, t1=1.0, kappa=0.8))\n    draws_P.append(dict(tau=0.92, eta=7.2, delta=0.20, t0=0.30, beta=4.0, eps=0.125,\n                        gamma=2.75, c=0.15, t1=0.95, kappa=1.5))\n\n    pR = np.mean([_predict(ramp='geo', **dr) for dr in draws_R], axis=0)\n    pP = np.mean([_predict(ramp='power', **dp) for dp in draws_P], axis=0)\n\n    dp = pP - pR\n    pbar = 0.5 * (pP + pR)\n    pq = np.maximum(pbar * (1.0 - pbar), 1e-3)\n    cvec = dp / pq  # precision-weighted contrast (linearised LLR)\n    denom = float(np.abs(cvec).sum())\n    if (not np.isfinite(denom)) or denom < 1e-12:\n        return 0.0\n    sc = s - (2.0 * pbar - 1.0)  # centre by the predicted null\n    return float(np.dot(cvec, sc) / denom)",
  "rationale": "STRUCTURE OF THE METRIC. The two theories share every module except the trust-ramp shape on the trusted (99%) channel and the noise calibration, so the only usable signal is the difference in per-trial choice probabilities that the two boxes imply. The metric is therefore an explicit two-model discriminant: for each trial it recomputes both theories' predicted P(choose A) (implementing both predict functions verbatim: distrust inversion with partial centering, geometric vs power-law trust ramp, floored convex primacy gate, multiplicative combination, logistic-plus-lapse), averages each theory's prediction over an ensemble of 35 parameter settings spanning its full box (centre + factorial corners over tau x t0 x (beta,epsilon) x (eta,delta) + kappa/t1 tails), and scores the observed choice with the precision-weighted contrast c_t = (P_PBDT - P_RTDT)/(pbar(1-pbar)). This is exactly the linearised log-likelihood ratio between the two predicted Bernoulli products, i.e. the statistic that maximises the between-theory mean gap relative to the per-subject (binomial) variance that the Welch test divides by. WHY THE WEIGHTS LAND WHERE THE THEORIES DISAGREE. Because the kernel cues share one validity, gamma/c drop out and the predictions differ only through (i) the trusted-channel weights - PBDT's position-1 trusted weight (raw 0.008-0.020) is 3-10x RTDT's (raw 0.0006-0.0064), so on the R12 rescue cell PBDT flips to the helper side (P ~ 0.40-0.50) while RTDT stays with the kernel winner (P ~ 0.49-0.52), and similarly opposite-signed d on D16/D26 duels, R1/S1/X16 swing and seesaw cells; and (ii) the noise boxes (beta 2.5-4.0, eps 0.12-0.20 vs beta 3.6-4.4, eps 0.11-0.14), which separate the theories on the high-signal screens (TOT, DINV) where both pick the same side but with different confidence. Theory-agreeing trials (R0, R6, S6, VETO) receive near-zero weight automatically, so they add no between-subject noise. PROPERTIES. The score is a per-trial weighted sum of signed choices, centred by the predicted null and normalised: it is well defined on the pooled DataFrame and on each single subject's 96-trial slice (between_subject_variance is available), it is mirror-symmetric by construction (any A/B side bias cancels because the weights are antisymmetric under mirroring), and it is robust to wherever in the box the simulator draws parameters because the weights are box-averages and the sign pattern of the contrast is consistent across both boxes. Expected values: RTDT-simulated data gives a negative score (early-trusted neglect, late-trusted amplification, noisier screens), PBDT-simulated data a positive score (early-trusted following, sharper screens), with the dominant contributions from R12 (largest single-cell sign flip, dp up to 0.11 at the disjoint corners), D16/D12, R1/S1/X16, and TOT/DINV. No single cell is decisive at N=25; the precision-weighted pooling over all 96 trials is what makes the contrast detectable, and this pooling is information-theoretically the best linear use of the data."
}
```

## Usage

```json
{
  "input_tokens": 11024,
  "output_tokens": 108301,
  "reasoning_tokens": 105512,
  "provider": "Sail Research"
}
```
