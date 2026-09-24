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
**Validities (n_features=7):** [0.99, 0.99, 0.72, 0.99, 0.72, 0.72, 0.99]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  trial 17: A=[0, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 19: A=[0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 20: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 0]
  trial 21: A=[0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1]
  trial 23: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 25: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 26: A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 27: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 28: A=[0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 29: A=[0, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 1]
  trial 30: A=[1, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PBDT and RTDT share the distrust-inversion kernel, the floored convex primacy gate, and the multiplicative bracket w_j ∝ g_j·[t_j·(1-τ)/n + τ·inv_j]; they differ in exactly one quantitatively decisive respect: the SHAPE OF THE TRUST RAMP on the flat-share (trusted) channel. PBDT's ramp is a capped power law t_j = t0 + (t1-t0)·(j/(n-1))^κ with t0 ∈ [0.25, 0.35] and t1 ≤ 1.0 — an early trusted expert keeps 25-35% of the flat share exactly where the gate is maximal, and no trusted expert ever exceeds the flat share. RTDT's ramp is geometric t_j = t0·(t1/t0)^((j/(n-1))^κ) with t0 ∈ [0.03, 0.12] and t1 ∈ [1.0, 1.6] — the early trusted expert is annihilated (3-12% of flat share) while the LAST trusted expert is amplified up to 1.6× ABOVE the flat share. Verified at the joint corners of both boxes (ramp × gate corners, with β/ε at worst-case ends): the raw position-1 trusted weight is DISJOINT — PBDT [0.0070, 0.0113] vs RTDT [0.0007, 0.0036]; the position-2 trusted weight (gate exactly 1.0, hence GATE-FREE, a pure t(1/6) assay) is DISJOINT — PBDT [0.0034, 0.0059] vs RTDT [0.0006, 0.0033]; the late trusted weight can exceed PBDT's maximum by 1.6× under RTDT only (RTDT [0.0014, 0.0046] vs PBDT [0.0016, 0.0029]). Prior design [18] probed this with n=6 and anchors at positions 1, 2, 6 using a position-4/5 kernel gate gap. This design is structurally distinct: n=7, a FOUR-POINT anchor set including a MID anchor (positions 1, 2, 4, 7 = ramp fractions 0, 1/6, 1/2, 1), a different counterweight (the positions-5/6 tail gate gap), and a new duel family exploiting the amplified position-1 gate against the exact-unity position-2 gate.

LAYOUT (n=7): v = [0.99, 0.99, 0.72, 0.99, 0.72, 0.72, 0.99]. Trusted 0.99 experts sit at ramp fractions 0, 1/6, 1/2, 1 (positions 1, 2, 4, 7); the three 0.72 experts occupy positions 3, 5, 6. Because all kernel cues share one validity, w_inv = 1/3 exactly for each and the 0.99 leakage is ≤ 10^-5 of kernel mass — γ and c DROP OUT of every prediction, isolating the ramp/gate architecture (the only place the theories' boxes differ). The kernel channel supplies (i) a finely calibrated counterweight γnet = weight(0.72@5) − weight(0.72@6) = 0.307·Δg + flat·Δ(gt), verified POSITIVE at every corner of both boxes: PBDT [+0.0007, +0.0024], RTDT [+0.0004, +0.0049] — a theory-agreeing base of exactly the scale of the trusted weights, so trusted endorsements can single-handedly flip it or fail to depending on the ramp; and (ii) the normalizer Z ≈ [0.20, 0.33].

AXIS 1 — EARLY-TRUSTED RESCUE (R1, R2, R12 + mirrors; the headline). Base rung B0: A wins 0.72@5 vs B wins 0.72@6 — both theories mildly prefer A (P ∈ [0.50, 0.52]). R1 adds the 99%@1 endorsement to B's side: PBDT's position-1 trusted weight (≥ 0.0070, disjoint from RTDT's ≤ 0.0036) overcomes γnet at EVERY corner → mirror-corrected P(A) ∈ [0.45, 0.49] (a flip toward B); RTDT straddles: P(A) ∈ [0.49, 0.51]. DISJOINT BOXES. R12 (both early trusted experts on B's side): PBDT [0.44, 0.48] vs RTDT [0.49, 0.51] — disjoint. R2 (the gate-free position-2 trusted expert): PBDT [0.46, 0.49] vs RTDT [0.49, 0.52] — near-disjoint. Observed mirror-corrected P(A) ≤ 0.47 pooled on R1/R12 is impossible for RTDT at any admissible parameter combination; P(A) ≥ 0.50 is impossible for PBDT. These are 1-vs-2 and 1-vs-3 count trials on which both theories pick the single-kernel-cue product when the trusted helper is absent — the flip is carried entirely by the trusted channel.

AXIS 2 — SWING-PROFILE RECONSTRUCTION (S1, S2, S4, S7 vs the R-cells + mirrors). The within-subject double difference P(S_p) − P(R_p) ≈ (1-ε)·β·ŵ(99@p) isolates the behavioral trusted weight at each anchor, with the base counterweight and Z cancelling exactly. Verified swing magnitudes: swing@1: PBDT [0.042, 0.077] vs RTDT [0.003, 0.022] — DISJOINT with a 2× gap; swing@2: PBDT [0.019, 0.040] vs RTDT [0.002, 0.019] — near-disjoint; swing@4 and swing@7 are overlapping fitting cells. The resulting four-point profile {ŵ1, ŵ2, ŵ4, ŵ7} is the quantitative core: PBDT requires ŵ1/ŵ2 = g_1·t0/t(1/6) ∈ [1.28, 2.76] (early trusted weight EXCEEDS the position-2 weight at every corner), while RTDT allows ŵ1/ŵ2 down to 0.75 (its geometric ramp rises faster than the gate falls over positions 1→2); and the fitted late/early ratio ŵ7/ŵ1 must land ≤ 0.41 under PBDT but can reach 6.7 under RTDT (observed ≥ 0.45 falsifies PBDT's entire space; ≤ 0.30 falsifies RTDT's). This is the first design to measure the ramp at a MID anchor (fraction 1/2), where PBDT's t ∈ [0.50, 0.72] and RTDT's t ∈ [0.17, 0.66] overlap in level but contribute opposite pulls in the joint profile fit.

AXIS 3 — TRUSTED-POSITION DUELS (D12, D14, D17 + mirrors; kernel cues all tied). D12 (99@1 vs 99@2): the amplified position-1 gate ((7/6)^η ≈ 2.2-3.1) times PBDT's t0-floor gives PBDT d > 0 at every corner → P(position-1 product) ∈ [0.508, 0.523]; RTDT's crushed t0 lets the position-2 expert match or beat it → [0.500, 0.505]. DISJOINT. D14 and D17 (position-1 vs mid/late trusted): PBDT [0.52, 0.53] vs RTDT [0.494, 0.506] — disjoint. A reliable preference ≥ 0.52 for the earlier trusted expert's product across the duel family is impossible for RTDT; chance performance falsifies PBDT. These cells also exclude naive tallying on DINV-family logic and are anti-TTB when combined with the screens.

AXIS 4 — SCREENS AND ANCHORS. DINV (the three 0.72 experts' product vs the four 0.99 experts' product, 3-vs-4 count): BOTH theories confidently prefer the low-validity product (P ∈ [0.80, 0.92]) — verifies subjects read and distrust-invert the communicated validities; excludes normative weighting, TTB (top expert 0.99@1 on the losing side), and tallying (4 cues beat 3). K35 (0.72@3 vs 0.72@5): theory-agreeing gate assay, P ≈ [0.68, 0.80], verifying position/gate sensitivity with a detectable effect. B0 is the theory-agreeing base rung. TOT is the comprehension anchor (both ≥ 0.93).

HONESTY ABOUT EFFECT SIZES. Both boxes pin τ ≥ 0.91, capping the flat share at ~1.6% per cue, so trusted-channel single-cell proportions deviate from 0.5 by only 0.5-6%. Statistical power comes from the within-subject structure: every cell type appears in A/B-mirrored form with K = 3 repetitions (6 trials per cell per subject), and the decisive readouts are family-level — the pooled early-rescue sign test (R1+R2+R12), the four swing double differences, and the fitted profile ratios ŵ1/ŵ2 and ŵ7/ŵ1 — all of which are β/ε-robust (they cancel in the linear regime that holds here, |d| ≤ 0.05).

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PBDT reference evidence (w_j ∝ g_j·[t_j·(1-τ)/7 + τ·inv_j] at t0=0.30, t1=0.975, κ=1.15, η=7.2, δ=0.20, τ=0.92) versus RTDT reference evidence (t0=0.06, t1=1.3, κ=0.8, η=6.5, δ=0.18, τ=0.93) loads with opposite profiles: the two evidence axes differ almost exclusively on the early-trusted entries (positions 1-2), where PBDT's weights are 2-10× RTDT's, and on the late entry, where RTDT's super-unity t1 can exceed PBDT's capped t1 by 1.6×. Single-number readouts: (i) R1 sign (PBDT ≤ 0.49; RTDT ≥ 0.49); (ii) R12 (PBDT [0.44, 0.48] vs RTDT [0.49, 0.51]); (iii) D12 (PBDT ≥ 0.508 vs RTDT ≤ 0.505); (iv) swing@1 (disjoint: ≥ 0.042 vs ≤ 0.022); (v) fitted ŵ1/ŵ2 (≥ 1.28 vs can be < 1); (vi) fitted ŵ7/ŵ1 (≤ 0.41 vs can exceed 1). With 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    def to_vec(x):\n        if isinstance(x, np.ndarray):\n            return np.asarray(x, dtype=float)\n        if isinstance(x, (list, tuple)):\n            return np.asarray(x, dtype=float)\n        s = str(x).strip().lstrip('[').rstrip(']')\n        return np.asarray([float(q) for q in s.split(',') if q.strip() != ''], dtype=float)\n\n    if data is None or len(data) == 0:\n        return 0.0\n\n    first = to_vec(data['option_a_ratings'].iloc[0])\n    n = int(first.shape[0])\n    if n < 2:\n        return 0.0\n\n    V = np.array([0.99, 0.99, 0.72, 0.99, 0.72, 0.72, 0.99], dtype=float)\n    if V.shape[0] != n:\n        V = np.linspace(0.95, 0.55, n)\n    pos = np.arange(n, dtype=float)\n    span = float(n - 1)\n\n    # ---------- PBDT reference weights (box-centre parameters) ----------\n    # tau=0.92, eta=7.2, delta=0.20, gamma=2.75, c=0.15,\n    # t0=0.30, t1=0.975, kappa=1.15, beta=4.0, epsilon=0.125\n    err = 1.0 - V\n    cen = np.maximum(err - 0.15 * float(err.min()), 0.0)\n    invP = np.power(cen, 2.75)\n    w_invP = invP / invP.sum()\n    tP = 0.30 + (0.975 - 0.30) * np.power(pos / span, 1.15)\n    gP = 0.20 + 0.80 * np.power((n - pos) / span, 7.2)\n    brP = tP * (1.0 - 0.92) / float(n) + 0.92 * w_invP\n    rawP = gP * brP\n    wP = rawP / rawP.sum()\n\n    # ---------- RTDT reference weights (box-centre parameters) ----------\n    # tau=0.93, eta=6.5, delta=0.18, gamma=3.0, c=0.15,\n    # t0=0.06, t1=1.30, kappa=0.8, beta=3.25, epsilon=0.16\n    invR = np.power(cen, 3.0)\n    w_invR = invR / invR.sum()\n    fracR = np.power(pos / span, 0.8)\n    tR = np.exp(np.log(0.06) + fracR * (np.log(1.30) - np.log(0.06)))\n    gR = 0.18 + 0.82 * np.power((n - pos) / span, 6.5)\n    brR = tR * (1.0 - 0.93) / float(n) + 0.93 * w_invR\n    rawR = gR * brR\n    wR = rawR / rawR.sum()\n\n    def p_a(a, b, w, beta, eps):\n        d = float(np.dot(w, np.sign(a - b)))\n        return (1.0 - eps) * (0.5 + 0.5 * np.tanh(0.5 * beta * d)) + 0.5 * eps\n\n    a_col = list(data['option_a_ratings'].values)\n    b_col = list(data['option_b_ratings'].values)\n    r_col = np.asarray(data['response'].values, dtype=float)\n\n    total = 0.0\n    count = 0\n    for i in range(len(a_col)):\n        a = to_vec(a_col[i])\n        b = to_vec(b_col[i])\n        if a.shape[0] != n:\n            a = np.resize(a, n)\n        if b.shape[0] != n:\n            b = np.resize(b, n)\n        pP = p_a(a, b, wP, 4.0, 0.125)\n        pR = p_a(a, b, wR, 3.25, 0.16)\n        pbar = 0.5 * (pP + pR)\n        v = 4.0 * pbar * (1.0 - pbar)\n        wgt = (pP - pR) / max(v, 0.02)\n        y = 1.0 - 2.0 * float(r_col[i])  # +1 if chose A, -1 if chose B\n        total += wgt * y\n        count += 1\n\n    if count == 0:\n        return 0.0\n    return float(total / float(count))",
  "rationale": "MATCHED-FILTER (VARANCE-WEIGHTED) THEORY-ALIGNMENT SCORE. The two theories share the distrust kernel, the gate family, and the additive-logistic machinery; they diverge in (a) the trust-ramp shape, whose only behavioural fingerprint is the early-trusted rescue/duel cells (R1, R2, R12, S1, S2, D12, D14, D17: PBDT's floored power-law ramp gives positions 1-2 trusted weights 2-10x RTDT's geometric ramp, and RTDT's super-unity t1 can reverse D17), (b) the K35 gate assay (RTDT's smaller normalizer concentrates more weight on position 3), and (c) the sharpness anchors TOT and DINV, where PBDT's tighter noise box (beta 3.6-4.4, eps 0.11-0.14) vs RTDT's softer box (beta 2.5-4.0, eps 0.12-0.20) produces the single largest per-cell probability gaps (~3% at box centres). Because the per-cell gaps are small (0.5-3%) and each subject sees each unique pair only 3 times, no single cell can survive N=25; the only viable statistic is one that pools ALL 96 trials with statistically optimal weights. The metric therefore implements both theories' predict functions at their box-centre reference parameters (PBDT: t0=0.30, t1=0.975, kappa=1.15, eta=7.2, delta=0.20, tau=0.92, gamma=2.75, c=0.15, beta=4.0, eps=0.125; RTDT: t0=0.06, t1=1.30, kappa=0.8, eta=6.5, delta=0.18, tau=0.93, gamma=3.0, c=0.15, beta=3.25, eps=0.16), computes per trial the reference choice probabilities pP and pR for choosing A, and scores the observed choice y in {-1,+1} with weight w = (pP - pR) / (4*pbar*(1-pbar)). This is exactly the locally optimal discriminant (likelihood-ratio direction, inverse-Bernoulli-variance weighted): trials where the theories disagree strongly (TOT, DINV, R12, S1, R1, D17, K35) get large weights, trials where they agree get near-zero weight, and low-variance high-consistency anchor trials are upweighted relative to 50/50 cells, which is what maximises the between-theory mean gap per unit of per-subject (binomial) variance. The metric returns the mean of w*y over all rows: it is positive-shifted under PBDT-generated data (choices align with the PBDT reference direction: stronger early-trusted rescue on R1/R12, position-1 preference on the duels, sharper anchor consistency) and negative-shifted under RTDT-generated data (choices align with the crushed early-trust ramp, late-trusted preference on D17, softer anchor consistency). The same computation runs unchanged on the pooled DataFrame and on each subject's 96-trial slice, so the between-subject variance is the pure per-subject sampling variance of the composite; by construction (matched filter) this is the smallest achievable variance for a statistic with this mean gap, which is the best possible position for the Welch t-test at N=25. The stimulus-derived weights are identical for every subject (same 96 pairs), so the per-subject values differ only through behaviour, and the mirror-paired design makes the weight sum sign-symmetric so pure side bias cancels in expectation."
}
```

## Usage

```json
{
  "input_tokens": 10857,
  "output_tokens": 88545,
  "reasoning_tokens": 86561,
  "provider": "Sail Research"
}
```
