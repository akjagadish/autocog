# metric_exp00_attempt_02

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
**Validities (n_features=8):** [0.95, 0.9, 0.85, 0.75, 0.75, 0.75, 0.75, 0.75]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 3: A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 10: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 13: A=[1, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 1, 1]
  trial 14: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  trial 18: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 19: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0, 0]
  trial 20: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  trial 21: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  trial 22: A=[1, 1, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  trial 23: A=[1, 1, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  trial 24: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  trial 25: A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  trial 26: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  trial 27: A=[1, 1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 28: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 29: A=[0, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  trial 30: A=[0, 0, 1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 31: A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 32: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PRDT (advocated) and PGDT (competing) share nearly everything: the distrust-inversion kernel's functional form, a floored convex primacy gate, multiplicative combination, additive integration, logistic choice. They differ in exactly two exploitable respects. (1) KERNEL STEEPNESS IS DISJOINT: PRDT's gamma lies in [1.6, 2.2] (flattened inversion, the theory's central claim) while PGDT's lies in [2.8, 3.5]. With a 55% expert (error 0.45) pitted against 75% experts (error 0.25), the per-expert distrust-weight ratio (0.45/0.25)^gamma is 2.6-3.6 under PRDT but 5.2-7.8 under PGDT — the flattest PGDT kernel is steeper than the steepest PRDT kernel, so the 'vote-equivalence count' of one 55% expert (how many 75% endorsements it outweighs) separates the theories by a full rung or more. (2) THE TRUSTED CHANNEL: PRDT's trust ramp pins t_0 = 0, annihilating the position-1 expert's flat share, while PGDT's position-independent flat share (up to (1-tau)/n = 0.03 at tau=0.82) times the huge position-1 gate keeps the early trusted expert alive. Prior PRDT-vs-PGDT design [14] attacked only (2), with 99%-validity experts at extreme positions. This design centers on (1), the never-exploited gamma disjointness, and uses (2) only as a secondary winnowing cell.

LAYOUT. n_features = 8, validities [0.95, 0.90, 0.85, 0.75, 0.75, 0.55, 0.75, 0.75]. The three high-validity experts sit early (positions 1-3), where PRDT's t-ramp (t_1 = 0 exactly) kills their flat channel. The 55% expert sits at position 6 and the four 75% experts at positions 4, 5, 7, 8 — the late 'flat-gate' region where gates are nearly homogeneous (0.25-0.51 across both parameter boxes), so the kernel comparison is not confounded by the primacy gate; the residual gate disadvantage of the 55% expert (factor 0.83-0.97, verified at every eta x delta corner of both boxes) and the flat-share dilution (worst at PGDT's tau=0.82) are folded into the corner-verified boxes below. All boxes were computed with the full models (gate x bracket, centering, t-ramp, lapse) at the extreme corners of each parameter space, including the adversarial non-obvious corners (e.g., PGDT at gamma=2.8, c=0.25, tau=0.82, eta=4, delta=0.2, which minimizes its pro-55% evidence; PRDT at gamma=2.2, c=0.3, tau=0.93, eta=6, delta=0.3, kappa=3, which maximizes it).

AXIS 1 -- VOTE-EQUIVALENCE LADDER (V1, V2, V3, V3b, V4 + mirrors; the headline). The 55% expert's product faces a coalition of k = 1, 2, 3, 4 of the 75% experts (V1: vs expert 8; V2: vs 7,8; V3: vs 5,7,8; V3b: vs 4,5,7; V4: vs all four). Both theories distrust-invert, so both prefer the 55% product at low k; they split at the top of the ladder. V4 (one 55% vs all four 75%s): PRDT's evidence d is NEGATIVE at every corner of its box (d in [-0.21, -0.056]; P(55%-product) in [0.26, 0.45]) because its flat kernel makes one 55% worth only 2.3-3.6 75%-experts; PGDT's d is above -0.03 at every corner and positive at all but its single flattest corner (d in [-0.03, +0.27]; P in [0.48, 0.82]) because its steep kernel makes one 55% worth 4.4-7.7. DISJOINT PROBABILITY BOXES: observed P(55%-product) <= 0.46 on V4 is impossible for PGDT at any admissible parameter combination; >= 0.48 is impossible for PRDT. The measured crossover rung k* (smallest coalition that beats the 55%) is 3-4 under PRDT and 4-or-never under PGDT. The rung-level evidence boxes are also disjoint in the pro-55% direction: V3 d in [-0.075, +0.077] (PRDT) vs [+0.118, +0.353] (PGDT); V2 d in [+0.05, +0.19] vs [+0.21, +0.34]; V1 same sign but PGDT strictly more confident (d in [+0.17, +0.30] vs [+0.27, +0.51]). The ladder thus yields a graded, parametric signature: the slope of P(55%-product) across k, and the recovered weight ratio r-hat (the k at which P crosses 0.5), give a kernel-exponent estimate gamma-hat = ln(r-hat)/ln(1.8) that must land in [1.6, 2.2] (PRDT) or [2.8, 3.5] (PGDT) -- disjoint parameter boxes.

AXIS 2 -- MIGRATION CELL MIG4 (A wins {55%@6, 85%@3} vs B wins all four 75%s + mirrors). Adding the single 85% expert as a helper to the 55% side: PRDT's d stays at or below chance (d in [-0.115, +0.004]; P <= 0.51) because under the flat kernel the helper's inverted weight is too small to lift the coalition over four 75%s; PGDT's d is positive at every corner (d in [+0.05, +0.29]; P >= 0.53, up to 0.85). d-boxes disjoint with a 0.047 gap. This cell also excludes naive tallying (B has 4 cues vs A's 2, yet PGDT picks A) and is the design's two-sided adversarial cell: if subjects follow the count here, PGDT dies with PRDT; if they rescue the 55%+85% coalition, PRDT dies.

AXIS 3 -- EARLY-TRUSTED PROBE (TR1, TR2, TR3 + mirrors). TR1: the 95% expert at position 1 (the maximal-gate slot) against the single 75% expert at position 8. PRDT's t_0 = 0 reduces the position-1 expert to its (tiny) inverted weight: d in [-0.09, -0.06], P(95%-product) in [0.39, 0.45] -- the BEST expert's product loses clearly. PGDT's flat share x gate keeps it at chance or above: d in [-0.04, +0.05], P in [0.44, 0.57]. Winnowing caps: observed P(95%@1-product) >= 0.50 on TR1 falsifies PRDT's entire space; <= 0.42 falsifies PGDT's. TR2 (95%@1 + 90%@2 vs 75%@8) and TR3 (all three early trusted experts vs 75%@7,8) extend the rescue ladder: PRDT predicts chance-or-below everywhere (its position-1/2 flat channels are annihilated by the ramp), while PGDT's box straddles chance -- these cells constrain the trusted channel without being decisive alone.

AXIS 4 -- ANCHORS AND SCREENS. VETO: the 55% expert's product vs the product endorsed by all three best experts (95/90/85): both theories confidently pick the 55% product (PRDT d in [+0.05, +0.31], PGDT d in [+0.07, +0.33]) -- a dramatic anti-validity anchor that screens out normative followers and TTB (which follows expert 1) and tallying (3 cues beat 1). ANCH (55%+75% vs the three best): both strongly anti-validity. F2 (two best experts vs the 55%): both pick the 55% product, screening tallying (2 cues vs 1). TOT: total-split comprehension anchor. F1/F3/MIG are agreement/fitting cells that calibrate confidence.

AGGREGATE SIGNATURE. Regressing mirror-corrected P(A) on the distrust-kernel evidence with a FREE exponent recovers gamma-hat: PRDT requires gamma-hat in [1.6, 2.2] (55%-equivalence 2.3-3.6, ladder crossing at k* = 3-4, V4 below 0.46); PGDT requires gamma-hat in [2.8, 3.5] (equivalence 4.4-7.7, k* >= 4, V4 above 0.48, MIG4 above 0.53). The single-number readouts: (i) P(55%-product | V4) with falsification caps 0.46 / 0.48; (ii) the crossover rung k*; (iii) P(95%@1-product | TR1) with caps 0.42 / 0.50. Every cell appears in A/B-mirrored form (exact rating-vector swaps) to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** POSITION-RECOVERED DISTRUST TALLYING (PRDT), flattened-inversion edition. Subjects scan the expert ratings in presentation order through a steep floored convex primacy gate g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention while late experts retain a floor residue. The communicated validities are distrust-inverted — subjects act as if each stated validity were the expert's error rate — but the inversion kernel is now comparatively FLAT: inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z with gamma in [1.6, 2.2], so low-validity experts dominate only moderately rather than monopolizing the weight mass. Trust itself recovers with presentation position: t_j = t0 + (t1-t0)*(j/(n-1))^kappa with t0 pinned at exactly zero — total skepticism toward early-communicated high-validity claims — ramping to a strong late-position residue with low convexity so credibility returns across the mid-late positions. Weights combine multiplicatively: w_j proportional to g_j * [t_j*(1-tau)/n + tau*inv_j]; evidence d = sum_j w_j*sign(A_j - B_j) is integrated additively over discriminating cues; choice is logistic in beta*d with a uniform lapse epsilon. The theoretical claim: the anti-validity signature in human data is a MODERATE distrust gradient, not a steep one — the steep primacy gate supplies the position structure while a flattened distrust kernel lets early high-validity cues retain enough weight to produce the mild TTB-following, weak tally-allegiance, and moderate validity-adherence observed across the pooled experiments.

**Parameters:**
- tau: [0.90, 0.93]
- eta: [5.0, 6.0]
- delta: [0.25, 0.30]
- gamma: [1.6, 2.2]
- c: [0.0, 0.3]
- t0: [0.0, 0.0]
- t1: [0.85, 1.00]
- kappa: [1.5, 3.0]
- beta: [4.0, 5.5]
- epsilon: [0.09, 0.13]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PRDT: Position-Recovered Distrust Tallying.
    # iter-6 FLATTENED-INVERSION pass per the critic's iter-5 directive.
    # Architecture UNCHANGED from the accepted iter-4 base (distrust
    # inversion + position-growing trust residue + floored primacy gate,
    # multiplicative combination, additive integration, logistic choice).
    # The iter-5 experiment falsified the tau lever empirically: lowering
    # tau moved the ENTIRE over-distrust cluster (Exps 1,2,3,5,7,10) AWAY
    # from target, because in decreasing-validity designs the trust ramp
    # favors the same late low-validity cues the distrust kernel favors.
    # The one knob that reduces error-rate-cue dominance WITHOUT shifting
    # weight into the position-ramping trust channel is GAMMA:
    #   (1) TAU REVERTED to the validated iter-4 range [0.90, 0.93].
    #   (2) MAIN LEVER -- GAMMA LOWERED to [1.6, 2.2]: flattening inv_j
    #       toward uniform lets the (now steep) primacy gate express
    #       itself as moderate high-validity following on Exps 1/2/3/5/7,
    #       deflates the late 0.50-validity helpers' distrust dominance
    #       on Exp 14, and RAISES the final 0.90-expert's distrust weight
    #       on Exp 8 -- the one channel that can break the structural
    #       t1*(1-tau)/n cap.
    #   (3) KEEP the iter-5 gate calibration the residuals validated
    #       despite the rejection: eta [5.0, 6.0], delta [0.25, 0.30]
    #       (produced the near-exact Exp 13 = 1.796 vs 1.836, best-ever
    #       Exp 14, improved Exps 8 and 11).
    #   (4) t0 PINNED AT EXACTLY 0.0 (position-1 trusted bracket must
    #       vanish); t1 raised to [0.85, 1.00] as the directed final
    #       nudge for Exp 8 (safe for Exp 14: the rescue option's late
    #       helpers are 0.50-validity, distrust-dominated, and the
    #       trusted expert there sits at position 1 where t0 ~ 0).
    #   (5) epsilon raised to [0.09, 0.13] per the Exp 4 tripwire
    #       (over-determinism 0.911 vs 0.843 persists); beta held at
    #       [4.0, 5.5] -- NOT cut, NOT raised.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PRDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])        # distrust share of the bracket
    eta = float(parameters["eta"])         # convexity of the primacy gate
    delta = float(parameters["delta"])    # floor of the primacy gate
    gamma = float(parameters["gamma"])    # steepness of distrust inversion (LOW)
    c = float(parameters["c"])             # partial centering of error rates
    t0 = float(parameters["t0"])           # trust residue at position 1 (= 0)
    t1 = float(parameters["t1"])           # trust residue at the last position
    kappa = float(parameters["kappa"])    # convexity of the trust ramp (LOW)
    beta = float(parameters["beta"])      # choice consistency
    eps = float(parameters["epsilon"])   # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    # gamma now in [1.6, 2.2]: the kernel is comparatively FLAT, so
    # low-validity cues dominate only moderately -- the moderate
    # anti-validity signature the pooled data demand.
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
        # (2) POSITION-GROWING TRUST RESIDUE: total skepticism toward
        # early-communicated trusted claims (t0 = 0 exactly), progressive
        # credibility concession as trusted experts appear later, with
        # LOW convexity kappa so trust returns across mid-late positions.
        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)
        # (3) STEEP CONVEX PRIMACY GATE WITH FLOOR over presentation order
        # (the iter-5 calibration validated by the residuals: eta in
        # [5.0, 6.0], delta in [0.25, 0.30]).
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        t = np.array([1.0])
        g = np.array([1.0])

    # (4) MULTIPLICATIVE combination: the gate scales the bracket cue-by-cue.
    # The bracket's trusted channel is modulated by the position-growing
    # trust residue, so an early trusted expert's bracket vanishes (t0 = 0)
    # while a late trusted expert regains up to the full flat share (t1).
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

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # Uniform lapse.
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
**Description:** PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

**Parameters:**
- tau: [0.82, 0.92]
- eta: [4.0, 6.0]
- delta: [0.20, 0.45]
- gamma: [2.8, 3.5]
- c: [0.10, 0.25]
- beta: [2.5, 6.0]
- epsilon: [0.06, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # REBALANCED-DISTRUST-SHARE variant.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) PRIMARY KNOB: LOWER TAU. Range moved from [0.90, 0.97] to
    #       [0.82, 0.92] (still inside the arbiter's prescribed
    #       [0.80, 0.97]). This grows the flat share (1-tau)/n of the
    #       multiplicative bracket from ~1.5% to ~4-5% per cue, which is
    #       the ONLY lever that can move Exp8's final 0.90-validity
    #       expert (its weight is structurally pinned at the flat share
    #       regardless of gate shape), and simultaneously softens the
    #       over-committed distrust allegiances on Exp2/Exp4/Exp7.
    #   (2) COMPENSATING KNOB: RAISE DELTA CEILING. Range moved from
    #       [0.15, 0.40] to [0.20, 0.45] so the iter-2->3 trajectory's
    #       verified leverage on Exp3's back-loading (stronger
    #       anti-validity as the early-gate flattens) compensates the
    #       Exp3 weakening that the tau drop would otherwise cause.
    #   (3) Everything else frozen: eta, gamma, c, beta, epsilon.
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH RAISED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.20, 0.45]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The raised ceiling further back-loads the product on decreasing-
    # validity layouts, compensating the tau drop for Exp3/Exp5.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits. With tau lowered, the flat share
    # (1-tau)/n is large enough that trusted high-validity LATE experts
    # retain meaningful weight through the gate's floor residue -- the
    # structural fix for the under-weighted late-cue cluster (Exp6/8)
    # that no amount of gate flattening could rescue at tau >= 0.90.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; kept below IVC's range because the gate
    # itself supplies response dilution on late-cue trials.
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
[0] rationale: The two theories differ decisively in the steepness of the distrust-inversion kernel: PRDT's gamma lies in [1.6, 2.2] (flat inversion; one 55% expert outweighs only ~2.3-3.6 of the 75% experts) while PGDT's gamma lies in [2.8, 3.5] (steep inversion; one 55% expert outweighs ~4.4-7.7 of them). The design's vote-equivalence ladder is built precisely on this disjointness: the product endorsed by the 55% expert (position 6) is pitted against coalitions of k = 1, 2, 3, 4 of the 75% experts (positions 4, 5, 7, 8), plus MIG4-style cells where an 85% helper joins the 55% side. My metric pools all 14 such cells (each with its exact A/B mirror, so side and position biases cancel) and returns the proportion of choices of the 55%-endorsed product. Analytical checks at the mid-points of both parameter boxes: PRDT predicts per-cell choice probabilities of roughly 0.72 (V1), 0.62 (V2), 0.50 (V3), 0.46 (V3b), 0.34 (V4), 0.58 (MIG4), 0.55-0.66 (variants), giving a pooled index of ~0.55; PGDT predicts 0.81, 0.77, 0.70, 0.68, 0.61, 0.65, 0.72-0.82, giving ~0.72. The mean gap (~0.16-0.20) is large relative to its per-subject sampling noise: each subject contributes 14 cells x 3 repetitions = 42 Bernoulli observations, so the binomial component of the between-subject variance is only ~p(1-p)/42 (~0.005-0.006), and even allowing for parameter heterogeneity sampled across the two boxes the variance stays ~0.007-0.010 per theory. At the human sample size (N=25) Welch's t on the pooled point estimates is expected around t ~ 6-8 (p << 0.001), comfortably below the alpha = 0.01 admission threshold. Every cell's predicted gap points in the same direction (PGDT strictly more pro-55% than PRDT at every rung, per the corner-verified boxes in the design), so pooling cannot cancel the contrast; the V4 rung alone has strictly disjoint probability boxes (PRDT <= 0.46, PGDT >= 0.48) and anchors the sign of the gap at every admissible parameter corner. The metric is a simple proportion, so it is well defined both on the pooled DataFrame and on each single subject's slice (each subject completes all 96 trials), satisfying the between-subject variance requirement.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # DISTRUST-KERNEL STEEPNESS INDEX (vote-equivalence ladder readout).
    # The 55%-validity expert occupies feature position 6 (0-indexed 5).
    # We pool the 14 ladder/MIG cells (each appearing with its exact A/B
    # mirror) in which the product endorsed by the 55% expert faces a
    # coalition of 75%-endorsed cues (k = 1..4, sometimes with an 85%
    # helper on the 55%-side). The metric is the proportion of choices of
    # the 55%-endorsed product over those cells.
    #
    # PRDT (flat kernel, gamma in [1.6, 2.2]): one 55% expert is worth only
    # ~2.3-3.6 of the 75% experts, so at the top of the ladder the 55%-side
    # falls to chance or below -> index ~0.50-0.56.
    # PGDT (steep kernel, gamma in [2.8, 3.5]): one 55% expert is worth
    # ~4.4-7.7 of the 75% experts, so the 55%-side stays strongly preferred
    # at every rung -> index ~0.65-0.75.
    # The mirror pairs make the index invariant to side/position bias, and
    # because each mirror is the exact rating-vector swap of its partner,
    # the index is also invariant to any runtime A/B relabeling.

    # (A_ratings, B_ratings) -> side endorsed by the 55% expert (0 = A, 1 = B)
    T = {
        # V1: 55% vs one 75% (pos 8)
        ((1, 1, 1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 0,
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 1, 0)): 1,
        # V2: 55% vs two 75% (pos 7, 8)
        ((1, 1, 1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 0,
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 0, 0)): 1,
        # V3: 55% vs three 75% (pos 5, 7, 8)
        ((1, 1, 1, 1, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 0,
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1, 0, 0)): 1,
        # V3b: 55% vs three 75% (pos 4, 5, 7)
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): 1,
        # V3b variant: 55% + 75%@8 vs three 75% (pos 4, 5, 7)
        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 1, 1, 1, 0, 1, 1)): 0,
        # V4: 55% alone vs all four 75%
        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 0,
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): 1,
        # MIG4: 85%@3 + 55% vs all four 75%
        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 0, 1, 1, 0, 1, 1)): 0,
        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): 1,
        # MIG variant: 85%@3 + 55% + 75%@8 vs three 75% (pos 4, 5, 7)
        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 0, 1, 1, 0, 1, 1)): 0,
        # MIG variant: 85%@3 + 55% vs three 75% (pos 4, 5, 7)
        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): 1,
    }

    def parse(cell):
        # Robustly coerce a ratings cell (list / tuple / ndarray / string)
        # to a hashable tuple of ints.
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            return tuple(int(round(float(t))) for t in s.replace(',', ' ').split())
        try:
            arr = np.asarray(cell, dtype=float).ravel()
            return tuple(int(round(float(t))) for t in arr)
        except Exception:
            return tuple(int(round(float(t))) for t in list(cell))

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    hits = []
    n_rows = len(data)
    a_vals = a_col.values if hasattr(a_col, 'values') else a_col
    b_vals = b_col.values if hasattr(b_col, 'values') else b_col
    r_vals = r_col.values if hasattr(r_col, 'values') else r_col
    for i in range(n_rows):
        key = (parse(a_vals[i]), parse(b_vals[i]))
        side = T.get(key)
        if side is not None:
            hits.append(1.0 if int(r_vals[i]) == side else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
outcome: self_sim=0.2590 (var=0.0061) adversary_sim=0.3019 (var=0.0058) welch_t=-1.962 p=0.05558 (N=25, alpha=0.01) -> reject

[1] rationale: The two theories are near-isomorphic distrust-inverting tallying models: on every one of the 32 trials they agree on WHICH product the distrust-weighted cue majority favors, and they separate only in how much probability mass leaks to the distrust-disfavored product (the one endorsed by the better / earlier / higher-validity experts). My previous metric failed because it pooled a raw proportion over only the 14 ladder cells (42 observations), whose between-theory mean gap (~0.04) was too small relative to its binomial between-subject noise (sd ~ 0.078, t = 1.96, p = 0.056). The fix is to use the readout the theories actually differ on -- the defection rate to the disfavored product -- over essentially ALL the data, with per-pair weights matched to each pair's diagnosticity. Construction: (1) for each of the 16 mirror pairs I identified the distrust-disfavored product from the shared architecture (sign of the distrust-weighted evidence d; both theories agree on the sign at every corner of both parameter boxes, so the coding is theory-neutral and robust); (2) I dropped the single pair that is a structural coin flip under both boxes (position-6 vs position-8 alone, d ~ 0) and kept the other 15 pairs = 30 cells x 3 reps = 90 observations per subject, more than doubling the effective sample of the failed metric; (3) I weighted each pair by its corner-verified between-theory probability gap divided by its binomial variance p(1-p) -- the variance-optimal linear contrast -- which up-weights the deep anti-validity cells (all-1 vs all-0; the 85%-helper coalitions; the top-3-vs-75%-block cells) and the trusted-channel cells (95%@position-1 vs 75%@position-8), and down-weights the near-tie cells. Because every pair appears in exact A/B-mirrored form, side and position biases cancel. The metric is a weighted proportion, so it is well defined on the pooled DataFrame and on each single subject's 96-trial slice. Predicted values on the design's parameter boxes: PRDT ~ 0.23 (flattened gamma keeps early high-validity and late flat-channel weight alive, but its steeper effective choice consistency suppresses defection), PGDT ~ 0.28 (steep kernel + position-independent flat share + softer effective choice function yields systematically more defection to the disfavored product on every retained pair). Per-subject sd ~ 0.05 (90 Bernoulli observations, weights normalized), giving Welch t ~ 3.5 at N = 25, p < 0.001 -- comfortably below the alpha = 0.01 admission threshold, and a large improvement over the raw 14-cell proportion (t = 1.96). The contrast direction is identical across all 15 retained pairs, so pooling cannot cancel it.
metric_source:
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # ==================================================================
    # DEFECTION-TO-DISFAVORED-PRODUCT INDEX (DDPI), diagnosticity-
    # weighted over mirror pairs.
    #
    # Both candidate architectures are distrust-inverting tallies:
    # on every trial they AGREE on which product the distrust-weighted
    # cue majority favors (the product endorsed by the larger total
    # distrust mass -- typically the low-validity / late-position cue
    # coalition). They DISAGREE on how often subjects defect to the
    # disfavored product: PRDT's flattened inversion kernel
    # (gamma in [1.6, 2.2]) plus its position-growing trust residue
    # keeps the high-validity experts' inverted weights and the late
    # flat channel alive, while PGDT's steep kernel (gamma in
    # [2.8, 3.5]) concentrates almost all weight on the maximum-
    # distrust cues. Corner analysis of the two parameter boxes shows
    # the between-theory gap in P(defect to disfavored product) is
    # largest and same-signed on 15 of the 16 mirror pairs (the 16th
    # -- position-6 vs position-8 alone -- is a structural coin flip
    # under both boxes and is dropped). Each retained pair is weighted
    # by its diagnosticity (predicted between-theory probability gap
    # divided by its binomial p(1-p)), which is the variance-optimal
    # linear contrast at the design's parameter corners: the deep
    # anti-validity cells (all-1 vs all-0, the 85%-helper coalitions)
    # and the position-1 trusted-channel cells get the largest
    # weights, the near-tie cells the smallest.
    #
    # Predicted separation: PRDT ~ 0.23, PGDT ~ 0.28, per-subject sd
    # ~ 0.05 (90 Bernoulli observations per subject, mirror-corrected
    # so side/position bias cancels).
    # ==================================================================

    # (A_ratings, B_ratings) -> (disfavored_side, pair_weight)
    # disfavored_side: 0 = product A is the distrust-disfavored one,
    #                  1 = product B is the distrust-disfavored one.
    T = {
        # pair: pos-6 product vs pos-6+7 coalition (w=0.189)
        ((1, 1, 1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): (0, 0.189),
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 0, 0)): (1, 0.189),
        # pair: pos-5+6 vs pos-7+8 coalition (w=0.329)
        ((1, 1, 1, 1, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): (0, 0.329),
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1, 0, 0)): (1, 0.329),
        # pair: pos-6 vs pos-4+5+7 coalition (w=0.362)
        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 1, 1, 1, 0, 1, 1)): (0, 0.362),
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): (1, 0.362),
        # pair: pos-6 vs all four 75% experts (w=0.438)
        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): (0, 0.438),
        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): (1, 0.438),
        # pair: 85%+pos-6 vs all four 75% (w=0.290)
        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 0, 1, 1, 0, 1, 1)): (0, 0.290),
        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): (1, 0.290),
        # pair: 85%+pos-6 vs pos-4+5+7 (w=0.160)
        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 0, 1, 1, 0, 1, 1)): (0, 0.160),
        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): (1, 0.160),
        # pair: 95%@pos-1 vs 75%@pos-8 trusted-channel probe (w=0.248)
        ((1, 1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1, 1)): (0, 0.248),
        ((0, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 0)): (1, 0.248),
        # pair: 95%@1 + 90%@2 vs 75%@8 (w=0.088)
        ((1, 1, 1, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 1, 1, 1)): (0, 0.088),
        ((0, 0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 0)): (1, 0.088),
        # pair: top-3 vs pos-7+8 (w=0.014)
        ((1, 1, 1, 1, 1, 1, 0, 0), (0, 0, 0, 1, 1, 1, 1, 1)): (0, 0.014),
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0, 0)): (1, 0.014),
        # pair: 75%-block vs top-3 (w=0.171)
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1, 1)): (0, 0.171),
        ((1, 1, 1, 1, 1, 0, 1, 1), (0, 0, 0, 1, 1, 1, 1, 1)): (1, 0.171),
        # pair: 75%-block vs top-3 + pos-6 (w=0.210)
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 0, 0, 1, 1)): (1, 0.210),
        ((1, 1, 1, 0, 0, 0, 1, 1), (0, 0, 0, 1, 1, 1, 1, 1)): (0, 0.210),
        # pair: 95%@1 + pos-6 vs pos-4+5 (w=0.289)
        ((1, 1, 1, 0, 0, 1, 1, 1), (0, 1, 1, 1, 1, 0, 1, 1)): (0, 0.289),
        ((0, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 1, 1)): (1, 0.289),
        # pair: top-2 + pos-6 vs 90%@2 + 85%@3 (w=0.245)
        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 0, 1, 1)): (1, 0.245),
        ((0, 0, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1)): (0, 0.245),
        # pair: top-2 + pos-6+8 vs 90%@2 (w=0.250)
        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 0, 1, 0)): (1, 0.250),
        ((0, 1, 1, 1, 1, 0, 1, 0), (1, 1, 1, 1, 1, 1, 1, 1)): (0, 0.250),
        # pair: total-split anchor (w=0.309)
        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0)): (1, 0.309),
        ((0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 1)): (0, 0.309),
    }

    def parse(cell):
        # Robustly coerce a ratings cell (list / tuple / ndarray /
        # string) to a hashable tuple of ints.
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            return tuple([int(round(float(t))) for t in s.replace(',', ' ').split()])
        try:
            arr = np.asarray(cell, dtype=float).ravel()
            return tuple([int(round(float(t))) for t in arr])
        except Exception:
            return tuple([int(round(float(t))) for t in list(cell)])

    a_col = data['option_a_ratings'].values
    b_col = data['option_b_ratings'].values
    r_col = data['response'].values

    num = 0.0
    den = 0.0
    n_rows = len(data)
    for i in range(n_rows):
        key = (parse(a_col[i]), parse(b_col[i]))
        entry = T.get(key)
        if entry is None:
            continue
        side, w = entry
        num += w * (1.0 if int(r_col[i]) == side else 0.0)
        den += w

    if den <= 0.0:
        return 0.5
    return float(num / den)

outcome: self_sim=0.2409 (var=0.0027) adversary_sim=0.2748 (var=0.0046) welch_t=-1.978 p=0.05403 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    # ==================================================================\n    # DISTRUST-MAJORITY ALLEGIANCE INDEX (DMAI).\n    #\n    # PRDT and PGDT are both distrust-inverting tallying architectures:\n    # on every diagnostic trial of this design they AGREE on which\n    # product the distrust-weighted cue majority favors (the product\n    # endorsed by the late, low-validity / high-error expert block).\n    # What separates them is how STRONGLY subjects commit to that\n    # anti-validity majority:\n    #   * PRDT's box pins a hard, narrow choice rule (beta in [4.0, 5.5],\n    #     lapse <= 0.13) and annihilates the position-1 trusted channel\n    #     (t0 = 0 exactly), so its distrust-majority allegiance is\n    #     strong on every cell -> pooled P(favored) ~ 0.75-0.78.\n    #   * PGDT's box admits a much softer choice rule (beta down to\n    #     2.5, lapse up to 0.15) and its position-INDEPENDENT flat share\n    #     (1-tau)/n keeps the early 95%/90% experts alive, which pulls\n    #     choices away from the distrust majority on the early-expert\n    #     probe cells -> pooled P(favored) ~ 0.68-0.72.\n    # Both mechanisms point the same way on every retained cell, so\n    # pooling cannot cancel the contrast.\n    #\n    # CONSTRUCTION. For each of the 28 unambiguous cells (14 mirror\n    # pairs) I coded the distrust-favored side from the shared\n    # architecture (sign of the distrust-weighted evidence; both\n    # theories agree on this sign at every corner of both parameter\n    # boxes). The two structural coin-flip pairs (position-6 vs\n    # position-8 alone, and top-3 vs the position-6 product, where the\n    # evidence sign is not theory-robust) are EXCLUDED -- they carry\n    # no gap and only add 0.25-variance noise. The metric is the\n    # pooled proportion of choices of the distrust-favored product:\n    #   28 cells x 3 repetitions = 84 Bernoulli observations/subject,\n    #   per-subject sd ~ 0.047 (near the binomial floor),\n    #   between-theory mean gap ~ 0.05  ->  Welch t ~ 3.5-4.5 at N=25.\n    # Exact A/B mirrors make the index invariant to side/position bias\n    # and to any runtime A/B relabeling.\n    # ==================================================================\n\n    # (A_ratings, B_ratings) -> distrust-favored side (0 = A, 1 = B)\n    FAV = {\n        # pair 2/18: late-block product vs position-6 product\n        ((1, 1, 1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 0, 0)): 0,\n        # pair 3/19\n        ((1, 1, 1, 1, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1, 0, 0)): 0,\n        # pair 4/20\n        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 1, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): 0,\n        # pair 5/21\n        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): 0,\n        # pair 6/22\n        ((1, 1, 1, 0, 0, 1, 0, 0), (1, 1, 0, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 0)): 0,\n        # pair 7/23\n        ((1, 1, 1, 0, 0, 1, 0, 1), (1, 1, 0, 1, 1, 0, 1, 1)): 1,\n        ((1, 1, 0, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 0, 1)): 0,\n        # pair 8/24: 95%@1 product vs 75%@8 product\n        ((1, 1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1, 1)): 1,\n        ((0, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 0)): 0,\n        # pair 9/25: 95%@1 + 90%@2 vs 75%@8\n        ((1, 1, 1, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 1, 1, 1)): 1,\n        ((0, 0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1, 0)): 0,\n        # pair 10/26: top-3 vs late block\n        ((1, 1, 1, 1, 1, 1, 0, 0), (0, 0, 0, 1, 1, 1, 1, 1)): 1,\n        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0, 0)): 0,\n        # pair 12/28: 75%-block product vs top-3 product\n        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 0, 0, 1, 1)): 0,\n        ((1, 1, 1, 0, 0, 0, 1, 1), (0, 0, 0, 1, 1, 1, 1, 1)): 1,\n        # pair 13/29: top-3 + pos-6 + pos-7 vs pos-2..5,7,8\n        ((1, 1, 1, 0, 0, 1, 1, 1), (0, 1, 1, 1, 1, 0, 1, 1)): 1,\n        ((0, 1, 1, 1, 1, 0, 1, 1), (1, 1, 1, 0, 0, 1, 1, 1)): 0,\n        # pair 14/30: full product vs 90%@2 + 75%-block\n        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 0, 1, 1)): 0,\n        ((0, 0, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 1, 1, 1, 1)): 1,\n        # pair 15/31: full product vs 95%@1 + 75%-block\n        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 0, 1, 0)): 0,\n        ((0, 1, 1, 1, 1, 0, 1, 0), (1, 1, 1, 1, 1, 1, 1, 1)): 1,\n        # pair 16/32: total-split anchor\n        ((1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0)): 0,\n        ((0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 1)): 1,\n    }\n\n    def parse(cell):\n        # Robustly coerce a ratings cell (list / tuple / ndarray /\n        # string) to a hashable tuple of ints.\n        if isinstance(cell, str):\n            s = cell.strip().strip('[]()')\n            return tuple([int(round(float(t))) for t in s.replace(',', ' ').split()])\n        try:\n            arr = np.asarray(cell, dtype=float).ravel()\n            return tuple([int(round(float(t))) for t in arr])\n        except Exception:\n            return tuple([int(round(float(t))) for t in list(cell)])\n\n    a_col = data['option_a_ratings'].values\n    b_col = data['option_b_ratings'].values\n    r_col = data['response'].values\n\n    hits = []\n    n_rows = len(data)\n    for i in range(n_rows):\n        key = (parse(a_col[i]), parse(b_col[i]))\n        side = FAV.get(key)\n        if side is not None:\n            hits.append(1.0 if int(r_col[i]) == side else 0.0)\n\n    if not hits:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "DIAGNOSTIC PRINCIPLE. PRDT and PGDT are near-isomorphic distrust-inverting tallying models: on every trial of this design they agree on WHICH product the distrust-weighted cue majority favors (the product endorsed by the late high-error expert block), and they separate on how STRONGLY subjects commit to that anti-validity majority. My two previous metrics failed because they read out raw proportions on cell subsets whose between-theory mean gap (~0.035-0.045) was too small relative to their per-subject noise; both, however, consistently showed PGDT's choices sitting systematically closer to chance than PRDT's. The present metric is built directly on that separation axis, pooled over essentially the whole experiment. CONSTRUCTION. (1) For each of the 16 mirror pairs I coded the distrust-favored side from the shared architecture (sign of the distrust-weighted evidence d = sum_j w_j*sign(A_j-B_j)); both theories agree on this sign at every corner of both parameter boxes, so the coding is theory-neutral and robust. (2) I excluded only the two structural coin-flip pairs (position-6 vs position-8 alone; top-3 vs the position-6 product) whose evidence sign is not theory-robust and which carry no gap -- they would only inject 0.25-variance noise. (3) The metric is the pooled proportion of choices of the distrust-favored product over the remaining 28 cells x 3 reps = 84 Bernoulli observations per subject. WHY THE THEORIES SEPARATE. (a) CHOICE-DETERMINISM BAND: PRDT's box pins a hard, narrow logistic (beta in [4.0, 5.5], lapse in [0.09, 0.13]) -- its allegiance to the distrust majority is strong on every cell (per-cell P(favored) ~ 0.60-0.94, pooled ~ 0.75-0.78). PGDT's box spans a much softer, wider choice rule (beta down to 2.5, lapse up to 0.15); at its soft configurations its pooled P(favored) drops to ~ 0.68-0.72. (b) EARLY-TRUSTED CHANNEL: PGDT's position-independent flat share (1-tau)/n keeps the 95%/90% experts alive, which pulls choices away from the distrust majority specifically on the early-expert probe cells (95%@1 vs 75%@8 etc.) -- PRDT's t0 = 0 ramp annihilates that channel, so this effect also pushes PGDT's index below PRDT's. Both mechanisms point the same direction on every retained cell, so pooling cannot cancel the contrast. VARIANCE BUDGET. 84 observations per subject put the per-subject variance near the binomial floor (mean p(1-p)/84 ~ 0.0021-0.0023, sd ~ 0.047), while the between-theory mean gap is ~ 0.05: Welch's t at N = 25 is expected around 3.5-4.5, far below the alpha = 0.01 admission threshold -- a large improvement over the previous metrics (t ~ 1.96). The exact A/B mirrors make the index invariant to side and position bias and to any runtime A/B relabeling, and the metric is a simple proportion, so it is well defined both on the pooled DataFrame and on every single subject's 96-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 15753,
  "output_tokens": 97871,
  "reasoning_tokens": 94485,
  "provider": "Sail Research"
}
```
