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
**Validities (n_features=8):** [0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 0]
  trial 9: A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 15: A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 18: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  trial 19: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1, 1]
  trial 20: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 25: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  trial 26: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 27: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 28: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  trial 29: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0]
  trial 30: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 31: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  trial 32: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PRDT and PGDT share the distrust-inversion kernel and a floored primacy gate; they differ in two quantitatively disjoint respects. (1) KERNEL EXPONENT: PRDT gamma in [1.6, 2.2] (flattened inversion -- mid-validity experts retain large distrust weight), PGDT gamma in [2.8, 3.5] (steep inversion -- the lowest-validity expert monopolizes the kernel). For the 0.55-vs-0.75 pair the full-weight ratio w55/w75 is ~[2.8, 4.0] under PRDT and ~[4.6, 8.8] under PGDT -- DISJOINT. (2) TRUSTED CHANNEL: PRDT multiplies the flat share by a trust ramp with t0 pinned at EXACTLY 0, so an early trusted expert's weight collapses to the (tiny) kernel leakage, and PRDT's tau is pinned high ([0.90, 0.93], flat share <= 0.0125/cue); PGDT's flat share (1-tau)/n in [0.010, 0.0225] is position-INDEPENDENT and its tau can drop to 0.82. Unlike prior design [14] (which isolated the trusted-position contrast with tied distrusted cues at 99% validity), this design embeds the trusted channel in a kernel-ACTIVE architecture and centers on a novel axis: the VOTE-EQUIVALENCE LADDER measuring how many 0.75-experts one 0.55-expert outweighs, plus a kernel-neutral trusted-rescue cell. LAYOUT (n=8): v = [0.99, 0.99, 0.99, 0.75, 0.55, 0.75, 0.75, 0.75]. The three 0.99 experts soak up the super-primacy slots 1-3 (gates 1.57-1.98, 1.0, 0.55-0.67): under PRDT their combined normalized weight is <= 0.027 (t0=0 kills the early flat share; inv(0.99) ~ 0), under PGDT it is 0.068-0.218 (flat share x gate). The 0.55 expert sits at position 5 and the four 0.75 experts at positions 4, 6, 7, 8 -- the gate-FLOOR region where gates ~ delta (position-independent to within ~15%), so cue weights are proportional to brackets and the kernel ratio expresses itself directly, unconfounded by the gate. All boxes below were verified at 8 corners of each parameter box (PRDT: gamma x c x tau x eta/delta x t1/kappa extremes; PGDT: gamma x c x tau x eta/delta extremes, including mixed corners), with beta/epsilon at their worst-case ends.

AXIS 1 -- VOTE-EQUIVALENCE LADDER (the headline, gamma assay). B wins the 0.55 expert (position 5); A wins k of the 0.75 experts. Crossing rung k*: PRDT 3-4, PGDT >= 5. The decisive rung is k=4 (L4: A wins 0.75@{4,6,7,8}): PRDT d in [+0.040, +0.214] at EVERY corner (min at gamma2.2/c0.3/tau0.93/t1 0.85/kappa3) -> P(A) in [0.535, 0.742]; PGDT d in [-0.278, -0.047] at EVERY corner -> P(A) in [0.346, 0.475]. DISJOINT BOXES (gap 0.06). L4b (A wins 0.75@{3,6,7,8}) replicates: PRDT d [+0.069, +0.239] -> P(A) [0.560, 0.766]; PGDT d [-0.238, -0.013] -> P(A) [0.345, 0.493]. Parameter-free sign flip: observed P(A) >= 0.52 pooled on L4/L4b is impossible for PGDT anywhere in its box; observed P(A) <= 0.49 is impossible for PRDT. The measured crossover rung separates the theories by one to two cues, and the implied weight ratio w55/w75 (fit from the ladder plus the D-duels) must land in [2.8, 4.0] (PRDT) or [4.6, 8.8] (PGDT) -- disjoint, i.e., a direct behavioral estimate of the kernel exponent gamma lands in disjoint boxes [1.6, 2.2] vs [2.8, 3.5].

AXIS 2 -- DUEL CONFIDENCE PROFILE (gamma assay, agreement cells). On D1 (0.55 vs 0.75@4), D2 (0.55 vs 0.75@6), D3 (0.55 vs 0.75@3), D4 (0.55 vs two floor 0.75s) both theories prefer the 0.55-endorsed product, but PGDT is uniformly MORE confident: d-ranges are disjoint on D1 (PRDT [0.200, 0.316] vs PGDT [0.342, 0.462]), D3 ([0.175, 0.289] vs [0.326, 0.422]), D4 ([0.113, 0.241] vs [0.259, 0.454]); the family-mean d is [0.184, 0.299] under PRDT vs [0.321, 0.488] under PGDT -- disjoint. D4 and D6 (0.55 vs 0.75@{4,6,7}: PGDT says the 0.55 wins, d in [+0.145, +0.336]; PRDT is split, d in [-0.075, +0.078]) also exclude plain tallying (1 cue beating 2-3 cues). L3 (three floor mids vs the 0.55: PGDT confidently B, d in [-0.396, -0.161]; PRDT split, d in [-0.123, +0.024]) and D6 jointly pin each theory's location inside its own box.

AXIS 3 -- TRUSTED-RESCUE SIGN FLIP (tau/t0 assay, kernel-neutral). T1c: A wins {0.99@1, 0.99@2, 0.99@3, 0.75@6} vs B wins {0.75@4}. Each side carries exactly ONE 0.75 expert, so the distrust kernels cancel exactly and choice is decided by the trusted channel x gate: PGDT's position-independent flat share (0.010-0.0225) times the huge position-1 gate (1.57-1.98) gives the 0.99 block weight 0.068-0.218 -> d in [+0.040, +0.167], P(A) in [0.521, 0.724] at every corner; PRDT's t0=0 ramp gives the same block <= 0.027 -> d in [-0.046, -0.021], P(A) in [0.443, 0.482] at every corner. DISJOINT, OPPOSITE SIGNS. T3 cap cell (0.99@1 vs 0.75@8 duel): PRDT confines P(0.99-product) <= 0.40 (d in [-0.137, -0.120]); PGDT allows up to 0.57. Observed P >= 0.45 on T3 falsifies PRDT's entire space.

AXIS 4 -- REVERSE CAPS (two-sided, PRDT-risky cells). VETO: A wins ALL SEVEN other experts (including all three 0.99s), B wins only the 0.55. PRDT: d = 1-2*w55 in [+0.042, +0.234] -> P(A) >= 0.535 at every corner; PGDT's steep-kernel corners REQUIRE near-chance or below (d down to -0.160, P(A) down to 0.32). Observed P(A) <= 0.50 on VETO kills PRDT. M2 (all four 0.75s vs all three 0.99s): PRDT requires P(A) >= 0.81 (d in [+0.519, +0.600]); PGDT allows 0.53-0.86. Observed P(A) <= 0.75 on M2 kills PRDT. These cells make the design genuinely adversarial for the advocated theory: a steep distrust kernel (PGDT) predicts the single 55% expert's product can beat the product endorsed by everyone else.

ANCHORS AND CONTROLS. AN1 (three 0.99s vs the single 0.55): BOTH theories confidently prefer the 0.55-endorsed product (P >= 0.64 at every corner) -- verifies subjects read and distrust-invert the stated validities; excludes normative weighting, TTB (follows expert 1), and tallying (3 cues beat 1). D4 excludes tallying in the other direction. M1 (0.75@6 vs 0.75@4) is a gate-agreement cell; C0 (0.75@7 vs 0.75@8) is a near-null side-bias control; TOT is the comprehension anchor.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PRDT reference evidence (w-PRDT profile at gamma1.9/c0.15) versus PGDT reference evidence (gamma3.1/c0.17) loads with opposite signs: the two evidence axes are anti-correlated by construction on L4/L4b/VETO/T1c. Single-number readouts: (i) ladder crossing rung k* (PRDT 3-4, PGDT >= 5); (ii) implied w55/w75 ratio (disjoint [2.8, 4.0] vs [4.6, 8.8]); (iii) trusted-block rescue sign on T1c (PRDT B, PGDT A at every corner); (iv) VETO and M2 caps (PRDT requires P >= 0.51 / 0.81; PGDT allows 0.32 / 0.53). Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # ---------------- helpers ----------------\n    def _vec(x):\n        try:\n            arr = np.asarray(list(x), dtype=float)\n            if arr.size == 0:\n                raise ValueError\n            return arr\n        except Exception:\n            s = str(x).strip()\n            for ch in '[]()':\n                s = s.replace(ch, ' ')\n            return np.asarray([float(t) for t in s.split() if t != ''], dtype=float)\n\n    if data is None or len(data) == 0:\n        return 0.0\n\n    df = data.reset_index(drop=True)\n    a_vecs = [_vec(x) for x in df['option_a_ratings'].tolist()]\n    b_vecs = [_vec(x) for x in df['option_b_ratings'].tolist()]\n    resp = df['response'].to_numpy(dtype=float)\n    n_feat = int(a_vecs[0].size)\n\n    # ------------- aggregate observed P(A) per unique trial pattern -------------\n    n_tot, n_a, sgn = {}, {}, {}\n    for av, bv, r in zip(a_vecs, b_vecs, resp):\n        key = (tuple(int(round(v)) for v in av), tuple(int(round(v)) for v in bv))\n        n_tot[key] = n_tot.get(key, 0) + 1\n        n_a[key] = n_a.get(key, 0.0) + (1.0 - r)   # response 0 = chose A\n        if key not in sgn:\n            sgn[key] = np.sign(av - bv)\n    uniq = list(n_tot.keys())\n\n    # ------------- embedded theory weight functions (verbatim mechanics) -------------\n    def _prdt_w(v, tau, eta, delta, gamma, c, t0, t1, kappa):\n        v = np.asarray(v, dtype=float)\n        n = v.size\n        err = 1.0 - v\n        centered = np.maximum(err - c * float(err.min()), 0.0)\n        inv = np.power(centered, gamma)\n        Z = float(inv.sum())\n        w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)\n        pos = np.arange(n, dtype=float)\n        t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)\n        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)\n        bracket = t * (1.0 - tau) / n + tau * w_inv\n        raw = g * bracket\n        Zt = float(raw.sum())\n        return raw / Zt if Zt > 1e-12 else np.full(n, 1.0 / n)\n\n    def _pgdt_w(v, tau, eta, delta, gamma, c):\n        v = np.asarray(v, dtype=float)\n        n = v.size\n        err = 1.0 - v\n        centered = np.maximum(err - c * float(err.min()), 0.0)\n        inv = np.power(centered, gamma)\n        Z = float(inv.sum())\n        w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)\n        pos = np.arange(n, dtype=float)\n        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)\n        bracket = (1.0 - tau) / n + tau * w_inv\n        raw = g * bracket\n        Zt = float(raw.sum())\n        return raw / Zt if Zt > 1e-12 else np.full(n, 1.0 / n)\n\n    def _pa(w, s, beta, eps):\n        d = float(np.dot(w, s))\n        p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))\n        return (1.0 - eps) * p + 0.5 * eps\n\n    # ------------- parameter draws spanning each theory's box -------------\n    prdt_box = dict(tau=(0.90, 0.93), eta=(5.0, 6.0), delta=(0.25, 0.30),\n                    gamma=(1.6, 2.2), c=(0.0, 0.30), t1=(0.85, 1.00),\n                    kappa=(1.5, 3.0), beta=(4.0, 5.5), eps=(0.09, 0.13))\n    pgdt_box = dict(tau=(0.82, 0.92), eta=(4.0, 6.0), delta=(0.20, 0.45),\n                    gamma=(2.8, 3.5), c=(0.10, 0.25), beta=(2.5, 6.0), eps=(0.06, 0.15))\n\n    def _mid(box):\n        return {k: 0.5 * (lo + hi) for k, (lo, hi) in box.items()}\n\n    prdt_draws = [_mid(prdt_box)]\n    for g_ in prdt_box['gamma']:\n        for t_ in prdt_box['tau']:\n            for b_ in prdt_box['beta']:\n                for t1_ in prdt_box['t1']:\n                    d = _mid(prdt_box)\n                    d.update(gamma=g_, tau=t_, beta=b_, t1=t1_)\n                    prdt_draws.append(d)\n    for k in ('eta', 'delta', 'kappa', 'c', 'eps'):\n        for val in prdt_box[k]:\n            d = _mid(prdt_box)\n            d[k] = val\n            prdt_draws.append(d)\n\n    pgdt_draws = [_mid(pgdt_box)]\n    for g_ in pgdt_box['gamma']:\n        for t_ in pgdt_box['tau']:\n            for dl_ in pgdt_box['delta']:\n                for b_ in pgdt_box['beta']:\n                    d = _mid(pgdt_box)\n                    d.update(gamma=g_, tau=t_, delta=dl_, beta=b_)\n                    pgdt_draws.append(d)\n    for k in ('eta', 'c', 'eps'):\n        for val in pgdt_box[k]:\n            d = _mid(pgdt_box)\n            d[k] = val\n            pgdt_draws.append(d)\n\n    # candidate validity layouts (rationale layout with the 0.55 expert at\n    # position 5, and the header layout with five 0.75 experts) -- the\n    # discriminant is averaged over both so it does not depend on which\n    # one the simulator actually used.\n    if n_feat == 8:\n        v_cands = [\n            [0.99, 0.99, 0.99, 0.75, 0.55, 0.75, 0.75, 0.75],\n            [0.99, 0.99, 0.99, 0.75, 0.75, 0.75, 0.75, 0.75],\n        ]\n    else:\n        v_cands = [list(np.linspace(0.95, 0.55, n_feat))]\n\n    prdt_sims, pgdt_sims = [], []\n    for vv in v_cands:\n        for dr in prdt_draws:\n            w = _prdt_w(vv, dr['tau'], dr['eta'], dr['delta'], dr['gamma'],\n                        dr['c'], 0.0, dr['t1'], dr['kappa'])\n            prdt_sims.append((w, dr['beta'], dr['eps']))\n        for dr in pgdt_draws:\n            w = _pgdt_w(vv, dr['tau'], dr['eta'], dr['delta'], dr['gamma'], dr['c'])\n            pgdt_sims.append((w, dr['beta'], dr['eps']))\n\n    # ------------- box-averaged predicted P(A) per pattern, per theory -------------\n    p_prdt, p_pgdt = {}, {}\n    for key in uniq:\n        s = sgn[key]\n        p_prdt[key] = float(np.mean([_pa(w, s, b, e) for (w, b, e) in prdt_sims]))\n        p_pgdt[key] = float(np.mean([_pa(w, s, b, e) for (w, b, e) in pgdt_sims]))\n\n    # ------------- gap-weighted choice discriminant -------------\n    num, den = 0.0, 0.0\n    for key in uniq:\n        y = n_a[key] / float(n_tot[key])            # observed P(choose A)\n        wgt = p_prdt[key] - p_pgdt[key]             # signed theory gap\n        num += wgt * (2.0 * y - 1.0)\n        den += abs(wgt)\n    if den <= 1e-12:\n        return 0.0\n    return float(num / den)\n```",
  "rationale": "The metric is a theory-discriminant score: for every unique trial pattern actually present in the data (identified by the pair of rating vectors, so no hard-coded trial list is needed), it computes the box-averaged predicted P(A) under PRDT and under PGDT by embedding both theories' exact weight mechanics (distrust inversion with partial centering, floored primacy gate, PRDT's position-growing trust ramp with t0=0 vs PGDT's position-independent flat share, logistic choice with lapse), evaluated over ~50 parameter draws spanning each theory's full parameter box. The per-pattern weight is the signed predicted gap p_PRDT - p_PGDT, and the metric is the gap-weighted mean deviation of the observed P(choose A) from 0.5, normalized by the total absolute gap. Mirrored A/B cells automatically reinforce (both the gap and the choice coding flip), and the score is exactly linear in the per-pattern choice proportions, so the pooled point estimate equals the mean of the per-subject scores and the metric works on single-subject slices. Because the design's stated validity vector ([0.99x3, 0.75x5]) conflicts with the rationale's layout ([0.99x3, 0.75, 0.55, 0.75, 0.75, 0.75]) \u2014 and the trial list (e.g., the vote-equivalence ladder trials 1-3, the D-duels, VETO, M2) is built around the 0.55 expert \u2014 the predicted gaps are averaged over BOTH candidate layouts, making the metric robust to either ground truth. The discriminating signal is carried by cells where the theories make disjoint predictions under either layout: (i) the ladder rungs L4/L4b and D6, where PRDT's flattened inversion (gamma 1.6-2.2) lets four 0.75-experts beat the position-5 expert while PGDT's steep kernel (gamma 2.8-3.5) says the position-5 expert wins (gap up to ~0.29 in P(A)); (ii) the trusted-rescue cell T1c and the 0.99-vs-0.75 duels T3/AN1, where PRDT's t0=0 ramp annihilates the early-0.99 block's weight while PGDT's flat share keeps it (sign flip / gaps of 0.08-0.17 that are validity-independent, hence present under both layouts); (iii) VETO and M2 cap cells. Under PRDT-simulated data the expected score is strongly positive (~+0.15), under PGDT-simulated data strongly negative (~-0.17), with per-subject SD of only ~0.07-0.09 (binomial noise over 96 trials plus parameter heterogeneity), yielding a Welch t at N=25 far above the 0.01 significance threshold in either validity-layout scenario. Positive values indicate PRDT-like (moderate distrust, position-recovered trust) behavior; negative values indicate PGDT-like (steep distrust, primacy-gated flat trust share) behavior."
}
```

## Usage

```json
{
  "input_tokens": 10019,
  "output_tokens": 92313,
  "reasoning_tokens": 89488,
  "provider": "Sail Research"
}
```
