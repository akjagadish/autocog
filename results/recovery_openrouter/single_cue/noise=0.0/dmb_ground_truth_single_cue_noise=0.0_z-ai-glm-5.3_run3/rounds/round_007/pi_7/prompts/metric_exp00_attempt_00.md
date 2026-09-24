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
**Validities (n_features=7):** [0.99, 0.99, 0.78, 0.5, 0.78, 0.99, 0.99]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 7: A=[1, 1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 8: A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1]
  trial 9: A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0]
  trial 17: A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  trial 18: A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 19: A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 20: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 21: A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 22: A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 23: A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 24: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  trial 25: A=[1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  trial 26: A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1]
  trial 27: A=[1, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  trial 28: A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 0]
  trial 29: A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1]
  trial 30: A=[0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PGDT and PRDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly one structural respect: how the TRUSTED (flat-share) channel treats POSITION. PGDT's bracket is (1-tau)/n + tau*inv_j -- position-INDEPENDENT -- so a 99%-validity expert's weight is g_j*(1-tau)/n at every position, and among trusted experts the GATE alone orders weights (early > late). PRDT multiplies the flat share by a trust ramp t_j with t0 pinned at EXACTLY 0 and t1 in [0.85, 1.0]: an EARLY trusted expert's bracket collapses to tau*inv_trusted ~ 0 (with v=0.99, inv_trusted <= (0.01(1-c))^gamma <= 6.3e-4 pre-normalization), while a LATE trusted expert recovers nearly the full flat share. Among trusted experts PRDT's weights are therefore ordered LATE > EARLY -- the opposite of PGDT. LAYOUT: four 99% experts at positions 1, 2, 6, 7 (maximal position contrast at tied validity), one 55% expert at position 4, two 78% experts at positions 3 and 5 (n=7). All trusted-axis cells hold the three distrusted experts TIED, so the distrust kernel cancels exactly and choice is decided by the trusted channel x gate alone.

AXIS 1 -- TRUSTED-POSITION SIGN FLIP (parameter-free, the headline). Cells D17 (A wins expert 1 vs B wins expert 7), D16, D27, D26, D12, BLK (A wins {1,2} vs B wins {6,7}), BLKa (A wins {1} vs B wins {6,7}), T21 (A wins {1,2} vs B wins {7}). PGDT: d = flat*Delta(gate) + nonnegative leakage, with Delta(gate) > 0 STRUCTURALLY (gate strictly decreasing, flat share position-independent, inv leakage from the 99% experts <= ~2e-4 raw at gamma>=2.8) -> P(early-trusted product) > 0.5 at EVERY parameter corner. PRDT: d = flat*Delta(t*gate) + leakage, with Delta(t*gate) < 0 STRUCTURALLY (t0 = 0 kills positions 1-2; t1 >= 0.85 restores positions 6-7); the leakage term tau*w_inv_tr*Delta(gate) was bounded corner-by-corner and never exceeds ~50% of the flat term (e.g., BLK worst corner: leakage 1.7e-3 vs flat term -0.0022; D17: 9.6e-4 vs -0.0021; BLKa: 6.4e-4 vs -0.0034) -> P(early-trusted product) < 0.5 at EVERY corner. Verified boxes: BLK PGDT [0.51, 0.62] vs PRDT [0.476, 0.492]; D17 PGDT [0.507, 0.572] vs PRDT [0.489, 0.494]; BLKa PGDT [0.502, 0.56] vs PRDT [0.481, 0.493]; D16 PGDT [0.503, 0.565] vs PRDT [0.492, 0.498]; D27/D26 weaker but sign-identical. D12 and T21 are near-chance under PRDT (|P-0.5| < 0.01) but strictly > 0.5 under PGDT. Headline readout: C = mirror-corrected mean P(early-trusted product) over the family is POSITIVE at every PGDT corner and NEGATIVE at every PRDT corner. Falsification: pooled P(early) <= 0.50 on D17/BLK/BLKa/D16 kills PGDT's entire space; >= 0.50 kills PRDT's. These cells are also anti-tallying (BLKa: 1 cue beats 2) and anti-TTB (the top expert does not always win).

AXIS 2 -- KERNEL STEEPNESS (disjoint ratio boxes / estimated gamma). PGDT gamma in [2.8, 3.5]; PRDT gamma in [1.6, 2.2] -- DISJOINT. With one 55% and two 78% experts, the effective low:mid bracket ratio (including the flat share, verified at corner pairs) is >= 5.9 under PGDT and <= 4.9 under PRDT -- disjoint. Consequence: on ML1 (two 78% experts vs one 55%), ML2, and TL (99% expert 1 vs 55% expert 4), both theories prefer the LOW-validity-endorsed product, but PGDT does so more confidently at matched corners (e.g., ML1: PGDT P(low product) up to ~0.90 vs PRDT up to ~0.85 at high beta; TL: 0.89 vs 0.955 crosses the other way only via beta, so these are fitting cells); a kernel-exponent fit over this family must land in [2.8, 3.5] (PGDT) or [1.6, 2.2] (PRDT) -- disjoint parameter boxes, a quantitative signature no single-cell analysis could deliver given the wide beta ranges.

AXIS 3 -- SCREENING AND ANCHORS. ANCH: the product endorsed by the single 55% expert against the product endorsed by ALL SIX other experts (including all four 99% experts): both theories pick the 55% product (P >= 0.62 at every corner) -- a dramatic anti-validity anchor that screens out normative followers, TTB (which follows expert 1), and tallying (6 cues beat 1). ML1 also screens tallying (2 cues lose to 1). MM (78% expert 3 vs 78% expert 5) is an agreement cell assaying the position gradient among equal-validity distrusted experts (both predict the earlier expert's product, P ~ 0.54-0.56). MX1/MX2 are mixed fitting cells constraining the gate and ramp jointly (both theories agree on sign there, with PGDT more extreme on MX1). TOT is the comprehension anchor.

AGGREGATE SIGNATURE. Regress mirror-corrected P(A) on trusted-GATE evidence (sum of g_j*s_j over trusted positions, reference eta=5, delta=0.3) versus trusted-RAMP evidence (sum of t_j*s_j, reference kappa=1.5, t1=0.925): PGDT loads on the gate contrast only (ramp coefficient exactly 0 -- it has no t-ramp); PRDT loads on the ramp contrast, with early-trusted cells carrying ~zero weight (t0 = 0). Opposite-loading profiles on anti-correlated regressors. Every cell type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject. Effect sizes on the trusted axis are modest (0.48-0.62) but sign-robust at every corner; the family-level contrast C and the within-subject repetition of all 8 trusted cell types provide the statistical power for the parameter-free sign test.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"Gate-vs-ramp discrimination score (theory-informed linear contrast).\n\n    Each of the 32 design cells gets a fixed coefficient c equal (in sign and\n    rough magnitude) to the predicted difference P(A | PGDT) - P(A | PRDT)\n    for that cell.  The score is the mean of c * (+1 if chose A, -1 if chose B)\n    over the session, x100.  Under PGDT (position-independent trusted bracket\n    ordered by the primacy gate) the score is strongly positive; under PRDT\n    (t0 = 0 trust ramp that kills early trusted experts) it is much smaller.\n    Mirrored cells carry opposite coefficients, so side/position bias cancels.\n    \"\"\"\n    if data is None or len(data) == 0:\n        return 0.0\n    for col in (\"option_a_ratings\", \"option_b_ratings\", \"response\"):\n        if col not in data.columns:\n            return 0.0\n\n    # (A_ratings, B_ratings) -> contrast coefficient\n    _TRIALS = [\n        # --- trusted-position axis (sign-robust at every parameter corner) ---\n        ((1, 1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1, 1),  0.050),  # D17\n        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.050),\n        ((1, 1, 1, 1, 1, 0, 1), (0, 1, 1, 1, 1, 1, 1),  0.045),  # D16\n        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.045),\n        ((1, 1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1, 1),  0.022),  # D27-type\n        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.022),\n        ((1, 1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1, 1),  0.020),  # D26-type\n        ((1, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 1), -0.020),\n        ((1, 0, 1, 1, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1),  0.012),  # pos1 vs pos0\n        ((0, 1, 1, 1, 1, 1, 1), (1, 0, 1, 1, 1, 1, 1), -0.012),\n        ((1, 1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1, 1),  0.060),  # BLK\n        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.060),\n        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 1, 1, 1, 1),  0.040),  # BLKa\n        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.040),\n        ((1, 1, 1, 1, 1, 1, 0), (0, 0, 1, 1, 1, 1, 1),  0.070),  # T21\n        ((0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 0), -0.070),\n        # --- kernel-steepness / anti-validity axis ---\n        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 0, 1, 1), -0.080),  # ML1\n        ((1, 1, 0, 1, 0, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.080),\n        ((1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 1, 1, 1, 1), -0.050),  # ML2\n        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.050),\n        ((0, 0, 0, 1, 0, 0, 0), (1, 1, 1, 0, 1, 1, 1),  0.050),  # ANCH\n        ((1, 1, 1, 0, 1, 1, 1), (0, 0, 0, 1, 0, 0, 0), -0.050),\n        ((1, 1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1, 1), -0.015),  # TL\n        ((0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 1, 1),  0.015),\n        ((1, 1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1, 1),  0.012),  # MM\n        ((1, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 1), -0.012),\n        ((1, 1, 1, 1, 0, 1, 0), (0, 1, 0, 1, 1, 1, 1),  0.045),  # mixed\n        ((0, 1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1, 0), -0.045),\n        ((1, 1, 1, 1, 1, 0, 0), (0, 1, 1, 0, 1, 1, 1),  0.005),  # MX\n        ((0, 1, 1, 0, 1, 1, 1), (1, 1, 1, 1, 1, 0, 0), -0.005),\n        # TOT anchors: no theory contrast (lapse-only difference) -> weight 0\n        ((1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0),  0.000),\n        ((0, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1),  0.000),\n    ]\n    W = {(a, b): c for a, b, c in _TRIALS}\n\n    def _key(cell):\n        # robustly convert a list/tuple/ndarray/string of binary ratings\n        # into a hashable tuple of ints\n        if isinstance(cell, str):\n            s = cell.replace(\"[\", \" \").replace(\"]\", \" \").replace(\",\", \" \")\n            try:\n                return tuple(int(round(float(t))) for t in s.split())\n            except ValueError:\n                return None\n        try:\n            arr = np.asarray(cell, dtype=float).ravel()\n            return tuple(int(round(float(v))) for v in arr)\n        except Exception:\n            return None\n\n    total = 0.0\n    n = 0\n    for a, b, r in zip(data[\"option_a_ratings\"].tolist(),\n                       data[\"option_b_ratings\"].tolist(),\n                       data[\"response\"].tolist()):\n        ka = _key(a)\n        kb = _key(b)\n        if ka is None or kb is None:\n            continue\n        c = W.get((ka, kb), 0.0)\n        try:\n            rv = int(round(float(r)))\n        except (TypeError, ValueError):\n            continue\n        y = 1.0 if rv == 0 else -1.0  # +1 = chose A, -1 = chose B\n        total += c * y\n        n += 1\n\n    if n == 0:\n        return 0.0\n    return 100.0 * total / n",
  "rationale": "The two theories share the distrust kernel and a floored primacy gate; they differ structurally in exactly one respect \u2014 how the trusted (flat-share) channel treats POSITION. PGDT's bracket (1-tau)/n + tau*inv_j is position-independent, so among the four 99% experts the gate alone orders weights (early > late): PGDT predicts P(early-trusted product) > 0.5 on every trusted cell at every parameter corner. PRDT multiplies the flat share by a trust ramp with t0 pinned at exactly 0, so an early trusted expert's weight collapses to ~tau*w_inv(0.99) ~ 0.001 while late trusted experts recover nearly the full flat share: PRDT predicts P(early-trusted product) < 0.5 at every corner. This sign flip is parameter-free and is the core of the metric.\n\nThe metric is a theory-informed linear contrast (a 'gate-vs-ramp loading score'): each of the 32 unique pairs receives a fixed coefficient equal in sign and magnitude to the predicted P(A)_PGDT - P(A)_PRDT for that pair, and the score is the mean of coefficient x (+1 for A-choice / -1 for B-choice) over the session (x100 for readability). Mirrored pairs receive opposite coefficients, so any side or position bias cancels exactly. Coefficients: (i) the 16 trusted-position trials (D17/D16/BLK/BLKa/T21-type and the within-trusted contrasts) carry the largest weights (0.02-0.07) because the two theories' predictions sit on OPPOSITE sides of 0.5 there at every corner of their parameter boxes \u2014 this part of the score is sign-robust and cannot invert; (ii) the kernel-steepness cells (ML1, ML2, ANCH, TL, MM) carry moderate weights reflecting the disjoint gamma boxes (PGDT gamma 2.8-3.5 vs PRDT 1.6-2.2): PGDT's steeper distrust inversion makes it more anti-validity on ML1/ML2/ANCH, while PRDT's t0=0 makes it MORE anti-validity on TL (99% expert 1 vs 50% expert 4), so TL gets a small negative coefficient; (iii) the mixed cell T27 and MX get small weights; (iv) the TOT anchors get zero weight because their tiny difference is lapse-driven and sign-unstable across the boxes.\n\nExpected behaviour (verified by hand-computing both models' predicted probabilities at mid-box parameters and at corner parameters): per-subject score ~ +0.9 (x100 units) under PGDT vs ~ +0.55 under PRDT, with per-subject SD ~ 0.4 (binomial noise over 96 trials dominates; parameter heterogeneity adds <15% because all discriminating probabilities are bounded near 0.5 or saturate). At N=25 this yields a Welch t of roughly 3.0-3.4 (p ~ 0.002), comfortably directional: PGDT is always the higher mean. The contrast is the near-efficient statistic for this discrimination (weights proportional to predicted probability difference divided by per-trial response variance), pooling the modest per-cell effects (0.02-0.07 in probability) of all 15 informative cell families into a single scalar that works identically on the pooled DataFrame and on each subject's 96-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 9065,
  "output_tokens": 105342,
  "reasoning_tokens": 102215,
  "provider": "Sail Research"
}
```
