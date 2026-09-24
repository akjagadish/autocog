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
**Validities (n_features=6):** [0.95, 0.85, 0.6, 0.6, 0.85, 0.95]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 12: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 15: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 17: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 18: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 19: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 21: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 23: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 24: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1]

**Rationale:** STRUCTURAL LEVER. PGDT and PBDT share the distrust-inversion kernel and a floored convex primacy gate; they differ in exactly two quantitatively dissociable respects. (1) THE FLAT (TRUSTED) SHARE. PGDT's bracket is (1-tau)/n + tau*inv_j with tau in [0.82, 0.92]: a POSITION-INDEPENDENT flat share of 1.3-3.0% per cue (n=6). PBDT's bracket is t_j*(1-tau)/n + tau*inv_j with tau in [0.91, 0.93] and a trust ramp t_j rising from t0 in [0.25, 0.35] to t1 in [0.95, 1.0]. Consequence, verified at the corners of both boxes: the raw weight of an EARLY high-validity (95%) expert is DISJOINT across the theories -- PGDT [0.013, 0.031] vs PBDT [0.004, 0.007], a 2-7x gap -- because PBDT's t0 crushes the flat share exactly where the gate is maximal. For any cue whose bracket is flat-share-relevant (validity 0.85-0.95), PBDT's ramp CANCELS its gate, so PBDT's position effects on the trusted/mid channel collapse toward zero, while PGDT's remain gate-scaled. (2) GATE SHAPE. PGDT's floor is high (delta 0.20-0.45, eta 4-6); PBDT's gate is steeper with a lower floor (delta 0.15-0.25, eta 6.5-8). On tail-position contrasts among kernel cues (positions 3 vs 4), PGDT's gate contrast (1-delta)(0.6^eta-0.4^eta) reaches 2.5x PBDT's maximum. LAYOUT: palindromic v = [0.95, 0.85, 0.60, 0.60, 0.85, 0.95]. The palindrome creates VALIDITY-MATCHED position duels at three validity levels -- 95s at positions 1 vs 6, 85s at 2 vs 5, 60s at 3 vs 4 -- so the distrust kernel cancels exactly within each duel and choice is decided by gate x flat-share architecture alone. All boxes below are mirror-corrected P(A), verified at the corners of both parameter spaces (tau x eta x delta x gamma x c, with beta/epsilon at worst-case ends).

AXIS 1 -- VALIDITY-MATCHED POSITION-TRANSFER BATTERY (T16, M25, L34 + mirrors). T16 (95@1 vs 95@6): PGDT d = b_tr*(1-delta) in [0.0075, 0.025] raw -> P(A) in [0.51, 0.66]; PBDT d = flat'*(t0 - delta*t1) + leakage in [0.0004, 0.0033] -> P(A) in [0.50, 0.52]. PBDT is HARD-CAPPED at ~0.52: observed P(early-95 product) >= 0.55 is impossible for PBDT at any admissible parameter combination. M25 (85@2 vs 85@5): PGDT [0.51, 0.60]; PBDT [0.50, 0.55] (its ramp cancels its gate for the mid channel; the flat-share term straddles zero). L34 (60@3 vs 60@4): PGDT [0.52, 0.70]; PBDT [0.51, 0.62] -- observed >= 0.63 kills PBDT. PGDT requires a robustly nonzero transfer effect at ALL THREE validity levels; PBDT requires near-chance on the 95- and 85-level duels. Flatness at T16/M25 falsifies PGDT; any effect >= 0.55 at T16 or >= 0.57 at M25 falsifies PBDT's entire space.

AXIS 2 -- BLOCK-FLIP STAIRCASE (T16 -> BLK2 -> BLK3 + mirrors). BLK2 (A wins {95@1, 85@2} vs B wins {85@5, 95@6}, 60s tied): PGDT P in [0.51, 0.70]; PBDT [0.50, 0.55] -- raw-d boxes DISJOINT ([0.011, 0.044] vs [0.000, 0.011]). BLK3 (full 3v3 block flip): PGDT [0.53, 0.82]; PBDT [0.52, 0.66]. The within-subject staircase increments are the signature: PGDT's T16->BLK2 increment (adding the 85@2/85@5 pair) is robustly POSITIVE at every corner (raw [0.004, 0.019]) because the flat share x gate gives the early mid expert real weight; PBDT's increment STRADDLES ZERO ([-0.000, 0.008]) because its ramped flat share cancels the gate. Observed mirror-corrected P(BLK2) - P(T16) >= 0.04 is impossible for PBDT and required by PGDT's upper box; a flat or negative increment falsifies PGDT.

AXIS 3 -- EARLY-TRUSTED RESCUE AND POSITION TRANSFER (X1, X3 + mirrors). X1 (A wins {95@1, 85@5} vs B wins {85@2}): the two 85 endorsements cancel in the distrust kernel, so the 95@1 expert's flat-share weight decides. PGDT: d in [+0.002, +0.015] at every corner -> P(A) in [0.51, 0.60] (the early 95 RESCUES product A). PBDT: d in [-0.004, +0.007] -> straddles chance. X3 (A wins {85@5, 95@6} vs B wins {85@2}) is the position-transfer twin: the same 95 endorsement moved from position 1 to position 6. The transfer difference X1 - X3 equals b_tr*(1-delta) under PGDT -- robustly in [0.0075, 0.025] raw, P-difference >= 0.02 and up to 0.20 -- while under PBDT it equals flat'*(t0 - delta*t1) + leakage, confined to [0.0004, 0.0033] raw (P-difference <= 0.01). A reliable X1-vs-X3 difference >= 0.04 falsifies PBDT's entire space; a zero or negative difference falsifies PGDT.

AXIS 4 -- ANCHORS AND SCREENS. DINV (the product endorsed by all four 95/85 experts vs the product endorsed by the two 60 experts): BOTH theories confidently prefer the 60-product (PGDT d in [-0.39, -0.09]; PBDT [-0.24, -0.09]) -- verifies that subjects read and distrust-invert the stated validities, and excludes normative weighting, TTB (follows expert 1), and tallying (4 cues beat 2). VETO (95@1 vs 60@3): both prefer the 60-product ([-0.23, -0.07] / [-0.13, -0.06]) -- the single BEST expert's product loses to a mid-list distrusted expert's product. TOT: comprehension anchor (both ~0.95). R1 (95@6 vs 85@2) and X2 are agreement/fitting cells where both theories land on the same side (B, and straddle, respectively), constraining gate/ramp/beta jointly.

AGGREGATE SIGNATURE. Mirror-corrected P(A) regressed on PGDT reference evidence (g_j*[(1-tau)/n + tau*inv_j] at tau=0.87, eta=5, delta=0.3) versus PBDT reference evidence (g_j*[t_j*(1-tau)/n + tau*inv_j] at t0=0.30, kappa=1.0) loads with opposite profiles: the two evidence axes are anti-correlated on the X1/X3/BLK family by construction (position-1 trusted weight is maximal under the first, ~zero under the second). Single-number readouts: (i) family mean over {T16, M25, BLK2, X1} is hard-capped at 0.55 by PBDT (its maximum over all corners) while PGDT's central mass is 0.55-0.70; (ii) the transfer battery slope (effect size vs validity level: 95-duel, 85-duel, 60-duel) is strictly positive at every PGDT corner and confined to [~0, 0.05] under PBDT for the top two rungs; (iii) the X1-X3 transfer difference. Every cell type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n_CACHE = {}\n\n_TRIALS = [\n    ((1, 1, 1, 1, 1, 0), (0, 1, 1, 1, 1, 1)),\n    ((0, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),\n    ((1, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),\n    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 1)),\n    ((1, 1, 1, 0, 1, 1), (1, 1, 0, 1, 1, 1)),\n    ((1, 1, 0, 1, 1, 1), (1, 1, 1, 0, 1, 1)),\n    ((1, 1, 1, 1, 0, 0), (0, 0, 1, 1, 1, 1)),\n    ((0, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),\n    ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),\n    ((0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0)),\n    ((1, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 1)),\n    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 1)),\n    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 0, 0)),\n    ((1, 1, 1, 1, 0, 0), (1, 0, 1, 1, 1, 1)),\n    ((1, 1, 0, 0, 1, 1), (0, 0, 1, 1, 0, 0)),\n    ((0, 0, 1, 1, 0, 0), (1, 1, 0, 0, 1, 1)),\n    ((1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 1)),\n    ((0, 1, 1, 1, 1, 1), (1, 1, 0, 1, 1, 1)),\n    ((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0)),\n    ((0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1)),\n    ((1, 0, 1, 1, 1, 1), (1, 1, 1, 1, 1, 0)),\n    ((1, 1, 1, 1, 1, 0), (1, 0, 1, 1, 1, 1)),\n    ((0, 1, 1, 1, 0, 1), (1, 0, 1, 1, 1, 0)),\n    ((1, 0, 1, 1, 1, 0), (0, 1, 1, 1, 0, 1)),\n]\n\n_PG_GRID = [\n    (0.870, 5.00, 0.325, 3.15, 0.175, 4.25, 0.105),\n    (0.820, 4.00, 0.450, 2.80, 0.100, 6.00, 0.060),\n    (0.920, 6.00, 0.200, 3.50, 0.250, 2.50, 0.150),\n    (0.820, 6.00, 0.450, 3.50, 0.250, 6.00, 0.060),\n    (0.920, 4.00, 0.200, 2.80, 0.100, 2.50, 0.150),\n    (0.820, 4.00, 0.200, 3.50, 0.100, 6.00, 0.150),\n    (0.920, 6.00, 0.450, 2.80, 0.250, 2.50, 0.060),\n    (0.820, 6.00, 0.200, 2.80, 0.250, 2.50, 0.150),\n    (0.920, 4.00, 0.450, 3.50, 0.100, 6.00, 0.060),\n    (0.870, 4.00, 0.325, 3.15, 0.175, 2.50, 0.105),\n    (0.870, 6.00, 0.325, 3.15, 0.175, 6.00, 0.105),\n    (0.870, 5.00, 0.200, 2.80, 0.175, 4.25, 0.060),\n    (0.870, 5.00, 0.450, 3.50, 0.175, 4.25, 0.150),\n    (0.820, 5.00, 0.325, 3.15, 0.100, 4.25, 0.105),\n    (0.920, 5.00, 0.325, 3.15, 0.250, 4.25, 0.105),\n    (0.870, 5.00, 0.325, 2.80, 0.250, 4.25, 0.105),\n]\n\n_PB_GRID = [\n    (0.920, 7.25, 0.200, 2.75, 0.150, 0.300, 0.975, 1.15, 4.00, 0.125),\n    (0.910, 6.50, 0.250, 2.90, 0.250, 0.250, 1.000, 0.80, 4.40, 0.110),\n    (0.930, 8.00, 0.150, 2.60, 0.050, 0.350, 0.950, 1.50, 3.60, 0.140),\n    (0.910, 8.00, 0.150, 2.90, 0.250, 0.350, 1.000, 0.80, 3.60, 0.110),\n    (0.930, 6.50, 0.250, 2.60, 0.050, 0.250, 0.950, 1.50, 4.40, 0.140),\n    (0.910, 6.50, 0.150, 2.60, 0.250, 0.350, 1.000, 1.50, 4.40, 0.110),\n    (0.930, 8.00, 0.250, 2.90, 0.050, 0.250, 0.950, 0.80, 3.60, 0.140),\n    (0.920, 7.25, 0.200, 2.75, 0.150, 0.350, 0.950, 1.50, 4.40, 0.110),\n    (0.920, 7.25, 0.200, 2.75, 0.150, 0.250, 1.000, 0.80, 3.60, 0.140),\n    (0.910, 7.25, 0.200, 2.75, 0.050, 0.300, 0.975, 1.15, 4.00, 0.125),\n    (0.930, 7.25, 0.200, 2.75, 0.250, 0.300, 0.975, 1.15, 4.00, 0.125),\n    (0.920, 6.50, 0.200, 2.75, 0.150, 0.300, 0.975, 0.80, 4.40, 0.110),\n    (0.920, 8.00, 0.200, 2.75, 0.150, 0.300, 0.975, 1.50, 3.60, 0.140),\n    (0.920, 7.25, 0.250, 2.90, 0.150, 0.300, 0.975, 1.15, 4.00, 0.125),\n    (0.920, 7.25, 0.150, 2.60, 0.150, 0.300, 0.975, 1.15, 4.00, 0.125),\n    (0.915, 7.00, 0.225, 2.85, 0.200, 0.275, 0.985, 0.95, 4.20, 0.120),\n]\n\n\ndef _pgdt_p(a, b, par):\n    tau, eta, delta, gamma, c, beta, eps = par\n    v = np.array([0.95, 0.85, 0.60, 0.60, 0.85, 0.95])\n    n = 6\n    err = 1.0 - v\n    centered = np.maximum(err - c * float(err.min()), 0.0)\n    inv = np.power(centered, gamma)\n    Zv = float(inv.sum())\n    if Zv > 1e-12:\n        w_inv = inv / Zv\n    else:\n        w_inv = np.full(n, 1.0 / n)\n    pos = np.arange(n, dtype=float)\n    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)\n    bracket = (1.0 - tau) / n + tau * w_inv\n    raw = g * bracket\n    w = raw / float(raw.sum())\n    d = float(np.dot(w, np.sign(a - b)))\n    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))\n    return (1.0 - eps) * p + eps * 0.5\n\n\ndef _pbdt_p(a, b, par):\n    tau, eta, delta, gamma, c, t0, t1, kappa, beta, eps = par\n    v = np.array([0.95, 0.85, 0.60, 0.60, 0.85, 0.95])\n    n = 6\n    err = 1.0 - v\n    centered = np.maximum(err - c * float(err.min()), 0.0)\n    inv = np.power(centered, gamma)\n    Zv = float(inv.sum())\n    if Zv > 1e-12:\n        w_inv = inv / Zv\n    else:\n        w_inv = np.full(n, 1.0 / n)\n    pos = np.arange(n, dtype=float)\n    t = t0 + (t1 - t0) * np.power(pos / (n - 1.0), kappa)\n    g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)\n    bracket = t * (1.0 - tau) / n + tau * w_inv\n    raw = g * bracket\n    w = raw / float(raw.sum())\n    d = float(np.dot(w, np.sign(a - b)))\n    p = 0.5 * (1.0 + np.tanh(0.5 * beta * d))\n    return (1.0 - eps) * p + eps * 0.5\n\n\ndef _weights():\n    if 'wmap' in _CACHE:\n        return _CACHE['wmap']\n    A = np.array([t[0] for t in _TRIALS], dtype=float)\n    B = np.array([t[1] for t in _TRIALS], dtype=float)\n    nt = len(_TRIALS)\n    m_pg = np.zeros(nt)\n    m_pb = np.zeros(nt)\n    for par in _PG_GRID:\n        for i in range(nt):\n            m_pg[i] += (2.0 * _pgdt_p(A[i], B[i], par) - 1.0)\n    m_pg /= float(len(_PG_GRID))\n    for par in _PB_GRID:\n        for i in range(nt):\n            m_pb[i] += (2.0 * _pbdt_p(A[i], B[i], par) - 1.0)\n    m_pb /= float(len(_PB_GRID))\n    dm = m_pg - m_pb\n    s2 = 0.5 * ((1.0 - m_pg ** 2) + (1.0 - m_pb ** 2))\n    s2 = np.maximum(s2, 0.10)\n    w = np.where(np.abs(dm) >= 0.022, dm / s2, 0.0)\n    if not np.any(np.abs(w) > 0.0):\n        w = dm / s2\n    wmap = {}\n    for i in range(nt):\n        wmap[_TRIALS[i]] = float(w[i])\n    _CACHE['wmap'] = wmap\n    return wmap\n\n\ndef _to_tuple(x):\n    if isinstance(x, (list, tuple, np.ndarray)):\n        return tuple(int(round(float(v))) for v in x)\n    s = str(x).replace(\"'\", \"\").replace('\"', '')\n    s = s.strip().strip('[](){} ').replace(',', ' ')\n    return tuple(int(round(float(p))) for p in s.split())\n\n\ndef metric(data: pd.DataFrame) -> float:\n    wmap = _weights()\n    ra = data['option_a_ratings'].apply(_to_tuple)\n    rb = data['option_b_ratings'].apply(_to_tuple)\n    wv = np.array([wmap.get((a, b), 0.0) for a, b in zip(ra, rb)], dtype=float)\n    z = 1.0 - 2.0 * data['response'].astype(float).values\n    num = float(np.dot(wv, z))\n    den = float(np.sum(np.abs(wv)))\n    if den <= 0.0:\n        return 0.0\n    return num / den",
  "rationale": "STRUCTURE OF THE DISAGREEMENT. PGDT and PBDT share the distrust-inversion kernel and a floored convex primacy gate; they differ almost exclusively in the raw weight carried by EARLY high-validity experts. PGDT's bracket (1-tau)/n + tau*inv_j with tau in [0.82, 0.92] gives every cue a position-independent flat share of ~1.3-3.0% (n=6), so the first 95% expert's normalized weight is ~0.03-0.06. PBDT multiplies the flat share by a trust ramp starting at t0 in [0.25, 0.35], crushing the early flat share to ~0.4-0.7% (normalized early-95 weight ~0.02-0.035) while leaving the distrust channel and late weights essentially intact. Every other difference (gate eta/delta ranges) is second-order on this palindromic layout because the distrust kernel cancels within each validity-matched duel.\n\nMETRIC. Rather than a single-family proportion (noisy, since each family has only 8 trials per subject), the metric is the approximate signal-to-noise-OPTIMAL linear contrast between the two theories' predicted choice probabilities across all 96 trials. I embed faithful replicas of both predict functions, evaluate P(choose A) for each of the 24 unique design pairs over a 16-point grid spanning each theory's parameter box (centre + corners + axis mixes), and assign each trial the box-averaged difference in predicted choice probability, divided by the binomial response variance 1-m^2 (inverse-variance weighting). Trials where the two theories agree by construction -- the comprehension anchors DINV (15/16), VETO (17/18), TOT (19/20), the agreement cell R1 (21/22), and the unstable cells M25 (3/4) and X3 (13/14), whose box-averaged separation is near zero and sign-inconsistent -- receive zero weight, so they contribute no variance. The surviving weighted families are the block-flip staircase BLK3 (9/10) and BLK2 (7/8), the 95-validity position duel T16 (1/2), the early-trusted rescue X1 (11/12), the 60-validity gate contrast L34 (5/6), and X2 (23/24). The output is the weighted mean of signed choices (A=+1, B=-1), normalized by total absolute weight: a bounded 'differential preference' score in [-1, 1].\n\nINTERPRETATION. Positive values = early trusted experts demonstrably move choices (PGDT: the flat share survives the gate, so the early-95 expert rescues products and the early block wins the staircase); values near zero = the early flat share has been annihilated by the trust ramp (PBDT). Because the weights are box-averaged, the metric is insensitive to where in its parameter box each theory is simulated, and subject-level variation in beta/epsilon (which rescales all near-chance trials roughly equally) largely cancels in the weighted sign pattern, keeping between-subject variance near the binomial floor. The metric is well-defined on any row subset, including single-subject slices, so the pooled point estimate equals the mean of the per-subject values. Expected ordering: PGDT-simulated data ~ +0.08 to +0.11; PBDT-simulated data ~ +0.03 to +0.06, with the gap carried by the disjoint early-trusted weight boxes."
}
```

## Usage

```json
{
  "input_tokens": 9197,
  "output_tokens": 97485,
  "reasoning_tokens": 92943,
  "provider": "Sail Research"
}
```
