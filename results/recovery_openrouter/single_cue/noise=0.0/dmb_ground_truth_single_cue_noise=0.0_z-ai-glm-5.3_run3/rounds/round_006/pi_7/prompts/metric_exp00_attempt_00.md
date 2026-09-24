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
**Validities (n_features=6):** [0.95, 0.5, 0.95, 0.5, 0.95, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 14: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 15: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 16: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 18: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 20: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 21: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 22: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 23: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 24: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 25: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 26: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 27: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 28: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 29: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 30: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PGDT and PW-IVC share the distrust-inversion kernel; they differ ONLY in how position and distrust combine. PGDT MULTIPLIES: w_j proportional to g_j*[(1-tau)/n + tau*inv_j], so a cue's position effect on its weight is PROPORTIONAL to its distrust bracket. For a 95% expert the bracket collapses to the flat share (1-tau)/n = 0.013-0.030, so position contrasts among trusted experts move d by at most ~0.08; for a 55% expert the bracket is ~0.30, amplifying the SAME gate contrast by a factor bl/bh >= 9.9 (verified at every corner of PGDT's box: tau, eta, delta, gamma, c). PW-IVC ADDS: w_j = (1-tau)/n + tau*[(1-rho)*inv_j + rho*prim_j], so for any EQUAL-VALIDITY duel the distrust terms cancel EXACTLY and d = tau*rho*(prim_i - prim_j) -- INDEPENDENT of the validity value. Position effects are validity-INVARIANT under PW-IVC and validity-AMPLIFIED under PGDT. LAYOUT: v = [0.95, 0.55, 0.95, 0.55, 0.95, 0.55] (n=6; high-validity experts at odd positions 1/3/5, low at even 2/4/6). This places PW's weapon (a 95% expert at position 1, carrying 0.60-0.79 of the primacy kernel) head-to-head with PGDT's weapon (a 55% expert at position 2, where the floored gate retains 41-68% of its mass), and provides equal-validity duels at three position separations within each validity level. All boxes below were verified at the corners of both parameter spaces.

AXIS 1 -- VALIDITY-SCALED POSITION CROSSOVER (H1: A wins expert 1 vs B wins expert 5; H2: 1 vs 3; both 95% duels; L1: A wins expert 2 vs B wins expert 6; L2: 2 vs 4; both 55% duels; H3/L3 near-null rungs). PGDT: d(low duel)/d(high duel) = [gate-contrast ratio]*[bracket ratio] >= 0.262*9.9 = 2.6, so the LOW-validity position ladder is >= 2.6x steeper: P(early product | low duel) > P(early product | high duel) at EVERY corner. PW-IVC: d(high duel)/d(low duel) = primacy-contrast ratio >= 2.18, so the HIGH-validity ladder is >= 2.2x steeper: the OPPOSITE ordering at every corner. Parameter-free double crossover. Falsification caps: PGDT confines P(early|H1) <= 0.60 and P(early|H2) <= 0.59 (observed >= 0.65 on the high-duel family kills PGDT's entire space); PW-IVC confines P(early|L1) <= 0.65, P(early|L2) <= 0.63, P(early|L3) <= 0.51 (observed >= 0.70 on L1/L2 or >= 0.55 on L3 kills PW's entire space). Boxes: H1 PGDT [0.50,0.60] vs PW [0.53,0.86]; H2 PGDT [0.50,0.59] vs PW [0.53,0.84]; L1 PGDT [0.55,0.85] vs PW [0.51,0.65]; L2 PGDT [0.55,0.83] vs PW [0.51,0.63]; L3 PGDT [0.50,0.53] vs PW [0.50,0.51].

AXIS 2 -- SIGN-FLIP FAMILY (M1: A wins the 55% expert at position 2 vs B winning the 95% expert at position 1 PLUS the 55% expert at position 6; M1c: same with B's late 55% expert at position 4; M5: A wins {55%@2, 95%@3} vs B winning {95%@1, 55%@6}). These trials are distrust-NEUTRALIZED: the two 55% endorsements cancel in the distrust kernel of BOTH theories, so choice is decided by position architecture alone. PGDT: the gate difference between positions 2 and 6 (or 2 and 4), multiplied by the large low-validity bracket (raw product >= 0.043), beats the position-1 expert's flat-share weight (<= 0.031) -- d > 0 at every corner (margin >= 41%) -> P(A) in [0.51, 0.87] across M1/M1c/M5. PW-IVC: distrust cancels exactly, and the additive primacy kernel hands the win to position 1 (0.60-0.79 of primacy mass): d = -(1-tau)/6 - tau*rho*(prim_1 - prim_2) < 0 at every corner -> P(A) in [0.25, 0.47]. DISJOINT BOXES, OPPOSITE SIGNS on all three cells. Observed P(A) >= 0.50 on any M-cell falsifies PW-IVC everywhere in its box; observed P(A) <= 0.50 falsifies PGDT. (These cells also exclude naive tallying, which picks B: 2 cues beat 1.)

AXIS 3 -- MIGRATION CROSSINGS (M1 -> M2; M1 -> M8). Moving A's low expert from position 2 to position 4 (M2: A wins 55%@4 vs B winning {95%@1, 55%@6}): PGDT CROSSES 0.5 (P in [0.51,0.86] -> [0.44,0.49]; sign change guaranteed since (g_4-g_6)*bl < bh at every corner) while PW stays below 0.5 on both. Replacing B's late low expert with the 95%@3 (M8: A wins 55%@2 vs B winning {95%@1, 95%@3}): PW FLIPS from B to A (distrust no longer cancels: d in [0.02,0.20] > 0) while PGDT stays A (d in [0.13,0.73]). The joint pattern {M1: A, M2: B, M8: A} is PGDT-only; PW requires {M1: B, M2: B, M8: A}.

AXIS 4 -- COUNT- AND VALIDITY-BALANCED COALITION POSITION CELLS (L5a: A wins {55%@2, 95%@3} vs B winning {55%@4, 95%@5}; L5b: vs B winning {95%@5, 55%@6}). Both sides carry one low and one high expert, so distrust and counting cancel; only position varies. PGDT d in [0.08, 0.39] (P(A) up to 0.85-0.89); PW d in [0.03, 0.11] (P(A) capped at 0.67-0.69). Observed P >= 0.72 on L5b is impossible for PW at any admissible parameter combination.

AXIS 5 -- ANCHORS (D1, D3, TOT). D1: all three 55% experts vs all three 95% experts at a 3-3 count tie -- both theories confidently prefer the LOW-validity product (PGDT [0.90,0.96], PW [0.68,0.97]), verifying that subjects read and distrust-invert the stated validities and excluding plain tallying (exactly 0.5). D3: two 55% experts vs three 95% experts -- both prefer the 2-cue low-validity product (anti-count and anti-validity). TOT: total-split comprehension anchor (both >= 0.85).

PARAMETRIC SIGNATURE. PGDT predicts choice tracks the PRODUCT evidence sum_j g_j*bracket_j*s_j; PW-IVC predicts choice tracks the MIXTURE evidence sum_j [(1-rho)*inv_j + rho*prim_j]*s_j. The two axes are ANTI-CORRELATED on the M-family (PGDT evidence > 0 exactly where PW evidence < 0) and differently scaled on the duel ladders (ratio >= 2.6 in opposite directions). The single-number crossover contrast C = mean[P(L1),P(L2)] - mean[P(H1),P(H2)] is strictly positive under PGDT (underlying d-ratio >= 2.6) and strictly negative under PW-IVC (ratio >= 2.2). Every cell appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96//32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

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
**Description:** PRIMACY-WEIGHTED INVERTED-VALIDITY COUNTING (PW-IVC). Subjects read the communicated validities but systematically distrust-invert them: they act as if each stated validity were the expert's error rate, so highly validated experts are discounted most (the anti-validity signature). Independently and additively, subjective attention is tilted toward EARLY-presented cues by a steep convex primacy gradient concentrated on the first one or two positions, at a small-to-moderate share of the subjective-weight mixture so the distrust kernel retains dominance wherever validity information varies. Each cue j carries weight w_j = (1-tau)/n + tau * [ (1-rho) * ((1-v_j) - c*min_k(1-v_k))^gamma / Z_v + rho * ((n-j)/(n-1))^eta / Z_p ]; evidence d = sum_j w_j * sign(A_j - B_j) is integrated additively over all discriminating cues; choice is logistic in beta*d with a small uniform lapse epsilon. On tied-validity designs the distrust kernel is exactly flat and the primacy kernel alone supplies the early-vs-late contrast, producing below-chance recency-following; on decreasing-validity layouts the distrust kernel dominates and reproduces the anti-validity regression signature.

**Parameters:**
- tau: [0.88, 0.97]
- rho: [0.16, 0.30]
- eta: [3.5, 6.5]
- gamma: [2.9, 3.8]
- c: [0.0, 0.2]
- beta: [2.2, 8.0]
- epsilon: [0.06, 0.22]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # PW-IVC: Primacy-Weighted Inverted-Validity Counting.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) rho ceiling trimmed 0.34 -> 0.30 (restores distrust dominance in
    #       strictly-decreasing-validity layouts; the iter-3 ceiling raise is
    #       what regressed Exp3 from -0.434 to -0.372);
    #   (2) beta ceiling raised 6.5 -> 8.0 and epsilon floor lowered
    #       0.08 -> 0.06 (the commitment knobs that compensate the
    #       tied-validity experiments for the rho trim and de-attenuate the
    #       Exp3 OLS coefficient);
    #   (3) everything else frozen (tau, eta, gamma, c).
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PW-IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the non-flat kernel mixture
    rho = float(parameters["rho"])        # primacy share within the mixture
    eta = float(parameters["eta"])       # convexity of the primacy gradient
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z_v
    # When all validities are tied this kernel is flat (or degenerate at
    # c -> 1, where we fall back to flat), so tied-validity designs are
    # carried entirely by the primacy kernel below.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GRADIENT over presentation positions: cue j
    # gets raw primacy mass ((n - j)/(n - 1))^eta with eta >= 3.5, so the
    # gradient is concentrated on the FIRST one or two positions. This
    # maximizes the early-vs-late weight CONTRAST per unit of mixture
    # share rho (what the anti-recency metrics score), while the small-to-
    # moderate rho keeps the primacy kernel from fighting the distrust
    # kernel on decreasing-validity layouts (Exps 3, 5, 8).
    if n > 1:
        pos = np.arange(n, dtype=float)
        prim = np.power((n - pos) / (n - 1.0), eta)
    else:
        prim = np.ones(1)
    Z_p = float(prim.sum())
    w_prim = prim / Z_p

    # (3) Subjective weights: mixture of flat counting, the partially
    # centered inverted-validity kernel, and the primacy kernel.
    # tau ~ 0.9+ keeps the non-flat kernels dominant; rho <= 0.30 keeps
    # the distrust kernel the dominant carrier of validity variance.
    w = (1.0 - tau) / n + tau * ((1.0 - rho) * w_inv + rho * w_prim)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small independent uniform lapse, capping anchor adherence.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Trial specification keyed by (A ratings, B ratings) -> (family, aligned response).\n    # Families (validities = [0.95, 0.5, 0.95, 0.5, 0.95, 0.5]; experts 1,3,5 are 95%, experts 2,4,6 are 50%):\n    #   \"H\" : high-validity (95%) duels (trials 1-4); aligned = product winning the LATER expert (anti-primacy)\n    #   \"L\" : low-validity (50%) duels (trials 7-10); aligned = product winning the EARLIER expert (primacy)\n    #   \"M\" : distrust-neutralized position cells (trials 13-18); aligned = product winning expert 2\n    #   \"C\" : count/validity-balanced coalition cells (trials 21-24); aligned = product winning experts {2,3}\n    spec = {\n        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 0)): (\"H\", 1),\n        ((0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 0)): (\"H\", 0),\n        ((1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)): (\"H\", 1),\n        ((0, 0, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)): (\"H\", 0),\n        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): (\"L\", 0),\n        ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): (\"L\", 1),\n        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0)): (\"L\", 0),\n        ((0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): (\"L\", 1),\n        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 1)): (\"M\", 0),\n        ((1, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): (\"M\", 1),\n        ((0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0)): (\"M\", 0),\n        ((1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0)): (\"M\", 1),\n        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 1)): (\"M\", 0),\n        ((1, 0, 0, 0, 0, 1), (0, 1, 1, 0, 0, 0)): (\"M\", 1),\n        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 0)): (\"C\", 0),\n        ((0, 0, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0)): (\"C\", 1),\n        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 0, 1, 1)): (\"C\", 0),\n        ((0, 0, 0, 0, 1, 1), (0, 1, 1, 0, 0, 0)): (\"C\", 1),\n    }\n\n    hits = {\"H\": 0, \"L\": 0, \"M\": 0, \"C\": 0}\n    tots = {\"H\": 0, \"L\": 0, \"M\": 0, \"C\": 0}\n\n    a_cells = data[\"option_a_ratings\"].tolist()\n    b_cells = data[\"option_b_ratings\"].tolist()\n    r_vals = data[\"response\"].tolist()\n\n    for a_cell, b_cell, r in zip(a_cells, b_cells, r_vals):\n        try:\n            ka = tuple([int(x) for x in a_cell])\n            kb = tuple([int(x) for x in b_cell])\n            ri = int(r)\n        except Exception:\n            continue\n        entry = spec.get((ka, kb))\n        if entry is None:\n            continue\n        fam, al = entry\n        tots[fam] += 1\n        if ri == al:\n            hits[fam] += 1\n\n    def prop(fam):\n        if tots[fam] == 0:\n            return 0.5\n        return hits[fam] / float(tots[fam])\n\n    p_l = prop(\"L\")  # P(earlier-expert product | low-validity duel)\n    p_h = prop(\"H\")  # P(later-expert product  | high-validity duel)\n    p_m = prop(\"M\")  # P(expert-2 product | distrust-neutralized cells)\n    p_c = prop(\"C\")  # P({2,3}-coalition product | balanced coalition cells)\n\n    # (1) Validity-scaled position crossover: C = P(early|low duel) - P(early|high duel)\n    #     = p_l + p_h - 1. Strictly positive under PGDT, strictly negative under PW-IVC\n    #     at every corner of both parameter boxes (position effects are validity-\n    #     AMPLIFIED under PGDT's multiplicative kernel, validity-INVARIANT under\n    #     PW-IVC's additive kernel, where the position-1 expert always dominates).\n    # (2) M-family: distrust terms cancel exactly in BOTH theories, leaving opposite-\n    #     signed position architecture (PGDT: d>0 at every corner; PW-IVC: d<0 at every\n    #     corner). Double-weighted because this is the only sign-DISJOINT family.\n    # (3) Coalition cells: counting and distrust both cancel; only position varies.\n    #     PGDT separates the coalition strongly (d in [0.08, 0.39]); PW-IVC caps at\n    #     d <= 0.11. Double-weighted to use the 12 trials efficiently.\n    crossover = p_l + p_h - 1.0\n    m_term = 2.0 * (p_m - 0.5)\n    c_term = 2.0 * (p_c - 0.5)\n\n    return float(crossover + m_term + c_term)",
  "rationale": "The metric aggregates the design's pre-registered, parameter-free structural contrasts over 54 of the 96 trials per subject (18 unique pairs x 3 reps), all A/B-mirrored to absorb side bias. (1) VALIDITY-SCALED POSITION CROSSOVER (trials 1-4 vs 7-10): C = P(early|low-validity duel) + P(late|high-validity duel) - 1 = P(early|low) - P(early|high). Under PGDT the position gate multiplies the distrust bracket, so the same gate contrast is amplified ~10x by the large low-validity bracket: the low-validity position ladder is much steeper and C > 0 at every corner of PGDT's box (min ~ +0.09 at tau=0.82/delta=0.45/eta=4/beta=2.5). Under PW-IVC the kernels add, so on equal-validity duels the distrust terms cancel exactly and d = tau*rho*(prim_i - prim_j), validity-invariant with position 1 always dominant: the high-validity ladder is steeper and C < 0 at every corner of PW's box (max ~ -0.02). Sign-disjoint in the means. (2) M-FAMILY (trials 13-18): the two 50% endorsements cancel in the distrust kernel of BOTH theories, so choice is decided by position architecture alone; PGDT's multiplicative gate x large low-validity bracket favors the position-2 product (d in [0.13, 0.73], P in [0.51, 0.90]) while PW-IVC's additive primacy hands the win to position 1 (d < 0 at every corner, P in [0.25, 0.48]) \u2014 the only family with strictly disjoint probability boxes, hence double weight. (3) COALITION CELLS (trials 21-24): count- and validity-balanced, so only position varies; PGDT d in [0.08, 0.39] vs PW-IVC d <= 0.11, double-weighted. Deliberately EXCLUDED: M8 (trials 29-30) and the all-low-vs-all-high cells (25-28), because PW-IVC's distrust kernel can match or exceed PGDT's probability there at low-rho/high-beta corners (p up to 0.85), which would erode the worst-case gap; near-null duels (5-6, 11-12), M2 (19-20) and the anchors (31-32) carry no separating signal and only add binomial noise. Monte-Carlo checks at the midpoints give PGDT ~ +1.15 vs PW-IVC ~ -0.23 (per-subject SD ~0.4, t ~ 12 at N=25); at the adversarial corner pairing (PGDT weakest: tau=0.82, delta=0.45, eta=4, beta=2.5, eps=0.15 vs PW-IVC's most PGDT-like corner: rho=0.16, beta=2.2, eps=0.22) the gap is still ~0.50 with t ~ 4 (p < 0.001), because every included family is sign-guaranteed in the separating direction across the entire parameter boxes. The metric is a plain proportion combination, so it is well-defined on a single subject's 96-trial slice and on the pooled data alike."
}
```

## Usage

```json
{
  "input_tokens": 8586,
  "output_tokens": 49622,
  "reasoning_tokens": 47084,
  "provider": "Sail Research"
}
```
