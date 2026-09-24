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
**Validities (n_features=6):** [0.95, 0.5, 0.5, 0.95, 0.5, 0.5]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 13: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 16: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 17: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 21: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 24: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 25: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 26: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 27: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 28: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 29: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 30: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PW-IVC and PGDT share the distrust-inversion kernel; they differ in how primacy and distrust COMBINE. PW-IVC ADDS: w_j = (1-tau)/n + tau*[(1-rho)*inv_j + rho*prim_j], so the primacy bonus for an early cue is validity-INDEPENDENT -- an early TRUSTED (95%) expert receives the full rho*prim_1 boost on top of its (tiny) distrust weight. PGDT MULTIPLIES: w_j proportional to g_j*[(1-tau)/n + tau*inv_j], so the gate boost scales with the bracket: for a 95% expert the bracket collapses to the flat share (1-tau)/n = 0.013-0.030, and the maximal gate (g_1 = 1) multiplies almost nothing; for a 55% expert the bracket is ~0.22-0.26 and the gate amplifies it. Consequence, verified at all 8 corners of each parameter box on the layout below: the position-1 95% expert carries PW weight 0.10-0.24 but PGDT weight only 0.03-0.11 (a 2-4x relative rescue), while the position-2 55% expert carries PGDT weight 0.29-0.41 vs PW 0.22-0.25. Conversely, the position gradient among 55% experts is 3-4x steeper under PGDT (gate x big bracket) than under PW, and the position gradient among 95% experts exists ONLY under PW (PGDT's is pinned at the flat share). Prior design [12] attacked this with alternating validities and position-duel ladders plus M-migrations; this design uses a different layout (2 trusted experts at positions 1 and 4, FOUR distrusted experts at 2, 3, 5, 6) and centers on a NOVEL axis: a COUNT-CROSSOVER / VOTE-EQUIVALENCE LADDER measuring how many 55% endorsements one early 95% expert is worth, plus a trusted-block assay and gradient-ownership duels.

LAYOUT: v = [0.95, 0.55, 0.55, 0.95, 0.55, 0.55] (n=6). The kernel ratio (0.44/0.05)^gamma >= 600 concentrates ~99.9% of BOTH distrust kernels on the four 55% experts (each ~0.25 of inv mass), so the distrust signal is essentially 'count of 55% endorsements per side' and is held nearly constant across cells; position architecture and the trusted-expert weight are what vary. All boxes below are P(A) for the listed A-set, verified at the corners of both parameter spaces (PW tau/rho/eta corners; PGDT tau/delta/eta corners, with beta/epsilon at their worst-case ends).

AXIS 1 -- VOTE-EQUIVALENCE LADDER (L0, L1, L2, L3; the headline). A wins the 95% expert at position 1 plus j late 55% experts (position 6, then 5, then 3); B wins the single 55% expert at position 2. PW: the early trusted expert is worth 0.6-1.1 late 55% experts, so the ladder crosses chance at rung j*=1: L1 (A={1,6} vs B={2}) has d in [+0.06, +0.19] at EVERY PW corner -> P(A) in [0.52, 0.79]. PGDT: the early trusted expert is worth only 0.1-0.3 of one 55% expert, so the ladder does not cross until rung 2-3: L1 has d in [-0.21, -0.01] at EVERY PGDT corner -> P(A) in [0.24, 0.50]. DISJOINT BOXES on L1: observed P(A) >= 0.55 is impossible for PGDT at any admissible parameter combination; observed <= 0.48 is impossible for PW. The measured crossing rung j* separates the theories by one to two rungs (PW j*=1; PGDT j* in {2,3}). L2: PW [0.71, 0.94] vs PGDT [0.47, 0.75]; L3: both A (agreement high end). L1e (helper = the position-3 55% expert) replicates the rung with an early helper: PW d in [+0.07, +0.19] (all positive) vs PGDT d in [-0.13, +0.01] (nearly all negative).

AXIS 2 -- TRUSTED-BLOCK ASSAY (TB, TB+, F3). TB: A wins BOTH 95% experts (positions 1 and 4) vs B wins the single position-2 55% expert. PGDT: d in [-0.36, -0.19] -> P(55%-product) in [0.60, 0.87]; the two BEST experts lose to one distrusted expert. PW: d in [-0.15, +0.03] -> P(A) in [0.28, 0.53], i.e., chance. Observed P(95%-product) >= 0.50 falsifies PGDT's entire space. TB+ (both 95% + the position-3 55% vs the position-2 55%): PW [0.54, 0.83] vs PGDT [0.36, 0.54]. F3 (both 95% + the late 55% at position 6 vs the two early 55% experts at 2 and 3): PGDT P(A) in [0.10, 0.28] -- the two early lows dominate a 3-cue coalition containing both best experts; PW in [0.27, 0.53]. TB also excludes naive tallying (2 cues beat 1, yet both theories pick the 1-cue 55% product).

AXIS 3 -- GRADIENT OWNERSHIP (HD vs LDf/LDa/LDm/LB). HD: duel between the two 95% experts (position 1 vs 4). PW: d in [+0.08, +0.23] -> P(early) in [0.53, 0.85]; PGDT: d in [+0.01, +0.08] -> P(early) in [0.51, 0.62]. LDf (55%@2 vs 55%@6): PW d in [+0.01, +0.08] -> P(early) in [0.50, 0.58]; PGDT d in [+0.07, +0.26] -> [0.54, 0.81]. LDa (55%@2 vs 55%@3): PW [0.49, 0.57] vs PGDT [0.52, 0.72]. LB (early-low pair {2,3} vs late-low pair {5,6}): PW [0.52, 0.68] vs PGDT [0.52, 0.86]. The aggregate contrast C = P(early | HD) - P(early | LDf) is POSITIVE at every PW corner (HD effect 1.5-4.5x the low-duel effect) and NEGATIVE at every PGDT corner (low-duel effect 1.9-7.7x the HD effect) -- a parameter-free sign flip. Observed P(early|HD) >= 0.68 kills PGDT; observed P(early|LDf) >= 0.65 or P(early|LB) >= 0.75 kills PW -- the design is genuinely two-sided, with the low-validity gradient cells as PGDT's home turf.

AXIS 4 -- DISTRUST-NEUTRAL SIGN-FLIP CELLS (L1, FL). On L1 and FL the 55% endorsements cancel exactly in both theories' distrust kernels, so choice is decided by position architecture alone. PW's additive primacy hands the win to the position-1 expert's product; PGDT's multiplicative gate x bracket hands it to the position-2 expert's product. FL (A = {95%@1, 55%@5} vs B = {55%@2, 95%@4}): PW d in [+0.05, +0.18] (all corners positive) -> P(A) in [0.52, 0.78]; PGDT d in [-0.22, -0.04] (all corners negative) -> P(A) in [0.22, 0.48]. DISJOINT, opposite signs.

AXIS 5 -- ANCHORS. DINV (four 55% experts vs the two 95% experts): both theories confidently prefer the 55% product (PW [0.72, 0.97], PGDT [0.85, 0.99]) -- verifies that subjects read and distrust-invert the communicated validities. TOT (total split): comprehension anchor, both ~1. LDm (55%@3 vs 55%@5) is a near-null rung assaying generic side bias.

PARAMETRIC SIGNATURE. Regressing P(A) on primacy-only evidence sum(prim_j * s_j) versus product evidence sum(g_j * bracket_j * s_j) yields opposite-loading profiles: the two axes are anti-correlated on the L1/FL/F3 family by construction. The single-number readouts: (i) the ladder crossing rung j* (PW: 1; PGDT: 2-3); (ii) the gradient-ownership contrast C (PW: positive everywhere; PGDT: negative everywhere); (iii) the trusted-block veto (PGDT uniquely predicts P(55%@2-product) >= 0.60 against both 95% experts). Every cell appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Critical \"primacy-rescue\" trial family: exactly one option carries the\n    # position-1 trusted (0.95) expert, the opposing option carries only\n    # early low-validity (0.50) experts. Six f1-side patterns (option that\n    # holds the position-1 trusted expert) x two counterpart patterns.\n    P = {\n        (1, 0, 0, 0, 0, 1),  # {95%@1, 50%@6}          (L1 rung)\n        (1, 0, 0, 0, 1, 1),  # {95%@1, 50%@5, 50%@6}   (L2 rung)\n        (1, 0, 1, 0, 1, 1),  # {95%@1, 50%@3,5,6}     (L3 rung)\n        (1, 0, 1, 0, 0, 0),  # {95%@1, 50%@3}          (L1e early helper)\n        (1, 0, 1, 1, 0, 0),  # {95%@1, 50%@3, 95%@4}   (TB+ cell)\n        (1, 0, 0, 0, 1, 0),  # {95%@1, 50%@5}          (FL cell)\n    }\n    Q = {\n        (0, 1, 0, 0, 0, 0),  # {50%@2}\n        (0, 1, 0, 1, 0, 0),  # {50%@2, 95%@4}\n    }\n\n    def to_key(x):\n        return tuple(int(v) for v in x)\n\n    hits = []\n    a_col = data[\"option_a_ratings\"]\n    b_col = data[\"option_b_ratings\"]\n    r_col = data[\"response\"]\n    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):\n        a = to_key(a_raw)\n        b = to_key(b_raw)\n        r = int(r_raw)\n        if a in P and b in Q:\n            # trusted-early product is option A\n            hits.append(1.0 if r == 0 else 0.0)\n        elif b in P and a in Q:\n            # trusted-early product is option B\n            hits.append(1.0 if r == 1 else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "METRIC: 'Primacy-rescue rate' \u2014 the fraction of choices falling on the product that contains the position-1 trusted (0.95) expert, computed over the 12 critical trial types (trials 3,4,5,6,7,8,9,10,13,14,23,24; 36 trials per subject, since every subject sees each of the 32 pairs 3 times in randomized order). In every one of these pairs the opposing product is built only from early low-validity experts ({50%@2} or {50%@2, 95%@4}), so the distrust kernel (which both theories share, and which concentrates ~99.9% of its mass on the four 50% experts) either cancels or favors the opponent; only the position-architecture difference between the theories can rescue the trusted-early product.\n\nWHY IT SEPARATES THE THEORIES: PW-IVC ADDS a validity-independent primacy bonus to the position-1 expert (rho*prim_1 on top of its near-zero distrust weight), giving the early trusted expert weight 0.10-0.24 \u2014 enough to beat or match one early 50% expert. PGDT MULTIPLIES the gate by the distrust bracket; for a 95% expert the bracket collapses to the flat share (1-tau)/6 \u2248 0.013-0.030, so even the maximal gate multiplies almost nothing, and the position-2 50% expert (weight 0.29-0.41) dominates. Verified at parameter-box corners and midpoints: PW-IVC predicts rescue rates per cell of roughly 0.52-0.82 (composite \u2248 0.55-0.71, mid \u2248 0.64); PGDT predicts roughly 0.14-0.57 with the composite \u2248 0.30-0.50 (mid \u2248 0.44). The strongest single cells (L1: {95%@1, 50%@6} vs {50%@2}; FL: {95%@1, 50%@5} vs {50%@2, 95%@4}) have DISJOINT predicted boxes \u2014 PW \u2265 0.51 everywhere, PGDT \u2264 0.50 everywhere.\n\nWHY IT PASSES THE ACCEPTANCE TEST: the contrast is a simple binomial proportion over 36 trials per subject, so per-subject sampling noise is small (sd \u2248 sqrt(0.25/36) \u2248 0.08) on top of modest parameter heterogeneity (corner-to-corner spread \u2248 0.17 for PW, 0.20 for PGDT). Expected between-subject variance \u2248 0.008-0.010 for both theories against a mean gap of \u2248 0.20, giving Welch t \u2248 7-9 at N=25 (p << 0.001). The metric is a plain proportion, so it is well-defined both on the pooled DataFrame and on any single subject's slice (every subject completes all 96 trials, hence all 36 critical trials). A/B-mirrored pairs are included symmetrically, so side bias cancels. Anchor trials (all-ones vs all-zeros, the four-50% vs two-95% cell) and PGDT-favoring gradient cells (LDf/LDa/LB) are deliberately excluded so the metric isolates the one structural axis \u2014 additive vs multiplicative combination of primacy and distrust \u2014 on which the two theories make opposite-signed predictions."
}
```

## Usage

```json
{
  "input_tokens": 8777,
  "output_tokens": 25955,
  "reasoning_tokens": 24554,
  "provider": "Sail Research"
}
```
