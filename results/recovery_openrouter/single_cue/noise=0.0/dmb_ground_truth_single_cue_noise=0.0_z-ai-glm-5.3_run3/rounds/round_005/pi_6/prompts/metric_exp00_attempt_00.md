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
**Validities (n_features=6):** [0.9, 0.6, 0.6, 0.6, 0.6, 0.9]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 10: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 11: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 14: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 15: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 16: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 17: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 18: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 20: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 21: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 22: A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 24: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 25: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 26: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 27: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 28: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 29: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 30: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. PW-IVC and IVC share the identical distrust-inversion kernel ((1-v)-centered, gamma-steep); they differ ONLY in that PW-IVC adds a convex primacy gradient concentrated on positions 1-2 (mixture share tau*rho in [0.14, 0.29]; position 1 alone carries 53-72% of the primacy kernel). IVC's weights are a function of validity VALUES ONLY -- strictly position-invariant. The design therefore manipulates POSITION while holding validity allegiances exactly fixed, using the layout v = [0.90, 0.60, 0.60, 0.60, 0.60, 0.90]: the two 0.90 experts occupy the EXTREME positions 1 and 6 (maximal position contrast at TIED validity), and the four 0.60 experts form an interior equal-validity block (positions 2-5) enabling a pure within-block position-gradient assay. Because (0.40/0.10)^gamma is ~23-256 across both boxes, the 0.60 experts absorb ~97-99% of the distrust kernel, so every 0.90-vs-0.60 allegiance imbalance moves d by ~-0.245 (normalized) under BOTH theories -- the distrust signal is held constant while primacy varies. All boxes below were verified at the corners of both parameter spaces.

AXIS 1 -- MATCHED-VALIDITY PRIMACY DOSE LADDER (20 of 32 trials: D0, Z1, D1, W1, W2, D2, D2b, D3, D4, D5 + mirrors). Every cell has IDENTICAL validity multisets on the two sides (e.g., D5: A wins {0.90@1, 0.60@2, 0.60@3} vs B wins {0.60@4, 0.60@5, 0.90@6}), so the distrust kernel and the flat kernel cancel EXACTLY: IVC predicts P(A) = 0.500 PARAMETER-FREE on all twenty (d = 0 for every tau, gamma, c; the lapse preserves 0.5). PW-IVC predicts d = tau*rho*C, where C is the primacy contrast (normalized position-gradient evidence), spanning C ~ 0.04 (D0: 0.60@3 vs 0.60@4) to C ~ 0.98 (D5). Verified boxes: D0 [0.50, 0.53] (near-null control rung), Z1 [0.51, 0.59], D1 [0.51, 0.64], W1 [0.52, 0.68], W2 [0.52, 0.75], D2 (0.90@1 vs 0.90@6 duel) [0.53, 0.82], D2b [0.51, 0.73], D3 [0.54, 0.84], D4 [0.55, 0.88], D5 [0.55, 0.88]. Headline cells: on D5 and D4, observed P(A) >= 0.55 is IMPOSSIBLE for IVC at any admissible parameter combination and REQUIRED by PW-IVC; observed P(A) <= 0.52 falsifies PW-IVC's entire space. The regression slope of P(A) on C is EXACTLY 0 under IVC and strictly positive under PW-IVC -- a parametric signature, with the ladder's monotone ordering (D0 < Z1 < D1 < W1/W2 < D2/D2b < D3 < D4 < D5) providing 10 ordered rungs.

AXIS 2 -- POSITION-PERMUTATION FAMILY AT FIXED ALLEGIANCE (X2a, X2b, X2d + mirrors). All three hold the allegiance fixed at A wins {one 0.90, one 0.60}, B wins {two 0.60s}, permuting only positions. IVC: d = w(0.90) - w(0.60) IDENTICALLY on all three -> one probability, box [0.25, 0.43] (confident preference for the B product, endorsed by two 60% experts). PW-IVC: graded by primacy -- X2a (A's wins at positions 1,2 vs B's at 4,5) box [0.37, 0.69], STRADDLING/ABOVE chance because position-1 primacy (up to tau*rho*0.93) rivals the distrust deficit ((1-rho)*0.245); X2b (A at 1,5 vs B at 2,4) box [0.44, 0.50], near chance; X2d (A at 5,6 vs B at 2,3) box [0.12, 0.39], confidently B. Pointwise, PW-IVC REQUIRES a spread P(X2a) - P(X2d) >= ~0.07 (up to 0.55 at high-rho/high-beta corners) where IVC requires EXACTLY 0. Any reliable within-family spread falsifies IVC's position-invariance; exact equality falsifies PW-IVC. X2a is the design's two-sided adversarial cell: if subjects robustly distrust-follow here (P(A) <= 0.43), PW-IVC's primacy share is bounded small -- but then the Axis-1 ladder must be flat at 0.500, which is itself PW-falsifying; the joint pattern is winnowing in both directions.

AXIS 3 -- PRIMACY-DILUTED WORST-EXPERT VETO (V1 + mirror). A wins the single 0.60@2 expert vs B winning BOTH 0.90 experts (positions 1 and 6). Both theories' distrust kernels favor A (the less valid expert's endorsement counts more), but PW-IVC's primacy simultaneously favors B (positions 1 and 6 carry the maximal primacy mass): IVC box [0.55, 0.75], PW-IVC box [0.53, 0.66] -- a confidence dissociation (observed P(A) >= 0.70 is impossible for PW-IVC), and V1 also screens out plain tallying (which would pick B, 2 cues > 1).

AXIS 4 -- ANCHORS. E1 (all four 0.60 experts vs both 0.90 experts): both theories confidently favor the 0.60 product (IVC [0.75, 0.95], PW [0.76, 0.96]), verifying that subjects read and distrust-invert the communicated validities. TOT (total split): comprehension anchor, both ~1.

AGGREGATE SIGNATURE. IVC predicts exactly 0.500 on all 20 matched cells plus ONE identical probability on the six X2-family cells -- i.e., only ~4 distinct probability levels across all 32 pairs. PW-IVC predicts a position-graded profile breaking every equality in a parameter-free direction. Regressing P(A) on primacy contrast C (computed at a reference eta, e.g., 5) with inverted-validity evidence controlled (it is exactly zero on the matched cells by construction) yields a primacy coefficient of exactly 0 under IVC and strictly positive under PW-IVC. Every cell appears in A/B-mirrored form (16 complement pairs) to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

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
**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Parameters:**
- tau: [0.83, 1.0]
- c: [0.0, 0.2]
- gamma: [2.3, 3.2]
- log10_beta: [0.25, 0.7]
- epsilon: [0.08, 0.22]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying),
    # PARTIAL-CENTERING edition, iter-6 kernel-alignment retune.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])
    gamma = float(parameters["gamma"])
    c = float(parameters["c"])
    # log-uniform (heavy-tailed) subject-level inverse temperature.
    # Partially retreated to [0.25, 0.70] (beta in [1.8, 5.0], median ~3.0):
    # the iter-5 raise to [0.35, 0.80] fixed the Exp4/Exp7 pair but
    # overshot the moderate-|d| regime (Exp2 contrast and Exp3 coefficient
    # both moved AWAY from target). This is the midpoint between iter-3's
    # [0.15, 0.70] and iter-5's [0.35, 0.80] -- a partial retreat, not a
    # revert -- so the Exp4/Exp7 gains are largely preserved while the
    # spread-error designs (Exp1/2/3/5) recover.
    beta = 10.0 ** float(parameters["log10_beta"])
    eps = float(parameters["epsilon"])

    # (1) PARTIAL distrust inversion. The stated validity is mapped to an
    # error rate (1 - v_j); a FRACTION c of the smallest error rate is
    # subtracted before the gamma power:
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z
    # c = 0 recovers the uncentered kernel; c = 1 is hard centering.
    # c TRIMMED to [0.0, 0.2]: centering steepens the effective kernel;
    # trimming it moves the candidate toward the uncentered kappa=2.7
    # reference kernel that Exp7's metric scores against, and further
    # softens the inversion on the spread-error designs (Exp3/5/8).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        # Degenerate design (all validities equal): fall back to flat tally.
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and the partially
    # centered inverted-validity kernel. tau's lower bound SPLIT to 0.83
    # (midpoint of iter-3's 0.78 and iter-5's 0.88): the 0.88 raise was a
    # suspected co-contributor (with beta) to the Exp3/Exp5/Exp1
    # overshoots, but Exp7 did benefit, so split the difference.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent.
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse, HELD at [0.08, 0.22]: the beta/eps
    # decoupling from iter-5 is working (Exp4 anchors at 0.8417 vs 0.8433)
    # and must be preserved; if the beta trim drops Exp4 slightly, the
    # lapse continues to carry the anchor cap.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Primacy-slope metric on validity-matched cells (with X2-family\n    orientation absorbed by a dummy).\n\n    Logic: on every trial where the two options' winning-cue validity\n    multisets are IDENTICAL (the 20 matched cells: D0,Z1,D1,W1,W2,D2,\n    D2b,D3,D4,D5 + mirrors), the distrust kernel and the flat counting\n    kernel cancel EXACTLY, so IVC predicts P(A)=0.500 parameter-free\n    (the lapse preserves 0.5). PW-IVC predicts a systematic deviation\n    graded by the primacy contrast C = w_prim . sign(A-B), with a\n    steep convex reference gradient (eta=5).\n\n    We also include the six X2-family cells (2 wins per side, unequal\n    validity multisets). Under IVC all X2 cells share ONE probability\n    that depends only on orientation (A-won-the-0.9 vs mirror), so a\n    single +-1 orientation dummy Z absorbs IVC's entire mean structure\n    there; the within-family position spread then loads on C.\n\n    Metric = OLS coefficient of C in y ~ 1 + C + Z, with y = +1 (chose\n    A) / -1 (chose B). Under IVC this coefficient is EXACTLY 0 in\n    expectation for every admissible parameter. Under PW-IVC it is\n    strictly positive (roughly 2*(1-eps)*beta*tau*rho scale), typically\n    0.15-0.8.\n    \"\"\"\n    import numpy as np\n    import pandas as pd\n\n    n = 6\n    v = np.array([0.9, 0.6, 0.6, 0.6, 0.6, 0.9], dtype=float)\n\n    # Reference primacy kernel (eta = 5): convex, front-loaded.\n    pos = np.arange(n, dtype=float)\n    prim = ((n - pos) / (n - 1.0)) ** 5.0\n    w_prim = prim / prim.sum()\n\n    # Reference inverted-validity kernel (gamma = 3, c = 0.1).\n    # Only the SIGN of its dot product with s is used (orientation).\n    err = 1.0 - v\n    cent = np.maximum(err - 0.1 * float(err.min()), 0.0)\n    u = cent ** 3.0\n    u = u / u.sum()\n\n    a = np.array([[float(t) for t in r] for r in data['option_a_ratings']], dtype=float)\n    b = np.array([[float(t) for t in r] for r in data['option_b_ratings']], dtype=float)\n    y = 1.0 - 2.0 * np.asarray(data['response'], dtype=float).ravel()  # +1 = A\n\n    m = a.shape[0]\n    if m == 0:\n        return 0.0\n\n    s = np.sign(a - b)\n    aWin = a > b\n    bWin = b > a\n\n    C = np.zeros(m)\n    Z = np.zeros(m)\n    keep = np.zeros(m, dtype=bool)\n\n    for i in range(m):\n        ka = tuple(sorted(v[aWin[i]].tolist()))\n        kb = tuple(sorted(v[bWin[i]].tolist()))\n        nA = int(aWin[i].sum())\n        nB = int(bWin[i].sum())\n        if ka == kb and nA > 0:\n            # Matched-validity cell: distrust+flat kernels cancel exactly.\n            keep[i] = True\n            C[i] = float(np.dot(w_prim, s[i]))\n        elif nA == 2 and nB == 2 and ka != kb:\n            # X2-family cell: orientation dummy absorbs IVC's constant\n            # allegiance effect; position spread loads on C.\n            keep[i] = True\n            C[i] = float(np.dot(w_prim, s[i]))\n            Z[i] = float(np.sign(np.dot(u, s[i])))\n\n    if int(keep.sum()) < 12:\n        return 0.0\n\n    Ck = C[keep]\n    Zk = Z[keep]\n    yk = y[keep]\n\n    X = np.column_stack([np.ones_like(Ck), Ck, Zk])\n    try:\n        coef, _, _, _ = np.linalg.lstsq(X, yk, rcond=None)\n        return float(coef[1])\n    except Exception:\n        return 0.0",
  "rationale": "The design's core lever is that on the 20 validity-matched cells (trials 1-20), any validity-only weighting (IVC) gives d = 0 exactly, so P(A) = 0.500 parameter-free (lapse preserves 0.5). PW-IVC instead produces d = tau*rho * (primacy contrast), so P(A) is graded by how early A's winning cues sit. The metric is the OLS slope of choice (coded +1/-1) on the primacy contrast C (reference eta=5 gradient), computed only on trials whose winning-cue validity multisets are identical across sides; the six X2-family cells (2 wins per side, unequal multisets) are added with a single +-1 orientation dummy Z that exactly absorbs IVC's constant allegiance effect there (all X2 cells share d = w(0.9)-w(0.6) under any validity-only weights, and the mirror shares its negative), so IVC's expected slope on C is exactly zero on the full included set. Under PW-IVC the slope is strictly positive: expected value approx (1-eps)*beta*tau*rho/2 per unit C in the small-signal regime, ranging ~0.12 (weak corner) to ~0.8 (saturated corner), typically ~0.35-0.40. Per-subject noise SD of the slope is about 1/sqrt(sum C^2) ~ 0.19-0.22 (60-78 binary trials with |C| up to ~1), so at N=25 the Welch t-statistic is roughly 5+ (IVC mean 0, var ~0.05; PW mean ~0.37, var ~0.07), comfortably below alpha=0.01. The metric works identically on pooled and single-subject slices since C and Z are pure stimulus-derived covariates, and it is not trivially true for PW-IVC (weak-parameter corners yield slopes near zero), but it is exactly zero-mean under IVC at every admissible parameter, giving a sharp, low-variance contrast."
}
```

## Usage

```json
{
  "input_tokens": 7700,
  "output_tokens": 19927,
  "reasoning_tokens": 18404,
  "provider": "Sail Research"
}
```
