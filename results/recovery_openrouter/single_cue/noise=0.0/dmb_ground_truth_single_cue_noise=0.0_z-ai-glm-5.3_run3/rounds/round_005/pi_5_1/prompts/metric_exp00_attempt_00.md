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
**Validities (n_features=6):** [0.95, 0.6, 0.6, 0.95, 0.6, 0.95]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 15: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 18: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 21: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 22: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 23: A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 24: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 25: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 26: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 27: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 28: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 29: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 30: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 31: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 32: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1]

**Rationale:** STRUCTURAL LEVER. IVC and PW-IVC share the distrust-inversion kernel; they differ ONLY in that PW-IVC adds a convex primacy gradient concentrated on the first one or two positions (mixture share tau*rho in [0.14, 0.29]; positions 1-2 carry ~82-94% of the primacy kernel, and position 1 alone carries 0.53-0.72). IVC's weights are a function of validity VALUES ONLY -- strictly position-invariant. The design therefore manipulates POSITION while holding validity allegiances fixed, using the layout v = [0.95, 0.60, 0.60, 0.95, 0.60, 0.95], which (i) decorrelates validity from position (three 0.60 experts at positions 2,3,5; three 0.95 experts at positions 1,4,6), (ii) places a 0.95 expert at POSITION 1 -- the maximal-primacy slot carrying ~0.1% of IVC's kernel mass -- creating head-to-head primacy-vs-distrust conflicts, and (iii) triplicates each validity level, enabling validity-matched cells where IVC is pinned to exact point predictions. Because (0.40/0.05)^gamma is ~125-340, the three 0.60 experts absorb ~99.8% of the distrust kernel: every 0.60-vs-0.95 allegiance imbalance moves d by ~tau*(1/3) regardless of which positions are involved. All boxes below were verified at the corners of both parameter spaces, respecting the coupling between the kernel coefficient tau*(1-rho) and the primacy coefficient tau*rho.

AXIS 1 -- VALIDITY-MATCHED PRIMACY DOSE LADDER (trials 1-14). Duels and coalitions whose validity multisets match exactly (e.g., A wins the 0.95 expert at position 1 vs B the 0.95 expert at position 6; A wins {0.95@1, 0.60@2} vs B {0.60@5, 0.95@6}). IVC: d = 0 EXACTLY for every parameter combination -> P = 0.500 parameter-free on ALL 14 matched cells. PW-IVC: d = tau*rho*(primacy contrast) > 0 strictly, growing with position contrast: near-zero dose D3 (positions 4 vs 6) P in [0.50, 0.52]; low doses (0.60-duels, positions 2v3, 3v5, 2v5) P in [0.50, 0.63]; high dose D1 (position 1 vs 6) P in [0.53, 0.82]; max dose C1 ({1,2} vs {5,6}) P in [0.55, 0.86]. Observed preference >= 0.55 for the earlier expert's product on the D1/C1 family is impossible for IVC at ANY admissible parameter combination; flat 0.500 across the ladder falsifies PW-IVC. The slope of P versus primacy contrast is EXACTLY 0 under IVC and strictly positive under PW-IVC -- a parametric signature.

AXIS 2 -- FIVE-WAY EXACT EQUALITY (T1a/T1b/T1c/M7a/M7b + mirrors; trials 15-20, 25-28). All five hold validity allegiances fixed (T1: A wins one 0.60 vs B one 0.95; M7: A wins two 0.60 + one 0.95 vs B one 0.60 + two 0.95) while permuting positions, so d_IVC = tau*(w60 - w95) IDENTICALLY: IVC predicts ONE probability, box [0.59, 0.81], on all TEN cells. PW-IVC predicts a strict ordering for every admissible parameter set: M7a (A's wins at positions 1-3) [0.65, 0.91] > T1b (0.60@2 vs 0.95@6) [0.57, 0.90] > T1c (0.60@3 vs 0.95@4) [0.53, 0.87] > T1a (0.60@5 vs 0.95@1) [0.50, 0.79] > M7b (A's wins at 3,5,6 vs B's at 1,2,4) [0.44, 0.77]. On M7b PW-IVC's box dips BELOW CHANCE on the very allegiance pattern where it predicts 0.65-0.91 when positions are permuted, while IVC requires the same probability. Any reliable spread across the family (or a sub-0.5 proportion on M7b) kills IVC; equality kills PW-IVC.

AXIS 3 -- 2v2 TRANSFER (trials 21-24). A wins both 0.60s, B both 0.95s, positions permuted (A@{2,3} vs B@{4,6} vs A@{3,5} vs B@{1,4}). IVC: identical [0.68, 0.93] on all four cells. PW-IVC: T2a [0.75, 0.92] vs T2b [0.59, 0.89] with a pointwise strict ordering -- a required spread where IVC requires exactly 0.

AXIS 4 -- PRIMACY-RESCUE (trials 31-32). The product endorsed by the three 95% experts (positions 1, 4, 6) vs the product endorsed by a single 60% expert (position 5). IVC: the worst-expert veto, P(60%-product) in [0.57, 0.81]. PW-IVC: position-1 primacy (0.53-0.72 of the primacy kernel) fights the distrust kernel, d in [-0.03, +0.17], P in [0.42, 0.78] with sub-chance values at strong-primacy corners -- a confidence dilution (and possible sign reversal) IVC cannot produce anywhere in its box.

AXIS 5 -- SHARED-CORE ANCHOR AND NULL CONTROL (trials 29-30; trial 3-4). E2: the three 60% experts' product vs the three 95% experts' product at a 3-3 count tie: both theories confidently favor the 60% product (IVC [0.74, 0.95], PW [0.71, 0.97]) -- verifying that subjects read and distrust-invert the communicated validities (and screening out plain tallying, which predicts exactly 0.5). D3 (0.95@4 vs 0.95@6) is the zero-dose rung: both theories ~0.5, assaying generic side/late-position bias.

PARAMETRIC SIGNATURE. IVC predicts only FIVE distinct probability levels across all 32 pairs (exactly 0.500 on the 14 matched cells; one level on the ten T1/M7 cells; one on the four T2 cells; one on E2; one on V1; mirrors give complements), with EXACT equalities within families. PW-IVC predicts a position-graded profile that breaks every equality in a parameter-free direction. Regressing P(A) on primacy evidence (position-gradient-weighted cue signs, e.g., eta=5) controlling for inverted-validity evidence yields a primacy coefficient of exactly 0 under IVC and strictly positive under PW-IVC. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials; every cell appears in A/B-mirrored form (16 complement pairs) to absorb side bias.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    # Within-allegiance-cell OLS slope of choice on reference primacy evidence.\n    #\n    # Identification: group every trial by (c1, c2) where\n    #   c1 = number of 0.60-expert wins for A minus for B (positions 2,3,5),\n    #   c2 = total signed cue count.\n    # Any position-INVARIANT validity-weighted model (IVC at ANY parameter\n    # combo, plain tallying, any lapse/transform) makes P(choose A) a function\n    # of (c1, c2) ONLY. Demeaning the primacy regressor within these cells\n    # therefore makes the population OLS slope EXACTLY 0 under IVC,\n    # parameter-free, for every subject. PW-IVC's primacy gradient breaks the\n    # cell-conditional independence, producing a strictly positive slope.\n\n    ratings_a = data['option_a_ratings']\n    ratings_b = data['option_b_ratings']\n    A = np.array([np.asarray(r, dtype=float) for r in ratings_a], dtype=float)\n    B = np.array([np.asarray(r, dtype=float) for r in ratings_b], dtype=float)\n    if A.ndim != 2 or A.shape[0] == 0:\n        return 0.0\n    n = A.shape[1]\n\n    # Reference primacy kernel, eta = 5 (middle of PW-IVC's eta range [3.5, 6.5]).\n    if n > 1:\n        pos = np.arange(n, dtype=float)\n        wp = ((n - pos) / (n - 1.0)) ** 5.0\n        wp = wp / wp.sum()\n    else:\n        wp = np.ones(1)\n\n    S = np.sign(A - B)\n    y = 1.0 - np.asarray(data['response'], dtype=float)  # 1 = chose A\n\n    x1 = S.dot(wp)  # primacy evidence favoring A\n\n    # Allegiance cell for this design: validities [0.95, 0.6, 0.6, 0.95, 0.6, 0.95]\n    # -> 0.60 experts at 0-indexed positions 1, 2, 4.\n    if n == 6:\n        c1 = S[:, 1] + S[:, 2] + S[:, 4]\n    else:\n        c1 = np.zeros(A.shape[0])\n    c2 = S.sum(axis=1)\n\n    g = c1.astype(int).astype(str) + \"_\" + c2.astype(int).astype(str)\n    df = pd.DataFrame({'x1': x1, 'y': y, 'g': g})\n\n    # Demean primacy evidence within allegiance cell: removes ALL validity-based\n    # mean structure (distrust kernel at any gamma/c/tau, flat counting, lapse).\n    df['x1c'] = df['x1'] - df.groupby('g')['x1'].transform('mean')\n    ssx = float((df['x1c'] ** 2).sum())\n    if ssx <= 1e-12:\n        return 0.0\n    ybar = float(df['y'].mean())\n    slope = float((df['x1c'] * (df['y'] - ybar)).sum() / ssx)\n    return slope",
  "rationale": "PRIMACY SLOPE AT FIXED VALIDITY ALLEGIANCE. The two theories share the distrust-inversion kernel and differ ONLY in the position-dependent primacy component, so the metric must isolate position effects while absorbing all validity-based structure. Key algebraic fact: for ANY position-invariant weight vector w (IVC at any tau/gamma/c, plain tallying, plus any lapse), the latent evidence is w\u00b7s = (w60\u2212w95)\u00b7c1 + w95\u00b7c2, where c1 = (s2+s3+s5) is the 0.60-expert allegiance count and c2 = total signed cue count. Hence P(choose A) is a function of the cell (c1, c2) alone. The metric groups trials by (c1, c2) (9 non-empty cells in this design: the 14 validity-matched pairs form cell (0,0); the T1/M7 family forms (\u00b11,0); T2 forms (\u00b12,0); E2 forms (\u00b13,0); V1 forms (\u22131,\u00b12)), demeans a fixed reference primacy kernel score x1 = w_prim\u00b7s (eta=5, the midpoint of PW-IVC's eta range, so the kernel mismatch attenuation is <2% at both eta corners) within cell, and returns the OLS slope of choice on the demeaned score. Because the within-cell sum of the demeaned regressor is exactly zero, the population slope is EXACTLY 0 under IVC for every admissible parameter combination \u2014 the per-subject distribution under IVC is pure binomial noise (SD \u2248 0.11 from ~0.22 residual variance over \u03a3x\u0303\u00b2 \u2248 18.2 across 96 trials), with no parameter heterogeneity contribution. Under PW-IVC the slope is strictly positive at every parameter point: within-cell position contrasts (e.g., M7a x1=+0.93 vs M7b x1=\u22120.72 inside cell (1,0); the matched-cell dose ladder inside cell (0,0); T2a vs T2b inside cell (2,0)) all push choices toward the earlier-position option. Corner calculations give slopes \u2248 0.05 (beta=2.2, rho=0.16), \u2248 0.15 (mid-range), \u2248 0.30 (beta=8, rho=0.30), i.e., mean \u2248 0.14\u20130.15 with between-subject SD \u2248 0.12 (binomial + parameter spread). At N=25 the Welch t-statistic is \u2248 4+ (p \u2248 1e-4), comfortably below alpha=0.01, with the IVC side pinned at zero variance-inflating structure. The metric works identically on the pooled DataFrame and on single-subject slices (each subject sees all 32 pairs \u00d7 3 reps, so cells are identically composed)."
}
```

## Usage

```json
{
  "input_tokens": 7773,
  "output_tokens": 39453,
  "reasoning_tokens": 38002,
  "provider": "Sail Research"
}
```
