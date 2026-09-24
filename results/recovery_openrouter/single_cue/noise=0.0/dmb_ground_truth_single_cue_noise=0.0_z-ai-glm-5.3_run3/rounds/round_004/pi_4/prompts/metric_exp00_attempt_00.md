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
**Validities (n_features=6):** [0.95, 0.75, 0.75, 0.95, 0.75, 0.75]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 12: A=[0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 13: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 15: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 16: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0]
  trial 17: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 18: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 23: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 25: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 26: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 27: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 28: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 29: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 30: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 31: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights are a function of validity VALUES ONLY (distrust kernel (1-v)^gamma), with zero position sensitivity; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak min-max validity component (<= 0.20). Prior IVC-vs-SWC designs used non-monotone duplicated validities [6], a strictly increasing ladder [7], or a single worst expert [8]. This design uses an untried PALINDROMIC validity profile v = [0.55, 0.95, 0.75, 0.75, 0.95, 0.55] (n=6). Because IVC's weights depend only on validity values, its weight profile is exactly POSITION-SYMMETRIC (w1 = w6, w2 = w5, w3 = w4), while SWC's is strictly position-graded and late-heavy. The palindrome therefore creates validity-MATCHED comparisons at every position separation, pinning IVC to exact, parameter-free point predictions that SWC's recency gradient must break. All boxes below were verified at the corners of both parameter spaces (SWC's eta corners 2.5/4.5 with lam*rho in [0.64, 1]; IVC's gamma/tau/c/beta/eps corners).

AXIS 1 - VALIDITY-MATCHED RECENCY LADDER (R1, R2, R3 + mirrors). Duels between validity-equal experts at maximal (e1 vs e6, both 55%), large (e2 vs e5, both 95%), and minimal (e3 vs e4, both 75%) position separation. IVC: d = 0 EXACTLY for every parameter combination -> P(A) = 0.50 parameter-free (the uniform lapse preserves 0.5). SWC: d = -lam*rho*(1 - (k/5)^eta-differences), giving P(A) boxes [0.15, 0.38] (R1), [0.28, 0.45] (R2), [0.42, 0.49] (R3). SWC REQUIRES a monotone ladder with within-subject spread P(R3) - P(R1) in [0.10, 0.32] (the ratio |d_R1|/|d_R3| >= 5.6 at every corner); IVC requires exactly 0 on all three. Observed P(later expert's product) > 0.55 on R1 is impossible for IVC at ANY admissible parameter combination and required by SWC.

AXIS 2 - VALIDITY-MATCHED COALITIONS (K1-K5, D4 + mirrors; 12 of 32 trials). Coalitions whose validity multisets match exactly (K1: {0.55,0.95} vs {0.55,0.95}; K2: {0.55,0.75} both; K3: {0.95,0.75} both; K4: {0.55,0.95,0.75} both; K5: {0.55,0.75} both; D4: interleaved {0.55,0.75,0.95} both). IVC predicts EXACTLY 0.50 on all twelve, parameter-free. SWC's recency gradient requires late-block preference with boxes P(A) in [0.07, 0.40] (K1), [0.12, 0.37] (K2), [0.22, 0.45] (K3), [0.05, 0.33] (K4 - the half-block flip, the headline trial), [0.17, 0.40] (K5), [0.23, 0.42] (D4). Any reliable deviation from 0.50 in the late direction on this family falsifies IVC's entire parameter space; flatness at 0.50 falsifies SWC's.

AXIS 3 - POSITION-TRANSFER PAIRS (T1a/T1b, T2a/T2b + mirrors). Same validity allegiance, opposite position allegiance. T1a: A wins the 55% expert at position 1 vs B wins the 75% expert at position 3; T1b: A wins the 55% expert at position 6 vs B wins the 75% expert at position 4. IVC: d = w(0.55) - w(0.75) on BOTH -> IDENTICAL predictions P(A) in [0.58, 0.84] on the pair. SWC: P(T1a) in [0.42, 0.50] (early low-validity cue loses to a mid cue) but P(T1b) in [0.65, 0.83] (the same low-validity cue, placed last, beats the mid cue) - a REQUIRED pair difference >= 0.15 where IVC requires exactly 0. T1a alone is also a sign dissociation: IVC > 0.58, SWC < 0.50. T2a/T2b repeat the logic with the 95% vs 75% experts (positions 2 vs 4, and 5 vs 3): IVC identical [0.40, 0.48] on both; SWC [0.47, 0.49] vs [0.57, 0.70], required spread >= 0.08. Any within-family difference kills IVC's position-invariance; flatness within families kills SWC's recency gradient.

AXIS 4 - IVC-DIAGNOSTIC CELLS (D1, D2 + mirrors; deliberately adversarial, two-sided design). D1: the 55% expert at position 1 vs the 95% expert at position 2. IVC's distrust inversion predicts subjects prefer the WORSE expert's product: P(A) in [0.63, 0.81]; SWC predicts [0.44, 0.50] (non-overlapping). D2 (worst-expert veto): the lone 55% first expert against FOUR better experts (95/75/75/95). IVC: P(55%-expert's product) in [0.52, 0.80]; SWC: [0.09, 0.41] (non-overlapping). Observed preference for the 55% expert beyond 0.50 on D1/D2 falsifies SWC's entire space - these cells make the design genuinely risky for the advocated theory rather than one-sided.

AXIS 5 - ANCHOR (TOT + mirror). Total split: both theories predict >= 0.69, establishing task comprehension and overall responsiveness.

PARAMETRIC SIGNATURE. IVC predicts only ~6 distinct probability levels across all 32 pairs (exactly 0.50 on the 18 validity-matched trials; one level per validity-allegation multiset on the transfer/duel trials; high confidence on TOT), with EXACT equalities within families. SWC predicts a position-ordered profile breaking every equality. Regressing P(A) on recency evidence sum((j/5)^3.5 * s_j) versus inverted-validity evidence sum((1-v_j)^2.7 * s_j) yields opposite-loading profiles: the two axes are decorrelated by the palindrome (expert 1: high inverted weight, zero recency; expert 6: equally high inverted weight, maximal recency - the transfer pairs exploit exactly this). Every conflict type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

**Parameters:**
- beta: [0.8, 1.7]
- epsilon: [0.0, 0.03]
- lambda: [0.8, 1.0]
- rho: [0.8, 1.0]
- eta: [2.5, 4.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
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
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Late-cue preference rate on theory-separating trials.\n\n    With validities v = [0.95, 0.75, 0.75, 0.95, 0.75, 0.75] (positions 1&4\n    are the 0.95 experts), select trials of two kinds:\n      (1) VALIDITY-MATCHED trials: the two options' discriminating cues have\n          identical validity multisets. IVC's weights depend only on validity\n          VALUES, so its evidence d = 0 exactly -> P = 0.50 parameter-free.\n          SWC's convex recency gradient always favors the option whose cues\n          sit later in the display -> P(late) >> 0.50.\n      (2) VALIDITY-CONFOUNDED-LATE trials: the later-cue option carries\n          strictly MORE high-validity (0.95) cues than the early option.\n          IVC's distrust inversion favors the 0.75-expert (early) side ->\n          P(late) < 0.50; SWC still favors the late side -> P(late) > 0.50.\n          (Sign dissociation on top of family (1).)\n\n    The metric is the fraction of choices of the LATER-cue option over the\n    selected trials. Mirrored A/B pairs make it immune to pure side bias.\n    \"\"\"\n    v = np.array([0.95, 0.75, 0.75, 0.95, 0.75, 0.75])\n    n = v.shape[0]\n    pos = np.arange(n, dtype=float)\n    hi = v >= (v.max() - 1e-9)\n\n    def to_arr(x):\n        if isinstance(x, np.ndarray):\n            return x.astype(float)\n        if isinstance(x, (list, tuple)):\n            return np.asarray(list(x), dtype=float)\n        s = str(x).strip(\"[]() \").replace(\",\", \" \").split()\n        return np.asarray([float(c) for c in s], dtype=float)\n\n    a_col = data[\"option_a_ratings\"].values\n    b_col = data[\"option_b_ratings\"].values\n    r_col = data[\"response\"].values\n\n    hits = []\n    for i in range(len(data)):\n        try:\n            a = to_arr(a_col[i])\n            b = to_arr(b_col[i])\n        except Exception:\n            continue\n        if a.shape != (n,) or b.shape != (n,):\n            continue\n\n        s = np.sign(a - b)\n        a_cues = np.where(s > 0)[0]  # features favoring A\n        b_cues = np.where(s < 0)[0]  # features favoring B\n        if a_cues.size == 0 or b_cues.size == 0:\n            continue  # no discrimination (also drops the all-1 vs all-0 anchor)\n\n        La = float(pos[a_cues].sum())\n        Lb = float(pos[b_cues].sum())\n        if La == Lb:\n            continue\n        late_a = La > Lb\n\n        va = sorted(v[a_cues].tolist())\n        vb = sorted(v[b_cues].tolist())\n        matched = (va == vb)\n\n        if not matched:\n            late_cues = a_cues if late_a else b_cues\n            early_cues = b_cues if late_a else a_cues\n            # keep only if the late option carries strictly more 0.95-expert\n            # cues (IVC must then prefer the EARLY side)\n            if int(hi[late_cues].sum()) <= int(hi[early_cues].sum()):\n                continue\n\n        resp = int(r_col[i])\n        if resp not in (0, 1):\n            continue\n        chose_late = (resp == 0) if late_a else (resp == 1)\n        hits.append(1.0 if chose_late else 0.0)\n\n    if not hits:\n        return 0.5\n    return float(np.mean(hits))\n```",
  "rationale": "The design's validities [0.95, 0.75, 0.75, 0.95, 0.75, 0.75] make cue POSITION and cue VALIDITY partially decorrelated, which is exactly what splits the two theories. IVC's weights are a function of validity values only, so on any trial whose two options have identical validity multisets its evidence is exactly d = 0 and P(choose either option) = 0.50 for every parameter combination (the uniform lapse preserves 0.5). SWC's weights contain a convex recency term rho*(j/5)^eta with rho >= 0.8, so on those same trials it reliably prefers whichever option's discriminating cues sit later. The metric therefore computes the rate of choosing the LATER-cue option over a curated trial set: (a) the 10 validity-matched trial types (trials 3/4, 9/10, 13/14, 17/18, 25/26 \u2014 e.g. trial 13 pits features {1,2,3} against {4,5,6} with identical validity multisets {0.95,0.75,0.75}), where IVC = 0.50 exactly and SWC's late-preference ranges ~0.60-0.83 across its parameter box; and (b) the 6 trial types where the later option carries strictly more 0.95-expert cues (trials 5/6, 11/12, 23/24), where IVC's distrust inversion favors the early 0.75-expert side (P(late) ~ 0.25-0.43 across its box) while SWC still favors the late side (P(late) ~ 0.54-0.72). All other trial types \u2014 where both theories agree in direction (e.g. anchors, 0.95-vs-0.75 duels at distant positions) \u2014 are excluded so they cannot dilute the contrast. Every selected type appears in A/B-mirrored form and orientation is recomputed per trial from cue positions, so pure side biases cancel. Expected values: IVC ~0.44 (exactly 0.50 on the matched subset, below 0.50 on the confounded subset), SWC ~0.62-0.72; per-subject noise is binomial over 48 selected trials (sd ~ 0.07-0.09), giving a Welch t around 8-10 at N=25 \u2014 a large mean gap with tight per-subject stability. A value near 0.50 with no late-ward tilt falsifies SWC's recency gradient; a reliable late-ward tilt above ~0.55 falsifies IVC's entire parameter space."
}
```

## Usage

```json
{
  "input_tokens": 7103,
  "output_tokens": 24025,
  "reasoning_tokens": 22453,
  "provider": "Sail Research"
}
```
