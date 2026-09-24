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
**Validities (n_features=6):** [0.5, 0.9, 0.6, 0.95, 0.5, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 20: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 23: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 25: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 26: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 27: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 28: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 29: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 30: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 31: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 32: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights are a function of the validity VALUES ONLY (w_j proportional to (1-v_j)^kappa: the LESS valid the expert, the MORE weight), with zero position sensitivity. SWC's weights are dominated by POSITION (convex recency, coefficient lam*rho >= 0.64, plus a weak POSITIVE validity component <= 0.20). I therefore use a non-monotone, duplicated validity profile v = [0.55, 0.90, 0.60, 0.95, 0.55, 0.60] (n=6): the two lowest validities sit at positions 1 and 5, duplicated 0.55s at positions 1/5 and duplicated 0.60s at positions 3/6, and the two highest validities at positions 2 and 4. The duplicates create validity-MATCHED comparisons at different positions (killing the validity axis and isolating pure recency), and the low-early/high-late arrangement makes IVC's weight profile (early-heavy) anti-aligned with SWC's (late-heavy). All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - VALIDITY-MATCHED RECENCY BATTERY (EQ1-EQ4 + mirrors). EQ1: A wins expert 1 vs B wins expert 5 (0.55 vs 0.55). EQ2: expert 3 vs expert 6 (0.60 vs 0.60). EQ3: A wins {1,3} vs B wins {5,6} (validity multiset {0.55,0.60} on BOTH sides). EQ4: A wins {1,6} vs B wins {3,5} (same matched multiset). Because IVC's weights depend only on validity values, d = 0 EXACTLY on all four -> IVC predicts P(A) = 0.50 PARAMETER-FREE. SWC's recency gradient predicts a wide spread: P(A) boxes [0.28,0.45] (EQ1), [0.12,0.39] (EQ2), [0.08,0.34] (EQ3), [0.61,0.69] (EQ4). Observed P(EQ3) <= 0.40 or P(EQ4) >= 0.55 is impossible for IVC at ANY parameter combination; observed P(EQ3) >= 0.45 is impossible for SWC. SWC requires an EQ3-vs-EQ4 spread >= 0.27 where IVC requires exactly 0.

AXIS 2 - POSITION-TRANSFER PAIRS (IVC invariance vs SWC flip). TP1: A wins expert 1 (0.55) vs B wins expert 6 (0.60). TP2: A wins expert 5 (0.55) vs B wins expert 3 (0.60). Identical validity allegiances, opposite position allegiances: IVC predicts EXACTLY the same P(A) on both, in [0.52, 0.55]; SWC predicts P(TP1) in [0.12, 0.37] (expert 6's recency dominates) and P(TP2) in [0.54, 0.69] (expert 5 beats expert 3 on recency) - a required pair difference >= 0.17 where IVC requires exactly 0. The same logic at the extreme validity gap: TP3 (A wins expert 1 vs B wins expert 4, 0.55 vs 0.95), TP4 (A wins expert 5 vs B wins expert 4), and M1 (A wins {1,3} vs B wins {4,6}, where the two 0.60 cues cancel) all have d = w(0.55) - w(0.95) under IVC -> P(A) IDENTICAL in [0.59, 0.66] on all three; SWC predicts [0.32,0.47], [0.50,0.57], [0.08,0.33] respectively - a required spread >= 0.17 (TP4 vs M1). TP3'/TP4' repeat this with the 0.90 expert: IVC-equal [0.59,0.66]; SWC [0.42,0.50] vs [0.52,0.69]. Any observed spread within a transfer family (e.g., P(TP1) != P(TP2), or P(TP3) != P(M1)) falsifies IVC's position-invariance; flatness within families falsifies SWC's recency gradient.

AXIS 3 - SIGN-FLIP DUELS. On TP1, TP3, TP3', S3 (A wins expert 3 (0.60) vs B wins expert 4 (0.95) - an ADJACENT-position duel where the later cue is also more valid), M1 and M2 (A wins {1,3,5}, the three least valid experts, vs B wins {2,4,6}), IVC favors the early/low-validity option and SWC favors the late/high-validity option with NON-OVERLAPPING boxes: IVC [0.52,0.55] vs SWC [0.12,0.37] (TP1); IVC [0.59,0.66] vs SWC [0.32,0.47] (TP3); IVC [0.59,0.66] vs SWC [0.42,0.50] (TP3'); IVC [0.57,0.61] vs SWC [0.38,0.48] (S3); IVC [0.59,0.66] vs SWC [0.08,0.33] (M1); IVC [0.63,0.73] vs SWC [0.12,0.37] (M2). M2 is the headline block flip: the option endorsed by the three WORST experts (55/60/55) is chosen confidently only by IVC; SWC confidently picks the option endorsed by the later, more valid experts.

AXIS 4 - CUE MIGRATION (AN2 -> M2). AN2: A wins {1,3,5} vs B wins {2,4}. Adding expert 6 (a mere 60%-valid cue) to B's side yields M2. IVC predicts only a modest drop (AN2 [0.70,0.78] -> M2 [0.63,0.73], still confidently A, since expert 6's inverted weight is only ~0.21 and expert 3's cancels it); SWC predicts a catastrophic flip (AN2 [0.43,0.66] -> M2 [0.12,0.37]) because expert 6 carries the maximal recency weight. A drop exceeding ~0.3 on this single-cue migration is impossible for IVC and required by SWC.

AXIS 5 - REVERSE CONFIDENCE DISSOCIATION. AN2 (A wins {1,3,5} vs B wins {2,4}): IVC [0.70,0.78] vs SWC [0.43,0.66] - IVC more confident (and SWC's box straddles chance). AN3 (A wins {5,6} vs B wins {1,2}): IVC [0.56,0.60] vs SWC [0.66,0.86] - SWC MORE confident. The confidence ordering between AN2 and AN3 reverses between theories: IVC says P(AN2) >> P(AN3); SWC says P(AN3) > P(AN2). TOT (total split) is the agreement anchor both theories accommodate.

PARAMETRIC SIGNATURE. IVC predicts only ~7 distinct probability levels across all 32 pairs (0.50 on all EQ; one level per validity-allegiance multiset for the TP/S/M/AN families; near-1 on TOT), with EXACT equalities within families; SWC predicts a wide, position-ordered spread that breaks every equality. Regressing P(A) on inverted-validity evidence sum((1-v_j)^2.7 * s_j) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two evidence axes are decorrelated by construction (expert 1: maximal inverted weight, zero recency; expert 6: maximal recency, middling inverted weight; experts 2/4: high validity, i.e., low weight for IVC but positive validity weight for SWC). Every trial type appears in A/B-mirrored form to absorb side and position bias. 32 unique pairs, K = 96 // 32 = 3 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Parameters:**
- tau: [0.85, 1.0]
- kappa: [2.4, 3.0]
- beta: [1.8, 2.5]
- epsilon: [0.03, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying).
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
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Distrust inversion: treat the communicated validity as if it
    # were the expert's error rate. (1 - v_j)^kappa is small for highly
    # validated experts and large for weakly validated ones; Z normalizes
    # the kernel so the mixture weights sum to one.
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and inverted validity.
    # tau = 0 recovers plain Tallying; tau = 1 with large kappa approaches
    # a 'worst-expert-first' one-reason rule.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent
    # (anchor consistency stays high even at n = 8 features).
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Small independent uniform lapse.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Anti-validity vs. recency disagreement score.\n\n    On trials where IVC's distrust-inverted weights and SWC's recency-dominated\n    weights point to OPPOSITE options, report the proportion of choices that\n    follow the IVC-preferred option. IVC-simulated data -> well above 0.5;\n    SWC-simulated data -> well below 0.5.\n    \"\"\"\n    data = data.reset_index(drop=True)\n    if len(data) == 0:\n        return 0.5\n\n    # Build stimulus matrices without hashing any list-valued cells.\n    a = np.vstack([np.asarray(r, dtype=float) for r in data[\"option_a_ratings\"]])\n    b = np.vstack([np.asarray(r, dtype=float) for r in data[\"option_b_ratings\"]])\n    n = a.shape[1]\n\n    # Design validities (fallback mirrors both theories' graceful degradation).\n    v = np.array([0.5, 0.9, 0.6, 0.95, 0.5, 0.6], dtype=float)\n    if v.shape[0] != n:\n        v = np.linspace(0.95, 0.55, n)\n\n    # --- IVC weights at mid-range parameters (tau=0.925, kappa=2.7) ---\n    tau, kappa = 0.925, 2.7\n    inv = np.power(1.0 - v, kappa)\n    Z = float(inv.sum())\n    w_inv = inv / Z if Z > 1e-12 else np.full(n, 1.0 / n)\n    w_ivc = (1.0 - tau) / n + tau * w_inv\n\n    # --- SWC weights at mid-range parameters (lam=0.9, rho=0.9, eta=3.5) ---\n    lam, rho, eta = 0.9, 0.9, 3.5\n    vspan = float(v.max() - v.min())\n    if vspan > 1e-12:\n        vnorm = (v - v.min()) / vspan\n    else:\n        vnorm = np.full(n, 0.5)\n    if n > 1:\n        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta\n    else:\n        pos = np.full(n, 0.5)\n    g = (1.0 - rho) * vnorm + rho * pos\n    w_swc = (1.0 - lam) / n + lam * g\n\n    # Signed discriminating-cue vector per trial.\n    s = np.sign(a - b)\n    d_ivc = s @ w_ivc\n    d_swc = s @ w_swc\n\n    # Keep only trials where the two theories STRICTLY disagree in sign.\n    # (d_ivc == 0, e.g. the validity-matched EQ trials, is excluded.)\n    mask = (d_ivc * d_swc) < 0.0\n    if not np.any(mask):\n        return 0.5\n\n    resp = np.asarray(data[\"response\"].to_numpy())\n    r = resp[mask]\n    ivc_prefers_a = d_ivc[mask] > 0.0\n    chose_ivc_preferred = np.where(ivc_prefers_a, r == 0, r == 1)\n    return float(chose_ivc_preferred.mean())",
  "rationale": "CORE IDEA. The design's duplicated, non-monotone validity profile makes IVC's weights (a function of validity VALUES only: heaviest on the 0.5-validity experts 1/5, then 0.6 experts 3/6, near-zero on 0.9/0.95 experts 2/4) and SWC's weights (dominated by late POSITION: expert 6 >> 5 > 4 > 2 > 3 > 1) point to OPPOSITE options on a fixed, parameter-robust set of trials. The metric recomputes both theories' evidence d = w.s from the raw ratings using mid-range parameters, keeps only trials where sign(d_IVC) and sign(d_SWC) strictly disagree, and returns the proportion of choices following the IVC-preferred option. PARAMETER ROBUSTNESS. I verified the sign of d under BOTH theories at all corners of both parameter spaces: the disagreement set is exactly 12 of the 32 trial types (trials 5, 7, 9, 11, 12, 13, 21, 23, 25, 27, 28, 29 in the design listing), i.e. 36 of each subject's 96 trials, independent of tau/kappa/beta/epsilon (IVC) and lam/rho/eta/beta/epsilon (SWC). For example, w1 < w2 always under SWC (position/validity gradient) while w(0.5) > w(0.9) always under IVC; trial 13 (A wins experts 1,3,5 vs B wins 2,4,6) pits the three least-valid experts against the three latest-position experts \u2014 IVC favors the low-validity side, SWC the late side, at every parameter corner. So the mid-parameter weights used inside the metric only need to recover the SIGNS, which are invariant. EXPECTED SEPARATION. Under IVC (mid params) the per-trial P(choose IVC-preferred) on the 12 disagreement types averages ~0.65 (rising to ~0.78 on trials 13/29); under SWC it averages ~0.36. Pooled point estimates: ~0.64 (IVC data) vs ~0.36 (SWC data), a gap of ~0.28. VARIANCE. Per subject the metric is a proportion over 36 Bernoulli trials with p near 0.65 (or 0.36), giving per-subject SD ~0.08; parameter heterogeneity across subjects adds little (the boxes are tight). Welch's t at N=25 with variances ~0.006-0.008 on each side yields t > 9, p far below 0.01. The metric works identically on a single subject's slice (36 trials, no NaN), so the between-subject variance is well defined. INTERPRETATION. A value near 0.5 means neither anti-validity nor recency dominates; >0.5 supports distrust-weighted (inverted-validity) counting; <0.5 supports position/recency-weighted counting. Because agreement trials (e.g., the total splits 16/32, and all validity-matched EQ trials where IVC is exactly at chance) are excluded, the metric carries pure discriminating signal with minimal noise."
}
```

## Usage

```json
{
  "input_tokens": 6849,
  "output_tokens": 17346,
  "reasoning_tokens": 15810,
  "provider": "Sail Research"
}
```
