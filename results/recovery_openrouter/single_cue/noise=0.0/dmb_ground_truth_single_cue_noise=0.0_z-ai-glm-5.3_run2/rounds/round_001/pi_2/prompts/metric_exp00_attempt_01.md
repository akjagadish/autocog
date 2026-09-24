# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 11: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1]
  trial 12: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 13: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 15: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]

**Rationale:** This design dissociates Tallying from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) along three novel quantitative axes that the previous designs did not exploit. SETUP: 6 features with validities [0.90, 0.80, 0.70, 0.65, 0.60, 0.55]. Because validities descend across positions, the misbound hierarchy (positional reversal) equals an ascending-validity cascade: the reversed TTB model consults f5 (0.55) first, then f4, f3, f2, f1, f0 (0.90) last. AXIS 1 — THE CONFIDENCE CEILING (T3, T4, T12): the competing theory's beta is locked in [0.45, 0.55] with epsilon in [0.11, 0.15], so its predicted choice probability on ANY discriminating trial is capped at ~(0.87)*sigmoid(0.55)+0.075 ≈ 0.62. T3/T4 are unanimous trials (margin ±6) and T12 is a sparse trial where only the two highest-validity experts discriminate, both favoring A. Every sensible strategy — including Tallying with a moderate beta (p(A) = (1-eps)*sigmoid(2*beta)+eps/2, up to ~0.93) — predicts near-ceiling consistency here, but the misbinding model is mathematically FORBIDDEN from exceeding ~0.62. Human consistency above ~65% on these cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. AXIS 2 — EXTREME-MARGIN CONFLICT WITH ORDINAL INVERSION (T1, T2, T14): on T1, A wins the four highest-validity experts (margin +4) but the lowest-validity expert (f5) favors B. Tallying predicts p(A) = (1-eps)*sigmoid(4*beta)+eps/2 ≥ 0.59 (well above 0.5 for any beta > 0.1); reversed TTB follows f5 to B, predicting p(A) ≈ 0.39. Crucially, compare T1 to T5 (margin 0, decider f5 favors A): Tallying predicts p(T1) >> p(T5) = exactly 0.50, while reversed TTB predicts p(T1) ≈ 0.39 < p(T5) ≈ 0.61 — a complete ORDINAL INVERSION of the two-cell response profile that no parameter setting of either model can reconcile with the other's prediction. T2 and T14 mirror this inversion, ruling out side bias. AXIS 3 — KNIFE-EDGE AND ISO-MARGIN FAMILIES (T5–T11, T13, T14, T16): five margin-0 trials (T5–T8, T16) pin Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — a parameter-free prediction — while reversed TTB predicts 0.61/0.39 splits on every one, at shallow (f5), middle (f3), and deep (f1) cascade depths, i.e., depth-invariant. The iso-margin family {T9, T10, T11, T12} all have margin +2: Tallying predicts IDENTICAL p(A) on all four for any parameters; reversed TTB predicts 0.61, 0.39, 0.39, 0.61 (decider flips between f5, f3, f1). The mirror family {T13, T14} does the same at margin -2, where Tallying prefers B (~0.30 for A) but reversed TTB prefers A on T14 (0.61) — opposite sides of 0.5. T15 (full tie) is the shared 50/50 control anchoring lapse and side bias. GLOBALLY: reversed TTB's entire 16-cell profile is a two-level step function (≈0.61 toward the lowest-validity discriminator, ≈0.39 against, 0.50 on the full tie), whereas Tallying predicts a monotone psychometric function of margin with exact 0.5 at m=0 and identical values within margin levels — two cross-structured profiles that model comparison on the full response profile separates cleanly even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Parameters:**
- beta: [0.45, 0.55]
- epsilon: [0.11, 0.15]
- kappa: {1.0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality: one-reason decision making (TTB
    # cascade) run over a SUBJECTIVE cue hierarchy that is
    # anti-correlated with the instructed validities. Subjects
    # misbind the communicated validity list to feature positions
    # (highest validity bound to the last-presented feature), so the
    # cascade effectively consults cues in ascending true validity.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 0 -> instructed hierarchy;
    # kappa = 1 -> fully reversed hierarchy (near-complete misbinding).
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta), mapping beta=0 to 50/50
    # guessing and large beta to deterministic choice.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] rationale: The metric exploits the parameter-free knife-edge structure of the design. Tallying's predictions depend only on the tally margin: p(A) = (1-eps)*sigmoid(margin*beta) + eps/2. Hence (i) on all margin-0 cells p(A) is EXACTLY 0.5 for any (beta, epsilon), and (ii) all cells sharing a margin get IDENTICAL p(A). The design was built so that within each of the margin families {0: T5-T8}, {+2: T9-T12}, {-2: T13-T14} the reversed-hierarchy decider (first discriminating feature in ascending-validity order) points toward A and toward B equally often. Therefore the signed follow-the-reversed-decider score, averaged over all margin in {-2, 0, +2} cells, has expectation EXACTLY 0 under Tallying for every parameter setting — a hard, parameter-free anchor. Under the competing theory every one of these cells is decided by the reversed cascade and followed with probability p = (1-eps)*sigmoid(beta) + eps/2, which its parameter bounds pin to [0.57, 0.64]; each cell therefore contributes v*(p(A)-0.5) = +0.11 in expectation, so the metric lands near +0.11. Per subject the metric uses 60 trials (10 cells x 6 reps), giving per-subject SD of roughly 0.06 under BOTH theories (binomial noise /6 reps, averaged over 10 cells), so at N=25 the Welch t-statistic is approximately 0.11 / sqrt(2*0.004/25) ~ 6, far beyond the 0.01 significance threshold, and remains significant (>4) even at the competing theory's weakest parameter corner. The contrast is a pure ordinal-inversion signature (flat-at-zero psychometric profile vs. decider-driven alternation within iso-margin families) that no parameter setting of either model can fake as the other.
metric_source:
```python
def metric(data: pd.DataFrame) -> float:
    # "Reversed-cascade alignment on iso-margin cells."
    #
    # For every trial whose TALLY MARGIN is -2, 0, or +2 (cells T5-T14
    # of the design), compute which option the REVERSED-HIERARCHY
    # one-reason cascade (consult features in ascending true validity,
    # i.e. reversed index order) would pick. Score +1 if the subject
    # picked that decider's option, -1 otherwise, and average.
    #
    # Under Tallying this expectation is EXACTLY 0 for any (beta, eps):
    # margin-0 cells give p(A)=0.5 exactly, and within the margin+2 and
    # margin-2 families the decider direction sums to zero while
    # Tallying assigns identical p(A) to equal-margin cells, so the
    # signed deviations cancel parameter-free.
    #
    # Under Reversed-Hierarchy Frugality every one of these cells has
    # the decider followed with p = (1-eps)*sigmoid(beta)+eps/2 in
    # [0.57, 0.64], so the metric sits near +0.11 -- a large, tight
    # gap away from Tallying's exact 0.
    vals = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)

        # Tally margin (strict feature-wise wins).
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin not in (-2, 0, 2):
            continue

        # Reversed-hierarchy decider: first strictly discriminating
        # feature when scanning features in REVERSE index order
        # (ascending true validity, since validities descend by index).
        v = 0
        for j in range(len(a) - 1, -1, -1):
            if a[j] > b[j]:
                v = 1
                break
            if b[j] > a[j]:
                v = -1
                break
        if v == 0:
            # Identical options (T15/T16): no decider, no information.
            continue

        resp = float(row["response"])  # 0 = chose A, 1 = chose B
        # +0.5 if subject followed the reversed decider, -0.5 otherwise.
        vals.append(v * (0.5 - resp))

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```
outcome: evaluation failed (SyntaxError: invalid syntax (<string>, line 1))

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Contrast metric built from two parameter-free anchors:\n    #\n    # (1) Margin-0 cells with a reversed-cascade decider (T5, T6, T11, T16).\n    #     Tallying predicts p(A) = 0.5 EXACTLY on any tie trial, for any\n    #     (beta, epsilon), so the signed follow-the-decider score has\n    #     expectation exactly 0. The reversed-hierarchy model follows its\n    #     decider with q = (1-eps)*sigmoid(beta) + eps/2 in [0.57, 0.64],\n    #     so its expected score is 2q-1 in [0.14, 0.28].\n    #\n    # (2) Disagreement cells (T1, T2, T10, T14): nonzero tally margin whose\n    #     sign OPPOSES the reversed decider. Tallying follows the margin\n    #     with p >= 0.5 for any beta > 0 (expected signed score 2p-1 >= 0);\n    #     the reversed model follows the decider, i.e. follows the tally\n    #     with 1-q <= 0.43 (expected signed score 1-2q < 0).\n    #\n    # metric = mean(follow-decider score on margin-0 cells)\n    #        - mean(follow-tally score on disagreement cells)\n    #\n    # Tallying: 0 - (2p-1) <= 0 (typically around -0.4 to -0.9).\n    # Reversed: (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56] > 0.\n    # The two theories land on opposite sides of zero with a large gap.\n    s1 = []\n    s2 = []\n    n_rows = len(data)\n    for i in range(n_rows):\n        row = data.iloc[i]\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        n = len(a)\n\n        # Tally margin: strict feature-wise wins for A minus for B.\n        margin = 0\n        for j in range(n):\n            if a[j] > b[j]:\n                margin += 1\n            elif b[j] > a[j]:\n                margin -= 1\n\n        # Reversed-hierarchy decider: first strictly discriminating\n        # feature scanning from the LAST index down (ascending true\n        # validity, since validities descend by index).\n        dec = 0\n        for j in range(n - 1, -1, -1):\n            if a[j] > b[j]:\n                dec = 1\n                break\n            if b[j] > a[j]:\n                dec = -1\n                break\n        if dec == 0:\n            # Identical options: no decider, no information.\n            continue\n\n        resp = int(row[\"response\"])\n        chose_a = (resp == 0)\n\n        if margin == 0:\n            # Knife-edge cell: tally is exactly 0.5 under Tallying.\n            follow_dec = (dec == 1 and chose_a) or (dec == -1 and not chose_a)\n            s1.append(1.0 if follow_dec else -1.0)\n        elif (margin > 0 and dec == -1) or (margin < 0 and dec == 1):\n            # Disagreement cell: tally direction opposes the decider.\n            follow_tally = (margin > 0 and chose_a) or (margin < 0 and not chose_a)\n            s2.append(1.0 if follow_tally else -1.0)\n\n    m1 = float(np.mean(s1)) if len(s1) > 0 else 0.0\n    m2 = float(np.mean(s2)) if len(s2) > 0 else 0.0\n    return m1 - m2",
  "rationale": "The metric combines the two strongest, parameter-robust contrasts in this design. ANCHOR 1 (knife-edge cells T5, T6, T11, T16): these have tally margin exactly 0 but a well-defined reversed-cascade decider. Tallying's softmax over equal tallies is exactly uniform for ANY (beta, epsilon), so the signed follow-the-decider score has expectation exactly 0 there - a hard parameter-free anchor. The competing model follows its decider with q = (1-eps)*sigmoid(beta) + eps/2, pinned by its parameter bounds to [0.57, 0.64], giving expected score 2q-1 in [0.14, 0.28]. ANCHOR 2 (disagreement cells T1, T2, T10, T14): here the tally margin (|m| = 4 or 2) points OPPOSITE to the reversed decider. Tallying follows the margin with p >= 0.5 for any beta > 0 (score 2p-1 >= 0, typically 0.5-0.9), while the reversed model follows its decider and thus the tally only with 1-q <= 0.43 (score 1-2q < 0). The metric subtracts the two signed scores, so both anchors push in the SAME direction: Tallying yields a negative value (0 minus a nonnegative quantity), while the reversed model yields (2q-1) - (1-2q) = 4q-2 in [0.28, 0.56], strictly positive. Even at the worst parameter corners of both theories the gap is at least ~0.3, and at typical parameters it is ~1.0. Each subject contributes 8 cells x 6 reps = 48 trials (24 per anchor), keeping per-subject SD around 0.25-0.30, so at N = 25 the Welch t-statistic is roughly 4-15 depending on the simulated parameters - comfortably beyond the 0.01 significance threshold. The metric works identically on pooled and single-subject data, uses only scalar groupings (no hashing of list cells), and is pure ASCII with no exotic syntax."
}
```

## Usage

```json
{
  "input_tokens": 5800,
  "output_tokens": 10106,
  "reasoning_tokens": 8676,
  "provider": "Sail Research"
}
```
