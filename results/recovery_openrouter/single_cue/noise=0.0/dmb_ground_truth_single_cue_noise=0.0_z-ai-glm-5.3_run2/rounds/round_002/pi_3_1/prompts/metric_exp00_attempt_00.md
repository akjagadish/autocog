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
**Validities (n_features=6):** [0.7, 0.9, 0.6, 0.95, 0.75, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 2: A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 13: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 15: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 16: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying, while adding a novel axis absent from prior designs: a SCRAMBLED (non-monotone) validity arrangement that makes the misbound hierarchy a non-trivial permutation, enabling a 'subjective-hierarchy ladder' and a 'decider-validity invariance' test. SETUP: 6 features, validities [0.70, 0.90, 0.60, 0.95, 0.75, 0.55] (positions f0..f5). The misbound subjective weights are w = val reversed = [0.55, 0.75, 0.95, 0.60, 0.90, 0.70], so the misbinding cascade consults f2 -> f4 -> f1 -> f5 -> f3 -> f0. Crucially, the misbound model's FIRST cue is Expert 3 (advertised only 60%), while the truly best expert (95%, f3) is consulted 5th. Tallying ignores validities entirely, so its predictions depend only on the tally margin m = a_wins - b_wins: p(A) = (1-eps)*sigmoid(beta*m) + eps/2. The misbinding model's predictions depend only on the decider: constant p = (1-eps)*sigmoid(beta)+eps/2 in [0.63, 0.78] toward the first cue in the order f2,f4,f1,f5,f3,f0 that discriminates. SIGNATURE 1 - margin sweep with fixed decider (T1-T6): Expert 3 (f2) always favors A while the margin sweeps -4, -2, 0, +1, +3 (T1, T3, T4, T5, T6), plus the A/B mirror at margin +4 with decider->B (T2). Misbinding TTB predicts a FLAT p(A) in [0.63, 0.78] on all six; Tallying predicts a strictly graded psychometric curve that passes through EXACTLY 0.5 at T4 (margin 0, parameter-free knife-edge) and stays below 0.5 at negative margins (T1: Tallying <= 0.45 for ANY beta in [0.1, 20], eps in [0, 0.5], while misbinding TTB >= 0.63 - non-overlapping prediction intervals on this single cell). SIGNATURE 2 - extreme-margin matched pair (T2 vs T15): both have margin +4, but the decider is f2->B on T2 and f2->A on T15. Tallying predicts IDENTICAL p(A) on the two cells for ANY (beta, eps) - a parameter-free prediction - while misbinding TTB predicts ~0.22-0.37 on T2 vs ~0.63-0.78 on T15, opposite sides of 0.5. SIGNATURE 3 - the subjective-hierarchy ladder at margin 0 (T4, T7-T11): six knife-edge cells, each with exactly one cue favoring each option, so Tallying is pinned to EXACTLY 0.50 on every rung; any consistent preference on any rung falsifies Tallying regardless of parameters. The rungs pit adjacent cues of the MISBOUND order against each other (f2 vs f4; f2 vs f1; f4 vs f1; f5 vs f3 and its mirror; f3 vs f0), so subjects' choices trace out their subjective priority order. Because the arrangement is scrambled, five of six rungs also place standard validity-respecting TTB on the OPPOSITE side (e.g., T9/T10: the 55% expert f5 vs the 95% expert f3 - misbinding follows the 55% expert, standard TTB the 95% expert), making the ladder a simultaneous three-way separation: misbinding TTB ~0.7 toward its decider, Tallying exactly 0.5, standard TTB often the reverse. SIGNATURE 4 - decider-validity invariance at fixed margin (T5, T12, T13, T14): four margin-+1 cells whose misbound deciders are advertised at 60%, 75%, 90%, and 55% respectively. Misbinding TTB predicts the SAME p(A) on all four (the follow rate cannot depend on the decider's advertised accuracy); any monotone effect of decider validity falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), but they pin the flat level as decider-independent, and the within-decider contrast T4 vs T5 (margin 0 vs +1, same decider) tests Tallying's strict margin effect where misbinding TTB predicts none. SIGNATURE 5 - consistency ceiling (T15): a margin-+4 agreement cell where only the 95% expert dissents. Misbinding TTB is capped at ~0.78 by its parameter ranges (beta <= 1.2, eps >= 0.08); human consistency above ~0.80 here falsifies the advocated theory at its boundary while Tallying accommodates it. CONTROLS: T16 is a full tie (both theories: exactly 0.5, anchoring lapse and side bias); T1/T2 are exact A/B mirrors; T10 and T8 provide decider->B cells at margin 0 to rule out side bias in the ladder. GLOBAL PREDICTED PROFILES: misbinding TTB produces a two-level step function over the 16 cells (~0.7 toward the misbound decider on all 15 discriminating trials, exactly 0.5 on T16), invariant to margin, cascade depth, and decider validity; Tallying produces a monotone psychometric function of margin with exact 0.5 on the six margin-0 cells and identical values within margin levels (T2 = T15; T5 = T12 = T13 = T14). The profiles are structurally disjoint (e.g., misbinding says p(T1) = p(T6); Tallying says p(T1) < 0.45 < p(T6)), so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Parameters:**
- beta: [0.8, 1.2]
- epsilon: [0.08, 0.14]
- kappa: {1.0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
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
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
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

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Decider-follow rate on the margin-0 DISCRIMINATING trials (the\n    # \"subjective-hierarchy ladder\": T4, T7-T11 in the design).\n    # On these trials the tally is exactly balanced, so Tallying is\n    # pinned to p = 0.50 toward either side for ANY (beta, epsilon)\n    # (softmax over equal tallies is uniform, lapse is uniform).\n    # Misbinding TTB instead follows its (reversed-hierarchy) first\n    # discriminating cue with p = (1-eps)*sigmoid(beta) + eps/2,\n    # which is >= 0.63 over the stated parameter ranges. The metric\n    # is the proportion of choices that agree with the misbound\n    # decider on exactly those trials: ~0.71 under misbinding TTB,\n    # ~0.50 under Tallying.\n    val = np.array([0.7, 0.9, 0.6, 0.95, 0.75, 0.6], dtype=float)\n    # kappa = 1 misbinding: subjective weights are the validity list\n    # bound in reverse to feature positions.\n    w = val[::-1]\n    cue_order = np.argsort(-w, kind=\"stable\")  # f2, f4, f1, f5, f3, f0\n\n    a = np.array([list(x) for x in data['option_a_ratings']], dtype=float)\n    b = np.array([list(x) for x in data['option_b_ratings']], dtype=float)\n    resp = np.asarray(data['response'], dtype=int).ravel()\n\n    if a.shape[0] == 0:\n        return 0.5\n\n    a_wins = (a > b).sum(axis=1)\n    b_wins = (b > a).sum(axis=1)\n\n    # Margin-0 trials that still have at least one discriminating cue\n    # (excludes the full tie T16, where both theories say 0.5).\n    mask = (a_wins == b_wins) & ((a_wins + b_wins) > 0)\n    idx = np.where(mask)[0]\n\n    if len(idx) == 0:\n        return 0.5\n\n    follow = []\n    for i in idx:\n        decider = None\n        for j in cue_order:\n            if a[i, j] > b[i, j]:\n                decider = 0\n                break\n            if b[i, j] > a[i, j]:\n                decider = 1\n                break\n        if decider is None:\n            continue  # shouldn't happen on discriminating trials\n        follow.append(1 if resp[i] == decider else 0)\n\n    if len(follow) == 0:\n        return 0.5\n    return float(np.mean(follow))",
  "rationale": "The metric exploits the design's cleanest parameter-free knife edge: the six margin-0 discriminating trials (T4, T7, T8, T9, T10, T11). On these trials the two options win on an equal number of features, so Tallying's softmax over equal tallies is exactly uniform for ANY beta in [0.1, 20] and ANY epsilon in [0, 0.5] \u2014 its follow rate toward any designated side is pinned at exactly 0.50 with only binomial sampling noise (36 trials per subject, variance 0.25/36 ~ 0.0069, and zero parameter heterogeneity). Misbinding TTB, in contrast, consults its reversed hierarchy (f2 -> f4 -> f1 -> f5 -> f3 -> f0), finds a strictly discriminating cue on every one of these trials, and follows it with p = (1-eps)*sigmoid(beta) + eps/2, which lies in [0.63, 0.78] over the stated parameter ranges (mean ~0.71). The decider side is computed directly from the data using the same reversed-validity argsort the advocated model uses, so the metric needs no fitted parameters. Expected separation: ~0.71 (self) vs ~0.50 (adv), a mean gap of ~0.21, with per-subject variances of ~0.0076 (self: binomial plus mild beta/eps heterogeneity) and ~0.0069 (adv: pure binomial). At N=25 this gives Welch t ~ 8-9, far below the alpha=0.01 threshold. The metric works identically on the pooled DataFrame and on a single subject's slice (a proportion over that subject's 36 ladder trials), so the between-subject variance is well-defined. It is not trivially true for the advocated theory in the parameter-free sense that matters here: the competing theory makes an exact point prediction (0.50) on these cells that the advocated theory's noise floor (0.63+) cannot reach, making the contrast robust to any parameter fitting on either side."
}
```

## Usage

```json
{
  "input_tokens": 5249,
  "output_tokens": 5921,
  "reasoning_tokens": 4792,
  "provider": "Sail Research"
}
```
