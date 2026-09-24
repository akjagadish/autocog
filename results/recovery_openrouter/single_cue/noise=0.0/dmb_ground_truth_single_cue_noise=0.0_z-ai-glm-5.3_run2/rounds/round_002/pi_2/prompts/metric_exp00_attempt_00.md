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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 9: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** NOVEL AXIS - THE CONCORDANCE LADDER: every prior design dissociated Tallying from TTB-family models using CONFLICT trials (some cues favor each side). This design's centerpiece is a family of zero-conflict trials where every discriminating cue favors the same option and only the NUMBER of supporting cues varies (ties fill the rest). Tallying's output depends only on the margin m = a_wins - b_wins, so it predicts a strictly GRADED psychometric curve p(A) = (1-eps)*sigmoid(beta*m) + eps/2 even when no cue disagrees — climbing from ~0.62 at m=+1 to ~0.97 at m=+6 for moderate beta. Reversed-Hierarchy Frugality is a one-reason model: on every concordance trial the first discriminating cue in its misbound cascade decides, so it predicts ONE CONSTANT p(A) = (1-eps)*sigmoid(beta)+eps/2 across the entire ladder — and, critically, its fixed noise regime (beta in [0.8,1.2], eps in [0.08,0.14]) mathematically CAPS its follow probability at ~0.75. Human accuracy above ~0.80 on high-rung concordance cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. SETUP: 6 features with ASCENDING validities [0.55, 0.60, 0.65, 0.75, 0.85, 0.95] (f0..f5) — an arrangement no prior design used. The misbound hierarchy w = val reversed = [0.95, ..., 0.55] makes the reversed cascade consult f0 first: the FIRST-listed, LOWEST-validity (55%) expert is the misbinding model's most trusted cue, and the 95% expert is consulted last. SIGNATURE 1 - CONCORDANCE LADDER (T1-T5): A weakly dominates B, margins +1, +2, +4, +5, +6; the misbound decider sweeps across f5 (95%), f4 (85%), f2 (65%), f1 (60%), f0 (55%). Misbinding TTB: flat ~0.70 on all five rungs (jointly a cascade-depth- AND decider-validity-invariance claim); Tallying: strictly increasing, exceeding the misbinding cap on the top rungs. SIGNATURE 2 - SINGLE-CELL NON-OVERLAP CONFLICT SWEEP (T6-T8, T9-T11): the 55% expert f0 always decides for A (T6: m=-4, T7: m=-2, T8: m=0) or always for B (T9: m=+4, T10: m=+2, T11: m=0). On T6/T7 Tallying is mathematically BELOW 0.50 for ANY parameters (negative margin), while misbinding is at or above 0.663 — the prediction intervals are disjoint, so each cell alone falsifies one theory. On T9/T10 the inversion is complete (Tallying > 0.50, misbinding <= 0.337). T6 vs T9 are exact A/B mirrors ruling out side bias. SIGNATURE 3 - KNIFE-EDGE FAMILY (T8, T11, T13, T15): four margin-0 cells (1-1, 1-1, 2-2, 3-3) where Tallying is pinned to EXACTLY 0.50 independent of (beta, epsilon) — any consistent preference on any of these cells falsifies Tallying — while misbinding predicts 0.70/0.30 splits (decider f0) at every tally composition, i.e., invariance to how many cues sit on each side. SIGNATURE 4 - ISO-MARGIN OPPOSITE-DECIDER TRIPLETS (parameter-free for Tallying): {T2, T10, T12} all have m=+2 — Tallying predicts IDENTICAL p(A) on all three for any parameters; misbinding predicts 0.70 (T2, decider f4->A), 0.70 (T12, decider f0->A), 0.30 (T10, decider f0->B). {T3, T9} both m=+4: Tallying identical; misbinding 0.70 vs 0.30. {T1, T14} both m=+1 with deciders at opposite ends of the subjective hierarchy (f5, 95% vs f0, 55%): BOTH theories predict identical p(A) here — a shared control that pins misbinding's flat level and tests its claim that the decider's advertised accuracy is irrelevant. T16 is the full-tie control (both theories: exactly 0.50, anchoring lapse and side bias). GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~0.70 on the eleven decider->A cells including ALL ladder rungs and ALL negative-margin conflicts; ~0.30 on the four decider->B cells; 0.50 on the tie), invariant to margin, tally composition, cascade depth, and decider validity. Tallying emits a monotone psychometric function of margin with exact 0.50 on the four margin-0 cells, exact equality within margin levels (T2=T10=T12; T3=T9; T1=T14), and values above the misbinding ceiling on T4/T5. The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5)=p(T6); Tallying says p(T6)<0.50<p(T1)<p(T5) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Knife-edge signed contrast on the four margin-0 discriminating cells.\n\n    Tallying predicts p(A) = 0.50 EXACTLY on every margin-0 cell, for any\n    (beta, epsilon): equal tallies give an exactly uniform softmax. The\n    reversed-hierarchy one-reason model, in contrast, lets a single cue\n    decide these same cells, producing a systematic ~0.70/0.30 split whose\n    sign is set by which option wins on the cascade's first cue.\n\n    Metric = mean p(choose A | T11, T13) - mean p(choose A | T8, T15),\n    where T11/T13 are the margin-0 cells whose cascade-first cue favors A\n    and T8/T15 are the margin-0 cells whose cascade-first cue favors B\n    (under the design's validities [0.95, 0.85, 0.75, 0.65, 0.6, 0.6]\n    reversed -> f5 consulted first).\n\n    Tallying: expectation exactly 0 for ANY parameter values.\n    Misbinding TTB: expectation ~ +/-(2p-1) in [0.33, 0.49].\n    \"\"\"\n    def to_key(v):\n        return tuple([int(x) for x in v])\n\n    a_keys = data[\"option_a_ratings\"].apply(to_key)\n    b_keys = data[\"option_b_ratings\"].apply(to_key)\n    chose_a = 1.0 - data[\"response\"].astype(float)  # response 0 = A\n\n    # Margin-0 (knife-edge) cells, identified by their rating vectors.\n    # Cascade-first cue (f5) favors A here:\n    hi_cells = [\n        ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)),  # T11: 1-1 split, f5->A\n        ((0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 0)),  # T13: 2-2 split, f5->A\n    ]\n    # Cascade-first cue (f5) favors B here:\n    lo_cells = [\n        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),  # T8:  1-1 split, f5->B\n        ((1, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)),  # T15: 3-3 split, f5->B\n    ]\n\n    def cell_pa(ak, bk):\n        mask = (a_keys == ak) & (b_keys == bk)\n        if int(mask.sum()) == 0:\n            return np.nan\n        return float(chose_a[mask].mean())\n\n    hi = [cell_pa(ak, bk) for ak, bk in hi_cells]\n    lo = [cell_pa(ak, bk) for ak, bk in lo_cells]\n    hi = [v for v in hi if not np.isnan(v)]\n    lo = [v for v in lo if not np.isnan(v)]\n    if not hi or not lo:\n        return 0.0\n\n    return float(np.mean(hi) - np.mean(lo))",
  "rationale": "CORE CONTRAST (Signature 3, knife-edge family). On the four margin-0 discriminating cells (T8, T11, T13, T15) the two options win on exactly as many features, so Tallying's softmax over equal tallies is EXACTLY uniform: p(A) = 0.50 for every (beta, epsilon) in its parameter space \u2014 a parameter-free prediction. The misbinding one-reason model has no such constraint: its reversed cascade lets the first discriminating cue decide, so these cells split ~0.70/0.30 (follow probability p = (1-eps)*sigmoid(beta)+eps/2 is bounded in [0.66, 0.75] by its fixed noise regime, so the signed contrast is bounded away from zero at 2p-1 in [0.33, 0.49]). I sign the contrast along the cascade's first cue under the design's stated validities ([0.95, ..., 0.6] reversed -> f5 consulted first): T11/T13 are the m=0 cells where f5 favors A, T8/T15 those where f5 favors B. Metric = mean p(A | T11,T13) - mean p(A | T8,T15). Expected values: Tallying = 0 exactly; misbinding = ~0.41. NOTE ON ROBUSTNESS: (1) the contrast cancels pure side bias (each side contains an A-favored and a B-favored cell); (2) even if the validity ordering were reversed, the misbinding prediction merely flips sign (~-0.41) and the two-sided Welch test is unaffected; (3) each subject contributes 12 trials per side (2 cells x 6 reps), so per-subject sampling sd is ~0.20, giving between-subject variance ~0.035-0.042 and a Welch t of roughly 0.41 / sqrt(0.077/25) ~ 7 at N=25 \u2014 comfortably below alpha=0.01. (4) Unlike a ladder-slope metric, this contrast cannot collapse: Tallying's value is pinned to 0 in expectation regardless of how extreme its sampled beta/epsilon are, while misbinding's is bounded away from 0 by its fixed noise regime."
}
```

## Usage

```json
{
  "input_tokens": 5144,
  "output_tokens": 13448,
  "reasoning_tokens": 12131,
  "provider": "Sail Research"
}
```
