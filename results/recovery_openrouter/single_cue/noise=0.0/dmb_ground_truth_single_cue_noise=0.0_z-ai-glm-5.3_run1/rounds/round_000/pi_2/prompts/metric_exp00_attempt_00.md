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
**Validities (n_features=6):** [0.92, 0.84, 0.76, 0.68, 0.6, 0.62]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 14: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** This design implements a fully crossed TWO-FACTOR double dissociation between Tallying and Take-The-Best (TTB), which is structurally stronger than a one-dimensional dose-response: each theory predicts a main effect of exactly one factor and a PARAMETER-FREE ZERO effect of the other. Factor 1 is the tally margin d = (A's cue wins) - (B's cue wins), the only thing Tallying's prediction depends on: p_Tally(A) = (1-eps)*sigmoid(beta*d) + eps/2. Factor 2 is the direction s of the highest-validity discriminating cue, the only thing TTB's prediction depends on: p_TTB(A) = (1-eps)*sigmoid(beta*s) + eps/2. The 16 pairs cross d in {-4,-3,-2,-1,0,+1,+2,+3,+4} with s in {+1,-1}. KEY QUANTITATIVE PREDICTIONS: (1) WITHIN-COLUMN FLATNESS falsifies TTB with no free parameters: trials P1-P6, P13, P15 all have s=+1 (top cue v=0.92 favors A), so TTB predicts the IDENTICAL p(A) on all eight for every (beta, eps). Yet d ranges from +4 (P15: A wins 5 of 6 cues) to -4 (P6: A wins ONLY the top cue, B wins the other five). Tallying predicts p(A) falling from ~(1-eps) to ~eps/2 across these same trials. If subjects' p(A) on P6 is any lower than on P15, TTB is falsified outright -- no parameters can rescue a flat function that is observed to slope. The s=-1 column (P7-P12, P14, P16) mirrors this. (2) WITHIN-ROW FLATNESS falsifies Tallying with no free parameters: matched pairs (P2,P11), (P3,P10), (P4,P9), (P5,P8) share the same d with s flipped, so Tallying predicts EXACTLY equal p(A) within each pair for all (beta, eps), while TTB predicts a swing of (1-eps)*(2*sigmoid(beta)-1) -- up to nearly 1.0. The d=0 knife pair (P13, P14) is even sharper: Tallying must predict exactly 0.50 on both (tied tallies give a uniform softmax for any beta, and the lapse also injects uniform), while TTB predicts symmetric deviations toward the top-cue winner. (3) GRADED DOSE-RESPONSE: within each column, Tallying predicts a strictly monotone p(A) in d (sigmoid(beta*d) for d = -4..+4), giving a 9-point psychometric curve at fixed TTB output; TTB predicts a horizontal line. Conversely, across columns at fixed d, TTB predicts the full sigmoid swing while Tallying predicts zero. (4) EVEN THE AGREEMENT TRIALS SEPARATE QUANTITATIVELY: on P15 (d=+4, s=+1) both theories choose A, but Tallying predicts sigmoid(4*beta) vs TTB's sigmoid(beta) -- e.g., at beta=1, 0.982 vs 0.731 -- so confidence on unambiguous trials discriminates the theories too. As an adversarial touch against my own advocated theory, cues are displayed in descending-validity order (0.92 first), which maximally favors TTB's serial search (display order matches its search order) while being irrelevant to Tallying; if Tallying still wins under TTB's best-case display, the evidence is stronger. 16 unique pairs x K=6 = 96 trials; A/B sides are balanced (8 trials TTB-favors-A, 8 TTB-favors-B; 7 tally-favors-each-side, 2 tied), P15/P16 and P6/P12 are label-swapped mirrors that control side bias, no all-tie trials, and every pair requires genuine cue comparison.

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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

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

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Dual-dissociation regression metric.\n    # Per trial, compute:\n    #   d = tally margin  = (#features A wins) - (#features B wins)\n    #   s = top-cue direction = sign(A[0] - B[0])   (feature 0 has the\n    #       highest validity 0.92, and every designed pair discriminates on it)\n    # Regress chose-A on [1, d, s] via OLS and return b_d - b_s.\n    # Tallying: choice probability depends only on d  -> b_d > 0, b_s = 0 -> metric > 0.\n    # TTB: choice probability depends only on s       -> b_d = 0, b_s > 0 -> metric < 0.\n    # The two theories push this scalar in opposite directions, and the\n    # partial regression cleanly removes the design's d-s correlation.\n    df = data.reset_index(drop=True)\n\n    A = np.stack([np.asarray(list(r), dtype=float) for r in df[\"option_a_ratings\"].values])\n    B = np.stack([np.asarray(list(r), dtype=float) for r in df[\"option_b_ratings\"].values])\n\n    # Tally margin: strict feature-wise wins; ties count for neither side.\n    d = (A > B).sum(axis=1).astype(float) - (B > A).sum(axis=1).astype(float)\n\n    # Direction of the highest-validity (top) cue, which always discriminates\n    # in this design: +1 if it favors A, -1 if it favors B.\n    s = np.sign(A[:, 0] - B[:, 0])\n\n    # Chose-A indicator (response: 0 = A, 1 = B).\n    y = 1.0 - df[\"response\"].to_numpy(dtype=float)\n\n    X = np.column_stack([np.ones(len(y)), d, s])\n    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)\n    b_d = float(coef[1])\n    b_s = float(coef[2])\n\n    # Tallying's signature (slope in d) minus TTB's signature (slope in s).\n    return b_d - b_s",
  "rationale": "The design factorially crosses the tally margin d (the ONLY input Tallying's prediction depends on) with the top-cue direction s (the ONLY input TTB's prediction depends on). I exploit both main effects simultaneously with a single OLS coefficient contrast: regress the chose-A indicator on [intercept, d, s] over all trials and return b_d - b_s. Under Tallying, p(A) = (1-eps)*sigmoid(beta*d) + eps/2, so the partial slope on d is strictly positive (up to ~0.125 per unit d when beta is large) while the partial slope on s is exactly 0 \u2014 s adds nothing once d is controlled. Under TTB, p(A) = (1-eps)*sigmoid(beta*s) + eps/2, so the partial slope on s is strictly positive (up to 0.5) while the partial slope on d is exactly 0 \u2014 within a fixed s-column TTB is flat in d by construction, and no (beta, eps) can change that. Hence the metric is positive under Tallying and negative under TTB, with a gap equal to the SUM of the two theories' signature effect sizes, making it robust: it discriminates even if one theory happens to be simulated with weak parameters, as long as the other shows its signature. Including s as a regressor also removes the mild d-s collinearity built into the 16 pairs (corr ~ 0.3), so the d-coefficient is a clean within-column slope rather than a confounded marginal. Per-subject computation is well-conditioned: every subject sees all 16 pair types x 6 reps (96 trials), d spans -4..+4 within each s-column, and the regressors are balanced (mean d = 0, eight s=+1 and eight s=-1 types). Per-subject sampling noise of b_d is small (~0.02) because it borrows strength across all nine d levels, and noise on b_s (~0.11 worst case, p=0.5) is dwarfed by TTB's s-effect (~0.5 at moderate beta). At N=25 the Welch t-test on the pooled point estimates with these between-subject variances yields a highly significant separation in every non-degenerate parameter regime."
}
```

## Usage

```json
{
  "input_tokens": 4621,
  "output_tokens": 14773,
  "reasoning_tokens": 13749,
  "provider": "Sail Research"
}
```
