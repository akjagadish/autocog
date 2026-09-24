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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=13):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]

**Rationale:** The design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice is determined solely by the single highest-validity discriminating cue, whereas Tallying integrates the sign of every feature comparison. I use 5 features with a clear validity hierarchy [0.90, 0.80, 0.70, 0.60, 0.55] so the cue cascade is unambiguous. The trial set is built to produce a QUANTITATIVE dissociation, not just a qualitative one. (1) Conflict trials T1-T3: option A wins only on the top cue (validity 0.90) while B wins on progressively fewer lower cues (tally margins -3, -2, -1 for A). TTB predicts p(A) is IDENTICAL across T1, T2, T3 (a constant (1-eps)*sigmoid(beta)+eps/2, since the TTB score is always [1,0]); Tallying predicts a strictly graded p(A) that increases with the tally margin (sigmoid(-3*beta), sigmoid(-2*beta), sigmoid(-beta)). A flat vs. graded profile across these three trial types is the central quantitative signature. (2) Mirror conflict trials T4-T6 test the same dissociation in the opposite direction (TTB->B, Tallying->A), ruling out side-bias explanations. (3) Tally-tie trials T7-T8: the tally is exactly 1-1, so Tallying is forced to 50/50 regardless of beta, while TTB is near-deterministic (top cue decides). These trials alone can falsify Tallying at the aggregate level: any consistent preference is TTB-only. (4) Cascade-depth trials T11-T12: the top cue ties and the SECOND cue (0.80) discriminates against the tally majority — TTB follows cue 2, Tallying follows the 3-vs-2 count. This confirms subjects respect the validity ordering, not just 'any high cue wins'. (5) Agreement trials T9-T10 (both theories favor the same option, margin +1) and a full-tie trial T13 (both predict exactly 50/50) serve as controls: they anchor the lapse rate and verify that neither theory is needed where both agree. Crucially, TTB predicts a single constant choice probability across all five TTB->A trial types (T1, T2, T3, T7, T9), whereas Tallying predicts five distinct probabilities spanning from well below 0.5 to well above 0.5 — model comparison on the response profile across these cells cleanly separates the theories even after fitting beta and epsilon to each. With 13 unique pairs and K=7 repetitions (91 trials), each cell is sampled enough to estimate choice proportions per subject.

**Computed schedule:** 13 unique pairs × 7 reps = 91 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    margins = []\n    matches = []\n    for a_cell, b_cell, resp in zip(\n        data[\"option_a_ratings\"],\n        data[\"option_b_ratings\"],\n        data[\"response\"],\n    ):\n        a = np.asarray(a_cell, dtype=float).ravel()\n        b = np.asarray(b_cell, dtype=float).ravel()\n        if a.shape[0] != b.shape[0] or a.shape[0] != validities.shape[0]:\n            continue\n        diff = a - b\n\n        # Take-The-Best cascade: first discriminating cue in validity order.\n        ttb_winner = -1\n        for j in cue_order:\n            if diff[j] > 0:\n                ttb_winner = 0\n                break\n            if diff[j] < 0:\n                ttb_winner = 1\n                break\n        if ttb_winner < 0:\n            # Full tie (trial 13): no TTB winner, uninformative for both theories.\n            continue\n\n        a_wins = float(np.sum(a > b))\n        b_wins = float(np.sum(b > a))\n        # Tally margin of the TTB-favored option (positive = tally agrees with TTB).\n        if ttb_winner == 0:\n            margin = a_wins - b_wins\n        else:\n            margin = b_wins - a_wins\n\n        r = int(resp)\n        match = 1.0 if r == ttb_winner else 0.0\n        margins.append(margin)\n        matches.append(match)\n\n    if not margins:\n        return 0.0\n\n    x = np.asarray(margins, dtype=float)\n    y = np.asarray(matches, dtype=float)\n\n    # (1) LEVEL: P(choosing the TTB-favored option) on trials where the\n    # tally does NOT favor it (tally majority against, or exact tally tie).\n    low = x <= 0\n    level = float(np.mean(y[low])) if np.any(low) else 0.0\n\n    # (2) SLOPE: OLS slope of the match indicator on the tally margin,\n    # across all trials with a TTB winner (margins -3, -2, -1, 0, +1).\n    vx = float(np.var(x))\n    if len(x) > 1 and vx > 0:\n        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx)\n    else:\n        slope = 0.0\n\n    return level - slope",
  "rationale": "The metric is a theory-opposed choice-profile contrast built from two TTB-diagnostic signatures in this design, combined into one scalar (level minus slope) so that both push in the same direction.\n\nConstruction: for every trial I reconstruct, from the rating vectors and the known validities [0.9, 0.8, 0.7, 0.6, 0.5], (i) the TTB-favored option (first discriminating cue in validity order) and (ii) the tally margin of that option (its feature-win count minus the opponent's). Trials with no discriminating cue (T13) are dropped. Then:\n\n(1) LEVEL = P(choose the TTB-favored option | tally does not favor it), i.e. over the 10 conflict/tie trial types (margins -3, -2, -1, 0). Under TTB every one of these trials has the same TTB score, so the level equals q = (1-eps)*sigmoid(beta) + eps/2, typically 0.6-0.95. Under Tallying these cells have probabilities sigmoid(-3b), sigmoid(-2b), sigmoid(-b) and 0.5, so the level is dragged well below 0.5 (e.g. 0.32 at beta=0.5, 0.13 at beta=2).\n\n(2) SLOPE = OLS slope of the match indicator on the tally margin across all discriminating trials (margins -3..+1). Under TTB the choice probability is flat in the margin, so the slope is exactly 0 in expectation for ANY (beta, epsilon) \u2014 this term is parameter-free for my theory. Under Tallying the probability strictly increases with the margin, giving a positive slope (e.g. 0.11 at beta=0.5, 0.22 at beta=2).\n\nSubtracting the slope from the level makes both deviations add: TTB yields approximately q - 0 (>= 0.5), while Tallying yields level - slope, which is pushed further down (0.21 at beta=0.5, -0.09 at beta=2, 0.44 even at beta=0.1). The between-theory gap is therefore large across the whole plausible parameter space (0.09 at beta=0.1/eps=0, 0.41 at beta=0.5, >0.9 at beta=2), while per-subject sampling noise is small: the statistic is a fixed linear contrast over ~84 trials per subject, with binomial standard error around 0.05-0.08. At N=25 this gives t > 3.9 even in the near-degenerate beta=0.1, eps=0 case, and t > 10 for any beta >= 0.3. The mirror trials (T4-T6, T8, T10, T12) are included with their signs flipped, so pure A/B side biases cancel. The slope term also protects against parameter heterogeneity across subjects: it is exactly zero for TTB regardless of each subject's beta and epsilon, so it cannot inflate my theory's between-subject variance."
}
```

## Usage

```json
{
  "input_tokens": 4124,
  "output_tokens": 19041,
  "reasoning_tokens": 17684,
  "provider": "Sail Research"
}
```
