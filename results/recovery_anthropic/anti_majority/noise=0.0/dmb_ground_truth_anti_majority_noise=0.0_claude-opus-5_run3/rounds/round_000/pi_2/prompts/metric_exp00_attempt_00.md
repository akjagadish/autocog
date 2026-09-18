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
**Validities (n_features=6):** [0.6, 0.93, 0.55, 0.8, 0.68, 0.57]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 4: A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 10: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 11: A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 12: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  trial 13: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** Goal: separate Take-The-Best (one-reason, validity-ordered, all-or-none confidence) from Tallying (unweighted vote count, confidence graded by the tally margin) using a design that is deliberately different from the previously explored one in three ways: (i) six cues instead of five, (ii) the validities are presented in a SCRAMBLED (non-descending) display order, so the top cue is Expert 2 (0.93) and the worst is Expert 3 (0.55) \-- this prevents a simple 'use the first-listed expert' strategy from mimicking TTB and forces genuine validity re-sorting, and (iii) the tally margins on conflict trials are pushed to a wider range (1, 3, 5 cue-wins against the top cue).

Validities: [0.60, 0.93, 0.55, 0.80, 0.68, 0.57] (display order). Descending validity order is therefore expert index 1 > 3 > 4 > 0 > 5 > 2.

Four trial families (16 unique pairs, K = 6, 96 trials), each mirrored across the A/B sides so that no response bias can imitate either model:

(1) DIRECTIONAL CONFLICTS with graded margin (pairs 1-6). The 0.93 cue favors one option while 2, 4, or 5 lower-validity cues favor the other (tally margins of 1, 3 and 4 against the top cue). TTB predicts the top-cue option with a CONSTANT probability sigmoid(beta) regardless of how lopsided the vote is; Tallying predicts the OPPOSITE option with probability increasing monotonically in the margin. These trials give opposite-sign predictions plus a within-model gradient, so a single conflict-choice curve identifies the model.

(2) TTB-DESCENT CONFLICT (pairs 7-8). The 0.93 cue ties, so TTB must descend to the 0.80 cue, which favors one option, while the remaining four cues (0.60, 0.68, 0.57, 0.55) all favor the other. Again opposite-sign predictions, but now the TTB-preferred option is supported by a non-top cue, testing whether the one-reason rule really tracks validity rank rather than salience of the best expert.

(3) TALLY-TIE / TTB-DECIDES (pairs 9-12). Each option wins an equal number of cues, so Tallying is exactly at chance (50/50 for every beta), while TTB decides confidently. In pairs 9-10 the deciding cue is the 0.93 expert; in pairs 11-12 the two highest cues tie and the tie-breaking cue is the mid-validity 0.68 expert. Systematic above-chance agreement with TTB here is impossible for Tallying at any parameter setting, and the pair 9-10 vs 11-12 contrast further checks TTB's claim of equal confidence irrespective of which cue happened to discriminate.

(4) MARGIN-SCALING CONTROLS, models agree in direction (pairs 13-16). Pairs 13-14: only the WORST cue (0.55) discriminates, everything else ties (tally margin 1). Pairs 15-16: unanimous 6-0 dominance (tally margin 6). Both models pick the same option, but TTB predicts identical choice consistency in both cases (sigmoid(beta) after a lapse), whereas Tallying predicts near-chance-plus-a-little on the 0.55-only trials and near-deterministic choice on the unanimous ones. This is a purely quantitative, direction-free dissociation, and it stresses the strongest counterintuitive commitment of TTB: a decision resting on a 55%-accurate expert should be made just as decisively as a 6-0 consensus.

Together the families deliver sign-level separation (families 1-2), a parameter-free zero-information prediction for Tallying (family 3), and a magnitude/confidence-gradient test (families 1 and 4).

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    validities = np.array([0.60, 0.93, 0.55, 0.80, 0.68, 0.57], dtype=float)\n\n    def ttb_winner(a, b, val):\n        order = np.argsort(-val, kind=\"stable\")\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return -1\n\n    hits = []\n    for _, row in data.iterrows():\n        a = np.asarray(row[\"option_a_ratings\"], dtype=float)\n        b = np.asarray(row[\"option_b_ratings\"], dtype=float)\n        n = a.shape[0]\n        val = validities if validities.shape[0] == n else np.linspace(0.9, 0.55, n)\n\n        a_wins = float(np.sum(a > b))\n        b_wins = float(np.sum(b > a))\n        margin = a_wins - b_wins\n        if margin > 0:\n            tally_win = 0\n        elif margin < 0:\n            tally_win = 1\n        else:\n            continue  # tally undecided: excluded\n\n        tw = ttb_winner(a, b, val)\n        if tw < 0 or tw == tally_win:\n            continue  # only trials where the two rules point opposite ways\n        if abs(margin) < 3:\n            continue  # keep only large-margin conflicts (strong tally signal)\n\n        hits.append(1.0 if int(row[\"response\"]) == tally_win else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The design contains conflict trials where the top-validity cue (or, when it ties, the 0.80 cue) favours one option while 4-5 lower-validity cues favour the other, giving tally margins of 3 or 4 (trial pairs 1/2, 5/6, 7/8, both A/B mirrorings). On exactly these trials the two heuristics make OPPOSITE-SIGN predictions, so the metric 'proportion of choices matching the tally winner' is pushed toward 1 by Tallying (a margin of 3-4 multiplied by beta saturates the softmax for almost all beta in [0.1,20]) and toward 0 by Take-The-Best (which picks the single-cue winner with probability sigmoid(beta), i.e. the tally loser). Restricting to margin>=3 keeps the Tallying-side signal near ceiling so the per-subject value is tight, and mirrored A/B versions of every pair mean no side bias can inflate or deflate the score. Each subject contributes ~36 trials to the estimate, so binomial noise is small; the remaining between-subject spread comes only from the sampled beta/epsilon, which affects both theories symmetrically around 0.5. Predicted values: Tallying ~0.8, TTB ~0.2."
}
```

## Usage

```json
{
  "input_tokens": 7255,
  "output_tokens": 3491
}
```
