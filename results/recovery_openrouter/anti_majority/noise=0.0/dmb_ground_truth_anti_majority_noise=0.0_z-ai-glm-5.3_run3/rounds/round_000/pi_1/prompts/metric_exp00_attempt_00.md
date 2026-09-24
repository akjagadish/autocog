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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 13: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 14: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** The design exploits two orthogonal dissociations between Take-The-Best (TTB) and Tallying. (1) SIGN dissociation with graded tally margins: On conflict trials, the top-validity cue favors one option while the majority of cues favors the other. Critically, TTB's predicted choice probability is a constant sigmoid(beta) on every such trial (it only ever uses the single top discriminating cue), whereas Tallying's confidence scales with the tally margin: sigmoid(beta) for 2-vs-1, sigmoid(2*beta) for 3-vs-1, sigmoid(3*beta) for 4-vs-1. So TTB predicts FLAT confidence across trials 1/3/5/13 (margins 3, 1, 2, 1 respectively) while Tallying predicts systematically increasing confidence in the majority option — a quantitative, trial-level signature that model recovery cannot confuse. Mirrored versions (trials 2/4/6/14) control for side bias. (2) TALLY-TIE dissociation: Trials 7/8 have a 2-2 tally with one tied cue, so Tallying must guess (exactly 50%) while TTB is confident (sigmoid(beta)) because the top cue discriminates. Agreement/control trials (9-12, 15-16) anchor both models and check that subjects are not simply counting or simply using cue 1. Validities are spread (0.95 down to 0.55) so the validity ordering is unambiguous and communicated up front; the second and third cues are placed far enough below the top cue that TTB's cascade is well defined, and low-validity cues (0.65, 0.60, 0.55) are close together so Tallying treats them as a coherent majority block. With 16 unique pairs and K=6 repetitions (96 trials), each dissociating cell gets enough repetitions for reliable per-cell choice proportions. Predicted pattern: if TTB is correct, choice proportions on trials 1, 3, 5, 13 should be statistically indistinguishable from each other (all ~sigmoid(beta) favoring the top-cue option) and well above 50% on trials 7/8; if Tallying is correct, choices on conflict trials should track the tally margin (increasingly favoring the majority option from trial 3 -> 5 -> 1) and trials 7/8 should sit at exactly 50%.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Dissociation score: on trials where Take-The-Best (top-validity\n    # cascade) and Tallying (cue-count majority) make DIFFERENT modal\n    # predictions, how often does the subject side with the TTB option?\n    # TTB predicts > 0.5 on every such trial (constant sigmoid(beta)\n    # confidence, including on tally-tie trials 7/8 where Tallying must\n    # guess at exactly 50%). Tallying predicts < 0.5 on conflict trials\n    # (majority option wins) and exactly 0.5 on tally ties. So the metric\n    # is positive under TTB and negative under Tallying.\n    V = np.array([0.95, 0.85, 0.65, 0.60, 0.50])\n    order = np.argsort(-V, kind=\"stable\")\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    resp = data[\"response\"].to_numpy()\n\n    matches = []\n    for i in range(len(a_col)):\n        a = np.asarray(a_col[i], dtype=float)\n        b = np.asarray(b_col[i], dtype=float)\n\n        # TTB winner: first discriminating cue in validity order.\n        ttb = -1\n        for j in order:\n            if a[j] > b[j]:\n                ttb = 0\n                break\n            if b[j] > a[j]:\n                ttb = 1\n                break\n        if ttb == -1:\n            continue  # TTB guesses; not diagnostic\n\n        # Tallying winner: strict cue-count majority.\n        a_wins = int(np.sum(a > b))\n        b_wins = int(np.sum(b > a))\n        if a_wins > b_wins:\n            tal = 0\n        elif b_wins > a_wins:\n            tal = 1\n        else:\n            tal = -1  # tally tie -> Tallying guesses\n\n        # Keep only dissociation trials: the two theories' modal\n        # predictions differ (tally tie, or tally majority against TTB).\n        if tal == ttb:\n            continue\n\n        matches.append(1.0 if resp[i] == ttb else 0.0)\n\n    if len(matches) == 0:\n        return 0.0\n    # Center at 0.5: positive => TTB-like, negative => Tallying-like.\n    return float(np.mean(matches)) - 0.5",
  "rationale": "The design contains 10 of 16 trial types where TTB and Tallying make different modal predictions: conflict trials (1-6, 13-14; 48 of 96 trials) where the top cue favors one option but the cue majority favors the other, and tally-tie trials (7-8; 12 of 96 trials) where the tally is 2-2 so Tallying must guess at exactly 50% while TTB is confident via the top cue. The metric recomputes both heuristics row-by-row from the ratings and the known validities, keeps only these dissociation rows, and returns P(choose the TTB-predicted option) - 0.5. Under TTB, every dissociation trial is chosen with probability (1-eps)*sigmoid(beta) + eps/2 > 0.5, so the metric is positive (approaching +0.5 for large beta). Under Tallying, the 48 conflict trials are chosen at (1-eps)*sigmoid(-beta*margin) + eps/2 < 0.5 and the 12 tie trials at exactly 0.5, so the metric is negative (approaching -0.4 for large beta). The sign itself flips between theories, giving a large mean gap (e.g., ~0.66 at beta=2, eps=0.1; still ~0.065 even at the near-degenerate beta=0.1, eps=0). Per-subject variability is pure binomial noise over ~60 trials (sd <= ~0.065 at worst), so Welch's t-test at N=25 yields enormous t-statistics across essentially the whole non-degenerate parameter space. The metric is computed identically on pooled and single-subject data (each subject has all 16 trial types x 6 reps), so between-subject variance is well defined. It avoids hashing any sequence-valued column (no groupby/unique on ratings), and it is not trivially true for TTB: it requires subjects to actually follow the one-reason cascade against the cue majority on the majority of dissociation trials."
}
```

## Usage

```json
{
  "input_tokens": 4156,
  "output_tokens": 6154,
  "reasoning_tokens": 5092,
  "provider": "Sail Research"
}
```
