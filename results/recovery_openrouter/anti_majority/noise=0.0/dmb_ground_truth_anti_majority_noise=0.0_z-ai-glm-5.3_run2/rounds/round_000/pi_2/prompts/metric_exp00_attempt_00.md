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

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 10: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 11: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 13: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 14: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 15: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 16: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 18: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 19: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 20: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 21: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 22: A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  trial 23: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 24: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design inverts the logic of the classic TTB-vs-Tallying conflict: instead of holding the top cue fixed and varying the tally margin (already explored), I hold the TALLY MARGIN fixed and parametrically vary WHICH cues the losing-side wins on. I use 5 experts with strictly ordered validities [0.90, 0.80, 0.70, 0.60, 0.55] so TTB's cascade order is unambiguous. CORE DISSOCIATION (pairs 1-7): every one of these pairs has the identical tally structure — A wins exactly 3 cues, B wins exactly 2 (margin +1 for A) — so Tallying predicts the exact same P(A) = sigmoid(beta) on all seven trials, a perfectly FLAT psychometric function over the series. TTB instead predicts a STEP FUNCTION: on pairs 1-3 the top cue (v=0.90) is one of B's wins, so TTB confidently picks B (P(A) = 1 - sigmoid(beta)); on pairs 4-7 the top cue favors A, so TTB confidently picks A (P(A) = sigmoid(beta)). The predicted discontinuity sits exactly between pair 3 (B wins cues {1,5}) and pair 4 (B wins cues {2,3}) — TTB specifies the step's location precisely, while Tallying specifies zero slope. This is a double dissociation in slope: the previously explored design had TTB flat / Tallying graded; this one has Tallying flat / TTB discontinuous, so the two designs jointly triangulate both theories. Pairs 8-14 are exact A/B mirrors of 1-7 to absorb side/response bias (Tallying: flat sigma(-beta); TTB: mirrored step). SECOND PROBE — cascade depth at graded tally (pairs 15-17): the top k cues are forced to tie so TTB's deciding cue moves down the validity ladder (cue 2, then cue 3, then cue 4) while it always picks B with the SAME confidence 1 - sigmoid(beta), because TTB's binary score is validity-invariant. Tallying, in contrast, predicts a strictly graded decline in P(A) — sigma(2*beta), sigma(beta), and exactly 0.5 — because the tally margin shrinks from +2 to +1 to 0 as ties remove A's winning cues. Pair 17 is the sharpest single trial: Tallying predicts exactly 50/50 guessing (tally 1-1), TTB predicts a confident B choice from cue 4 alone. Pairs 18-20 mirror these. CONTROLS: pair 21 (A wins all cues) is an agreement trial calibrating the shared sigmoid(beta) confidence asymptote; pair 22 (identical vectors) is a full-tie trial where both theories must guess, giving a pure lapse-rate baseline; pairs 23-24 are extreme single-cue-vs-all-others conflicts (top cue alone vs. four opposing cues) providing maximum-separation anchor points where the theories disagree most strongly in magnitude (Tallying: sigma(-3*beta) vs TTB: sigmoid(beta)). With 24 unique pairs, K = 96 // 24 = 4 repetitions yields exactly 96 trials. No feedback keeps subjective cue order anchored to instructed validities. DECISION RULE: if P(A) across pairs 1-7 is flat at the level fit on margin+1 trials, Tallying is supported; if it shows a sharp step exactly between pairs 3 and 4, TTB is supported; a graded monotone drift favoring neither step nor flatness would falsify both pure forms.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"Signed tally-following rate on TTB/Tallying conflict trials.\n\n    For every trial we reconstruct both theories' predicted winner from the\n    option ratings (validities [0.9, 0.8, 0.7, 0.6, 0.5], so the TTB cascade\n    order is feature 0 > 1 > 2 > 3 > 4).  We keep only trials where BOTH\n    theories are decisive AND they disagree (16 of the 24 unique pairs,\n    4 reps each = 64 trials per subject).  The metric is\n        2 * P(choose the Tallying side | conflict) - 1,\n    i.e. +1 = always follow Tallying, -1 = always follow TTB, 0 = chance.\n    \"\"\"\n    import numpy as np\n\n    validities = np.array([0.90, 0.80, 0.70, 0.60, 0.50])\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    def to_vec(x):\n        # Robustly convert a cell (list / tuple / ndarray / string) to a\n        # float vector of per-feature ratings.\n        if isinstance(x, str):\n            s = x.strip().strip(\"[]()\")\n            return np.array([float(v) for v in s.replace(\",\", \" \").split()])\n        arr = np.asarray(x, dtype=float).ravel()\n        return arr\n\n    followed = 0\n    total = 0\n\n    a_col = data[\"option_a_ratings\"]\n    b_col = data[\"option_b_ratings\"]\n    r_col = data[\"response\"]\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        a = to_vec(a_raw)\n        b = to_vec(b_raw)\n        if a.shape[0] == 0 or a.shape != b.shape:\n            continue\n\n        # --- Tallying prediction: count strict feature-wise wins ---\n        a_wins = float(np.sum(a > b))\n        b_wins = float(np.sum(b > a))\n        if a_wins == b_wins:\n            # Tallying is indifferent here (tie or all-tie): not a decisive\n            # conflict trial; skip.\n            continue\n        tally_picks_a = a_wins > b_wins\n\n        # --- TTB prediction: first discriminating cue in validity order ---\n        ttb_picks_a = None\n        for j in cue_order:\n            if a[j] > b[j]:\n                ttb_picks_a = True\n                break\n            if b[j] > a[j]:\n                ttb_picks_a = False\n                break\n        if ttb_picks_a is None:\n            continue  # TTB guesses; not decisive\n\n        if ttb_picks_a == tally_picks_a:\n            continue  # theories agree; uninformative for discrimination\n\n        # Conflict trial: did the subject follow the Tallying side?\n        chose_a = (float(resp) == 0.0)\n        total += 1\n        if chose_a == tally_picks_a:\n            followed += 1\n\n    if total == 0:\n        return 0.0\n\n    rate = followed / float(total)\n    return 2.0 * rate - 1.0\n```",
  "rationale": "The design contains 16 unique pairs (x4 reps = 64 trials/subject) where Tallying and Take-The-Best are both decisive and point in OPPOSITE directions: pairs 1-3, 11-16, 24 (TTB's top cue favors B while the 3-vs-2 tally favors A) and their mirrors 8-10, 18, 19, 23 (TTB favors A while the tally favors B). On exactly these trials the two theories make sign-opposite choice predictions, so the signed rate of following the Tallying side is a direct readout of which policy generated the data. Under Tallying, every conflict trial offers the subject a tally margin of at least +1, so P(follow tally) = (1-eps)*sigmoid(beta) + eps/2 >= 0.5 for ANY admissible parameters, giving a metric value in [0, 1) (typically ~0.6-0.9). Under TTB the same probability is 1 - [(1-eps)*sigmoid(beta) + eps/2] <= 0.5, giving a value in (-1, 0] (typically ~-0.6 to -0.9). The two distributions are therefore sign-separated by construction, and the expected between-theory gap is 2*(1-eps)*(2*sigmoid(beta)-1), which is large across the whole parameter space. Crucially for the acceptance test, the metric averages over 64 binary observations per subject, so per-subject binomial noise is tiny (sd ~ 0.04-0.06); the residual between-subject variance comes only from heterogeneity in beta/epsilon, which shifts both theories' means in opposite directions around 0 without collapsing the gap. This yields a Welch t-test at N=25 that is overwhelmingly significant (expected |t| > 10). The metric works identically on the pooled DataFrame and on any single subject's slice (every subject sees all 24 pairs x 4 reps), satisfying the between_subject_variance requirement. Side/response bias is automatically cancelled because the conflict set contains exact A/B mirror pairs: a subject with an A-bias follows the tally side on half the conflict trials and violates it on the mirrors, contributing ~0 to the metric under either theory."
}
```

## Usage

```json
{
  "input_tokens": 4722,
  "output_tokens": 10874,
  "reasoning_tokens": 9589,
  "provider": "Sail Research"
}
```
