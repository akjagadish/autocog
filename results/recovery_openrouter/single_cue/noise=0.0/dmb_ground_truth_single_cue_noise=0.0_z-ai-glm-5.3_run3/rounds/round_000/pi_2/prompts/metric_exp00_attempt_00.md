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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 13: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 14: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design's core novelty is a cue-allegiance flip at FIXED tally counts — the mirror image of the classic margin-gradient design. Tallying's predicted probability depends ONLY on the win counts through a softmax, which is shift-invariant (softmax over [3,2] equals softmax over [1,0]); it is therefore EXACTLY invariant to which features each option wins on and to cue depth. TTB's choice, in contrast, is determined solely by the allegiance of the highest-validity discriminating cue. The factorial core (trials 1-8) crosses tally margin {1, 2} x top-cue allegiance {pro-tally (agree), anti-tally (conflict)}: (i) C+ vs C- (and H+ vs H-) hold the win counts and even the tie placement identical (f0 tied, counts 3-2) while permuting cue allegiances so the top discriminating cue flips from favoring the tally winner to opposing it. Tallying predicts a precisely zero change in P(A) (identical sigma(beta)); TTB predicts a full flip from sigma(beta) to sigma(-beta). (ii) C vs H holds allegiance fixed and doubles the margin: TTB predicts exactly zero change (its probability is sigma(beta) for the top-cue winner regardless of how many cues oppose it), while Tallying predicts sigma(beta) -> sigma(2beta). The two theories thus occupy orthogonal cells of the 2x2: TTB shows a main effect of allegiance only; Tallying a main effect of margin only. The depth series (trials 9-14) places the anti-tally win at f0 (v=.95), f2 (v=.85), or f3 (v=.65), all at margin 1 with counts (3,2) or (2,1): both theories predict FLAT behavior here (TTB's probability is cue-depth invariant; Tallying's softmax is shift-invariant across (3,2)/(2,1)), so any validity-weighted hybrid would break this double invariance — a sharp signature that neither target theory but only mixtures would produce. The extreme anchors (trials 15-16) are 1-vs-5 splits where the top cue alone opposes five cues: TTB predicts P(cue winner) = sigma(beta) — numerically IDENTICAL to its prediction on the margin-1 conflict trials (invariance to 4 extra opposing cues), while Tallying predicts sigma(4beta) for the tally winner, near-deterministic at any reasonable beta. Quantitative dissociation summary: (1) on the allegiance flip (C+/C-, H+/H-), Tallying's predicted slope is exactly 0 and TTB's is -tanh(beta/2); (2) on the margin manipulation (C vs H), TTB's predicted slope is exactly 0 and Tallying's is positive; (3) across the depth series both are flat, jointly excluding weighted hybrids; (4) E1 vs C- isolates TTB's cue-count invariance against Tallying's steep margin gradient. Mirrors of every trial type absorb side/position bias. With 16 unique pairs, K = 96 // 16 = 6, yielding exactly 96 trials per subject.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Tally-vs-TTB choice-allegiance contrast.\n\n    For every trial we determine (i) the TALLYING winner (option with more\n    strict feature-wise wins) and (ii) the TAKE-THE-BEST winner (winner on\n    the highest-validity discriminating cue, validities fixed by design as\n    [0.95, 0.9, 0.85, 0.65, 0.6, 0.5]).  The metric is\n\n        mean( response == tally winner ) - mean( response == TTB winner )\n\n    On the 4 trial types where both heuristics pick the same option\n    (trials 1, 3, 5, 7 and mirrors) the two indicators are identical and\n    cancel.  On the 12 conflict trial types they are exact complements, so\n    the metric reduces to 0.75 * (2*P(tally winner | conflict) - 1): strongly\n    positive under Tallying, strongly negative under TTB.\n    \"\"\"\n    validities = np.array([0.95, 0.9, 0.85, 0.65, 0.6, 0.5])\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    tally_match = []\n    ttb_match = []\n\n    for ca, cb, cr in zip(a_col, b_col, r_col):\n        a = np.asarray(ca, dtype=float).ravel()\n        b = np.asarray(cb, dtype=float).ravel()\n        r = int(cr)\n\n        # Tallying winner: more strict feature-wise wins (ties on a feature\n        # contribute to neither side).\n        a_wins = float(np.sum(a > b))\n        b_wins = float(np.sum(b > a))\n        if a_wins > b_wins:\n            tally_winner = 0\n        elif b_wins > a_wins:\n            tally_winner = 1\n        else:\n            tally_winner = -1  # undiscriminating tally -> excluded\n\n        # TTB winner: first discriminating cue in descending-validity order.\n        ttb_winner = -1\n        for j in cue_order:\n            if a[j] > b[j]:\n                ttb_winner = 0\n                break\n            if b[j] > a[j]:\n                ttb_winner = 1\n                break\n\n        if tally_winner >= 0:\n            tally_match.append(1.0 if r == tally_winner else 0.0)\n        if ttb_winner >= 0:\n            ttb_match.append(1.0 if r == ttb_winner else 0.0)\n\n    if len(tally_match) == 0 or len(ttb_match) == 0:\n        return 0.0\n\n    return float(np.mean(tally_match) - np.mean(ttb_match))",
  "rationale": "The design's 16 trial types split into 4 agree types (tally winner = TTB winner: trials 1, 3, 5, 7) and 12 conflict types (tally winner = TTB loser: trials 2, 4, 6, 8-16). The metric is the difference between overall choice-consistency with Tallying and overall choice-consistency with TTB. On agree trials the two consistencies are identical and cancel; on conflict trials they are exact complements, so the metric equals 0.75 * (2*P(choose tally winner | conflict trials) - 1). Under simulated Tallying data, P(tally winner | conflict) = softmax over tallies (e.g. ~0.77 at beta=1, ~1.0 at large beta), giving a strongly POSITIVE value (~+0.4 at beta=1, up to +0.75). Under simulated TTB data, P(tally winner | conflict) = 1 - sigmoid(beta) (TTB always follows the top cue regardless of how many cues oppose it, including the 1-vs-5 anchors), giving a strongly NEGATIVE value (~-0.35 at beta=1, down to -0.75). The between-theory gap in the mean is therefore large at every plausible parameter setting (roughly 0.09 even at beta=0.1, ~0.27 at beta=0.3, ~0.75 at beta>=2), while per-subject variance stays tight because each subject contributes 96 trials (72 conflict) of simple binary indicators: per-subject SD ~0.08, so with N=25 subjects the Welch t-statistic ranges from ~4 (worst case, near-random parameters) to ~35 (moderate/large beta), comfortably below alpha=0.01. The metric works identically on the pooled DataFrame and on any single subject's slice (it is just a difference of two proportions), and it is side-balanced (the tally winner is option A on exactly 8 of the 16 trial types), so position/label bias cancels in expectation."
}
```

## Usage

```json
{
  "input_tokens": 4432,
  "output_tokens": 9221,
  "reasoning_tokens": 8045,
  "provider": "Sail Research"
}
```
