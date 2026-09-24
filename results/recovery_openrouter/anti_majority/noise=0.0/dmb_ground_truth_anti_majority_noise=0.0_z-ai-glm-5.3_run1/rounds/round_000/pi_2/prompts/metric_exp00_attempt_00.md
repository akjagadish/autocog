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

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 10: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 11: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 14: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 15: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 16: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 17: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 18: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 19: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 20: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 21: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 22: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** This design is the complementary mirror of the already-explored experiment: instead of holding TTB's deciding cue fixed and varying the tally margin, it holds the TALLY MARGIN fixed and varies WHICH CUE TTB's cascade lands on. Six experts with validities [0.95, 0.90, 0.85, 0.65, 0.60, 0.55] create a sharp gap between high-validity cues (1-3) and low-validity cues (4-6), so the identity of TTB's first discriminating cue is unambiguous. Every trial is built around a designated tally-winner W with a fixed tally margin m in {1, 2, 3}, crossed with the DIRECTION of the top discriminating cue: CONGRUENT cells (top cue favors W, so TTB and Tallying agree in direction) and CONFLICT cells (top cue favors the loser, so TTB and Tallying point in opposite directions). Within conflict cells at m=1, the deciding cue's position d is also varied (d=1..4), testing whether choices track the validity of the deciding cue. The quantitative signature is a 2x3 factorial of predicted P(W): Tallying predicts P(W) = sigmoid(m*beta) + lapse, i.e., IDENTICAL across congruent/conflict cells at each margin and a monotone gradient across margins [sigma(beta) < sigma(2*beta) < sigma(3*beta)]. TTB predicts P(W) = sigmoid(beta) in every congruent cell and 1 - sigmoid(beta) in every conflict cell, i.e., IDENTICAL across margins (its noise model does not scale with the number of supporting cues) and independent of which cue d decides. Thus Tallying predicts a main effect of margin only; TTB predicts a main effect of top-cue direction only. A subject whose P(W) rises with m at fixed direction is Tallying; a subject whose P(W) is flat across m but flips with cue direction is TTB. Even the congruent cells dissociate at m>=2 (sigma(beta) vs sigma(2*beta)), and the m=1 congruent cells serve as agreement baselines for estimating beta. All 12 base pairs are mirrored (A/B swapped) to control side bias, giving 24 unique pairs with K = 96 // 24 = 4 repetitions each (96 trials total, 8 samples per base cell per subject). No pair is degenerate (every pair has discriminating cues, none is an all-tie), and no pair leaves both theories in pure-guess agreement. Secondary exploratory axis: both theories predict FLAT P(W) across the deciding-cue position d (TTB because its softmax does not scale with validity, Tallying because it ignores validity entirely), so any graded sensitivity to d would falsify both — but the primary dissociation rests on the margin x direction interaction.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Signature: allegiance to the tally winner on CONFLICT trials\n    # (trials where the single top-validity discriminating cue favors\n    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5\n    # there (it follows the cue majority against the best cue);\n    # Take-The-Best predicts P < 0.5 (it follows the top cue against\n    # the majority). We return P(W | conflict) - 0.5, so the sign\n    # itself identifies the theory for any noise level beta > 0.\n    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])\n    cue_order = np.argsort(-validities, kind=\"stable\")\n\n    a_list = data[\"option_a_ratings\"].tolist()\n    b_list = data[\"option_b_ratings\"].tolist()\n    resp = data[\"response\"].to_numpy()\n\n    conflict_w = []\n    for a_raw, b_raw, r in zip(a_list, b_list, resp):\n        a = np.asarray(a_raw, dtype=float)\n        b = np.asarray(b_raw, dtype=float)\n\n        a_w = float(np.sum(a > b))\n        b_w = float(np.sum(b > a))\n        if a_w == b_w:\n            # Tally is tied -> heuristic guesses; uninformative cell.\n            continue\n        winner = 0 if a_w > b_w else 1\n\n        # First discriminating cue in descending-validity order.\n        top_favors = None\n        for j in cue_order:\n            if a[j] > b[j]:\n                top_favors = 0\n                break\n            if b[j] > a[j]:\n                top_favors = 1\n                break\n        if top_favors is None:\n            continue  # no discriminating cue at all\n\n        if top_favors == winner:\n            continue  # congruent trial: both heuristics point the same way\n\n        # Conflict trial: tally winner opposes the top cue's favorite.\n        conflict_w.append(1.0 if int(r) == winner else 0.0)\n\n    if len(conflict_w) == 0:\n        return 0.0\n    return float(np.mean(conflict_w)) - 0.5",
  "rationale": "The design's core dissociation is the CONFLICT cell: trials where the tally majority (the designated winner W) is opposed by the highest-validity discriminating cue. On those trials the two theories make opposite-signed predictions about P(choose W) for ANY noise parameters: Tallying gives P(W) = sigmoid(m*beta) with m >= 1, hence strictly > 0.5 (it obeys the cue majority even against the best cue); TTB gives P(W) = 1 - sigmoid(beta), hence strictly < 0.5 (it obeys the single top cue against the majority). So the metric P(W | conflict) - 0.5 is positive under Tallying and negative under TTB regardless of beta, epsilon, or which cue position d decides \u2014 the sign itself is diagnostic, and the magnitude grows with beta on both sides (the gap is (1-epsilon)*[weighted sigma(m*beta) + sigma(beta) - 1] over the 64 conflict trials per subject). This uses 64 of 96 trials per subject (conflict base trials 1,2,3,4,7,8,9,11,13,14,15,16,19,20,21,23 x 4 reps), giving tight per-subject estimates: e.g., at beta=1 the Tallying value is ~ +0.31 and the TTB value ~ -0.23 (gap ~0.55) with per-subject SD ~0.05-0.06, which at N=25 yields a Welch t in the dozens; even at the minimum beta=0.1 the gap (~0.07) still exceeds the per-subject noise (~0.018 SE). Congruent trials are deliberately excluded: at m=1 congruent both theories predict exactly sigmoid(beta), so they add variance without adding mean separation. The metric is computed identically on the pooled DataFrame and on single-subject slices (each subject has all 24 unique pairs), so between-subject variance is well defined. The classification (tally winner, top discriminating cue via the fixed descending-validity order) is recomputed from the raw ratings on every row, so no list-column hashing is needed."
}
```

## Usage

```json
{
  "input_tokens": 4705,
  "output_tokens": 10957,
  "reasoning_tokens": 9928,
  "provider": "Sail Research"
}
```
