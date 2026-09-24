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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 12: A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 13: A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 14: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 15: A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** This design dissociates Tallying from Take-The-Best (TTB) along two orthogonal quantitative dimensions simultaneously: (i) the tally MARGIN (a_wins - b_wins), which is the ONLY trial-dependent quantity in Tallying's softmax (p(A) = (1-eps)*sigmoid(beta*margin) + eps/2), and (ii) the identity/validity of the first discriminating cue, which is the ONLY trial-dependent quantity in TTB (p(A) = (1-eps)*sigmoid(beta) + eps/2 toward the top-cue winner, constant across ALL discriminating trials). I use 6 features with an unambiguous validity hierarchy [0.95, 0.85, 0.75, 0.65, 0.60, 0.55]. CENTRAL SIGNATURE 1 — the full crossing psychometric function (T1-T5 and mirror T6-T10): the top cue (0.95) always favors one option while the tally margin sweeps across its full range (-4, -2, 0, +2, +4). TTB predicts a FLAT p(A) = (1-eps)*sigmoid(beta)+eps/2 across all five trials in each direction; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (T3/T8) — a knife-edge prediction independent of beta that any consistent preference falsifies — and reverses sign at negative margins (T1/T2: TTB says A with high confidence, Tallying says B; T6/T7: the mirror sign reversal, ruling out side bias). T1 vs T10 is especially clean: identical tally margins (-4) but opposite TTB winners, so Tallying predicts identical p(A) on both while TTB predicts opposite extremes. CENTRAL SIGNATURE 2 — cascade-depth invariance (T11-T15): the top cue TIES and the TTB decision comes from cue 2 (0.85), cue 3 (0.75), or even cue 5 (0.60, T14). TTB predicts the SAME constant p(A) on T1, T2, T3, T11, T12, T13, T14 (all TTB->A regardless of which cue decided or how deep the cascade ran); Tallying predicts six distinct values spanning sigmoid(-2beta) to sigmoid(+1beta), including exact 0.5 on T3/T13/T14. T15 is a depth-2 sign reversal (cue 2 favors B, tally margin +2 favors A). T14 probes the bottom of the hierarchy: when cues 1-4 all tie, TTB commits to the single 0.60-validity cue while Tallying sits at exactly 50/50 — if subjects treat a lone low-validity cue as uninformative, TTB fails where Tallying succeeds. T16 (full tie) is the shared 50/50 control anchoring lapse rate and side bias. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice-proportion estimates. Model comparison on the full 16-cell response profile — flat vs. graded across margins, constant vs. varied across cascade depths, and exact-0.5 vs. deterministic on margin-0 trials — cleanly separates the theories even after fitting beta and epsilon to each.

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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef _to_matrix(col):\n    # col: pandas Series whose cells are lists/tuples/arrays of binary ratings\n    return np.array([list(r) for r in col], dtype=float)\n\n\ndef metric(data: pd.DataFrame) -> float:\n    if data is None or len(data) == 0:\n        return 0.0\n\n    A = _to_matrix(data['option_a_ratings'])\n    B = _to_matrix(data['option_b_ratings'])\n    if A.ndim != 2 or A.shape != B.shape or A.shape[0] == 0:\n        return 0.0\n\n    # Tally margin: number of features where A beats B minus number where B beats A.\n    margin = (A > B).sum(axis=1) - (B > A).sum(axis=1)\n\n    # Signed choice: +1 if the subject chose A (response == 0), -1 if B (response == 1).\n    resp = pd.to_numeric(data['response'], errors='coerce').to_numpy(dtype=float)\n    s = 1.0 - 2.0 * resp\n\n    # Alignment between the signed choice and the tally margin.\n    # Tallying: E[s | margin] = (1-eps) * tanh(beta*margin/2)  =>  E[s*margin] > 0,\n    # scaling like beta * E[margin^2] / 2 for small beta and saturating at mean(|margin|)\n    # for large beta (max value 28/16 = 1.75 on this design).\n    # TTB: E[s | trial] = (1-eps)*tanh(beta/2)*ttb_sign, and on this stimulus set\n    # sum over the 16 trial types of ttb_sign * margin = -2 (mean -0.125), so the\n    # TTB value is slightly NEGATIVE for any beta > 0.\n    return float(np.mean(s * margin))",
  "rationale": "CORE IDEA: the signed choice-tally covariance, mean over trials of s*margin where s=+1 for choosing A and margin = (#features A wins) - (#features B wins). This is the locally most powerful (weak-signal optimal) linear discriminator between the two theories on this stimulus set, and it works on both the pooled data and any single subject's 96-trial slice.\n\nWHY IT SEPARATES THE THEORIES:\n(1) Under Tallying, p(A) = (1-eps)*sigmoid(beta*margin) + eps/2, so E[s] = (1-eps)*tanh(beta*margin/2). Because tanh is odd, E[s*margin] = (1-eps)*|margin|*tanh(beta*|margin|/2) >= 0 on EVERY trial with nonzero margin \u2014 every trial type pushes the statistic in the same (positive) direction. The statistic equals (1-eps)*mean(|m|*tanh(beta|m|/2)), which for this design ranges from ~beta*E[m^2]/2 = 2.69*beta (small beta) up to (1-eps)*28/16 = 1.75*(1-eps) (large beta). So for any nondegenerate (beta, eps) the Tallying value is clearly positive.\n(2) Under TTB, E[s] = (1-eps)*tanh(beta/2)*ttb_sign, independent of margin. Summing ttb_sign*margin over the 16 trial types gives exactly -2 (T1..T5 contribute -4,-2,0,+2,+4 = 0; T6..T10 contribute -4,-2,0,+2,+4 = 0; T11..T14 contribute -1,+1,0,0 = 0; T15 contributes -2; T16 contributes 0), so the TTB value is -0.125*(1-eps)*tanh(beta/2) \u2014 slightly NEGATIVE for all beta>0. The sign of the statistic itself flips between theories.\n\nWHY THE CONTRAST IS TIGHT ACROSS SUBJECTS: each subject contributes 96 trials with 6 repetitions of each of the 16 cells; the statistic is an average of bounded per-trial values (|s*margin| <= 4). At moderate/large beta the Tallying choices on the high-margin cells (|m|=4: T1,T5,T6,T10; |m|=2: five more cells) are near-deterministic, so per-subject SD is tiny (~0.04 at beta=2), giving t > 50 against TTB's small negative value. Even at small beta the m-weighting is exactly the Neyman-Pearson-optimal weighting (signal proportional to margin), maximizing the mean-to-SD ratio; e.g. at beta=0.1, eps=0 the gap is ~0.27 with per-subject SD ~0.24, still yielding t ~ 4 at N=25. Margin-0 trials (T3, T8, T13, T14, T16) contribute exactly 0 under both theories, so they add no variance while the design's mirror structure (T1 vs T10, T6 vs T5) cancels any side bias.\n\nROBUSTNESS: the statistic requires no parameter fitting, no knowledge of beta/epsilon, uses only the schema columns, is invariant to trial order, and is well-defined on a single subject's slice (mean over their own trials), so between-subject variance is computable."
}
```

## Usage

```json
{
  "input_tokens": 4459,
  "output_tokens": 9288,
  "reasoning_tokens": 8059,
  "provider": "Sail Research"
}
```
