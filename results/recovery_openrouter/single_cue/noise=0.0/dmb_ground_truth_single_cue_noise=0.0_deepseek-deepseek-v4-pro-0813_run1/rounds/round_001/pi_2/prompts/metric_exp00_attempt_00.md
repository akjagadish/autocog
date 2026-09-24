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
**Validities (n_features=12):** [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Rationale:** Twelve experts: six high-validity (0.94-0.99) and six low-validity (0.50-0.55). The eight unique pairs form four groups. Group A (A1-A3): A wins 3-5 high-validity features while B wins 4-6 low-validity features, so Tallying margin is -1 (predicts p(A)<0.5), but validity-weighted evidence is positive (predicts p(A)>0.5). Group B (B1-B3): A wins 4-6 low-validity features while B wins 3-5 high-validity features, so Tallying margin is +1 (predicts p(A)>0.5), but validity-weighted evidence is negative (predicts p(A)<0.5). Groups C and D have zero Tallying margin (predicts p(A)=0.5) but all-high vs all-low wins give positive and negative validity-weighted evidence respectively. Thus Tallying and validity-weighted integration make opposite directional predictions for every pair, and the within-group graded evidence allows estimating beta for both accounts.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** People do not stop after the first discriminating cue, nor do they weight every cue equally. Instead, they form a continuous decision value by comparing the two options on every feature, weighting each signed feature comparison by a monotonically increasing function of the advertised cue validity. Cues with validity near chance are either ignored or heavily discounted, while highly valid cues receive much larger weight. Ties on a feature contribute zero evidence. The resulting validity-weighted evidence sum is passed through a softmax with inverse temperature beta, plus a small uniform lapse probability epsilon.

**Parameters:**
- beta: [1.0, 50.0]
- epsilon: [0.0, 0.25]
- offset: [0.0, 0.5]
- gamma: [0.5, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-weighted additive integration.
    # state is array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. History is ignored because validities are
    # advertised in the instructions and no trial-by-trial feedback is given.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-weighted integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    offset = float(parameters["offset"])
    gamma = float(parameters["gamma"])

    # Monotone diagnosticity weight: w_j = max(validity_j - offset, 0)^gamma.
    # For offset = 0.5 and gamma = 1 this is a validity-gain weighting,
    # giving chance cues zero weight.
    weights = np.maximum(validities - offset, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    high_a = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)\n    low_a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)\n\n    a_key = data['option_a_ratings'].apply(tuple)\n    mask_high_a = a_key == high_a\n    mask_low_a = a_key == low_a\n\n    if mask_high_a.sum() == 0 or mask_low_a.sum() == 0:\n        return 0.0\n\n    pA_high = 1.0 - float(data.loc[mask_high_a, 'response'].mean())\n    pA_low = 1.0 - float(data.loc[mask_low_a, 'response'].mean())\n\n    return float(pA_high - pA_low)\n",
  "rationale": "Trials 7 and 8 are the cleanest discriminating pair: A has all six high-validity features and B all six low-validity features on trial 7, with the mirror image on trial 8. Tallying counts six feature wins for each option on both trials, predicting chance A choices and a difference in A-choice rates near zero. Validity-weighted integration instead predicts a strong A bias on trial 7 and a strong B bias on trial 8, so the difference in A-choice rates should be strongly positive. Using only these two pair types on repeated presentations keeps the metric transparent and per-subject variance small."
}
```

## Usage

```json
{
  "input_tokens": 4262,
  "output_tokens": 9721,
  "reasoning_tokens": 9328,
  "provider": "Baidu"
}
```
