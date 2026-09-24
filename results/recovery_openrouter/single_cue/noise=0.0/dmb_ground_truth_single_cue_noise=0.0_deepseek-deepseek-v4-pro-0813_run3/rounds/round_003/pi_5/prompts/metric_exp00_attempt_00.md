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
**Validities (n_features=20):** [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 12: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** This 20-feature design uses 10 experts with validity 1.00 and 10 experts with validity 0.50, so mean validity is 0.75 and centered validities are +0.25 for the first 10 features and -0.25 for the last 10 features. All unique pairs have tally difference d = +1 or -1, so there are zero tally ties. The context-gated tallying theory therefore stays in its unweighted tallying mode throughout: tie prevalence is zero, the validity gate never opens, and p(A) depends only on d. Within the d=+1 block, the validity-composition score S = sum_j (v_j - 0.75) * (a_j - b_j) takes values +4.75, +2.75, +0.75, -0.75, -2.75, and -4.75; the d=-1 block is the exact A/B swap. The advocated theory predicts one flat p(A) across all six d=+1 pairs and one flat p(A) across all six d=-1 pairs. The competing validity-graded integration predicts D = d + kappa * S, so for any subject with kappa not equal to zero p(A) must be monotone in S within each d block; across subjects with a symmetric kappa prior this appears as S-linked slope or overdispersion rather than the advocated flat S-independent curve. Mirrored swaps control response bias and create a quantitative double dissociation: two flat d-based levels versus two S-ordered psychometric gradients.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People adaptively decide whether unweighted tallying is diagnostic in the current task. They monitor two context signals over recent stimulus history: the prevalence of tally ties and the mean absolute per-trial validity-weighted evidence on those tie trials. When ties are rare, choice remains essentially unweighted tallying. When ties are prevalent and validity evidence is diagnostically substantial in absolute magnitude on tie trials, a validity-based tie-break policy is recruited. The recruitment is near-binary rather than continuous, so low-tie tasks recover near-tallying behavior while high-tie diagnostic tasks show strong validity-aligned tie-breaking. Unlike a fixed validity-gated tallying rule, the validity route is switched by task composition, and unlike continuous validity grading, validity is not mixed into every trial.

**Parameters:**
- beta_tally: [4.0, 20.0]
- epsilon_tally: [0.0, 0.5]
- tau_tie: [0.0, 0.5]
- lambda_min: [0.0, 0.005]
- lambda_max: [0.93, 0.96]
- c0: [0.56, 0.575]
- gate_width: [0.03, 0.05]
- ctx_power: [0.5, 0.8]
- history_window: [100.0, 200.0]
- beta_v: [10.0, 30.0]
- epsilon_v: [0.0, 0.05]
- gamma: [3.0, 7.0]
- consistency_floor: [0.05, 0.10]
- min_cons_ties: {2, 3}
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Context-gated tallying expects a (2, n_features) stimulus; got shape %s.' % (stim.shape,))

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length %d != n_features %d.' % (validities.shape[0], n_features))

    a = stim[0]
    b = stim[1]
    diff = a - b

    wins_a = float(np.sum(a > b))
    wins_b = float(np.sum(b > a))
    tally_d = wins_a - wins_b

    tau_tie = float(parameters['tau_tie'])

    h_a = history.get('option_a_ratings', [])
    h_b = history.get('option_b_ratings', [])
    n_past = min(len(h_a), len(h_b))

    window = int(round(float(parameters['history_window'])))
    if window < 1:
        window = 1
    start = max(0, n_past - window)

    lin_weights = validities - 0.5
    tie_count = 0.0
    abs_evidence_sum = 0.0

    for i in range(start, n_past):
        pa = np.asarray(h_a[i], dtype=float)
        pb = np.asarray(h_b[i], dtype=float)
        past_wins = float(np.sum(pa > pb)) - float(np.sum(pb > pa))
        if abs(past_wins) <= tau_tie:
            tie_count += 1.0
            v_lin = float(np.dot(lin_weights, pa - pb))
            abs_evidence_sum += abs(v_lin)

    current_tie = 1.0 if abs(tally_d) <= tau_tie else 0.0
    tie_count += current_tie
    if current_tie:
        v_lin_current = float(np.dot(lin_weights, diff))
        abs_evidence_sum += abs(v_lin_current)

    total_considered = min(n_past, window) + 1.0
    c = tie_count / total_considered if total_considered > 0.0 else 0.0

    mean_abs_evidence = abs_evidence_sum / max(tie_count, 1.0)

    c0 = float(parameters['c0'])
    gate_width = float(parameters['gate_width'])
    ctx_power = float(parameters['ctx_power'])
    lambda_min = float(parameters['lambda_min'])
    lambda_max = float(parameters['lambda_max'])
    consistency_floor = float(parameters['consistency_floor'])
    min_cons_ties = int(parameters['min_cons_ties'])

    g = float(np.clip((c - c0) / max(gate_width, 1e-12), 0.0, 1.0))

    if tie_count >= min_cons_ties and mean_abs_evidence >= consistency_floor:
        veto = 1.0
    else:
        veto = 0.0

    lamb = lambda_min + (lambda_max - lambda_min) * (g ** ctx_power) * veto

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        p = e / np.sum(e)
        return p

    beta_t = float(parameters['beta_tally'])
    eps_t = float(parameters['epsilon_tally'])
    p_tally = stable_softmax(np.array([wins_a, wins_b]), beta_t)
    p_tally = (1.0 - eps_t) * p_tally + eps_t * np.array([0.5, 0.5])

    gamma = float(parameters['gamma'])
    weights = np.maximum(validities - 0.5, 0.0) ** gamma
    v = float(np.dot(weights, diff))

    beta_v = float(parameters['beta_v'])
    eps_v = float(parameters['epsilon_v'])
    p_valid = stable_softmax(np.array([v, -v]), beta_v)
    p_valid = (1.0 - eps_v) * p_valid + eps_v * np.array([0.5, 0.5])

    probs = lamb * p_valid + (1.0 - lamb) * p_tally
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:
        probs = np.array([0.5, 0.5])
    else:
        probs = probs / total

    return probs
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- kappa: [-0.10, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    subjects = list(data['subject_id'].unique())\n    z_scores = []\n    for sid in subjects:\n        sub = data[data['subject_id'] == sid]\n        d_vals = []\n        s_vals = []\n        y_vals = []\n        for _, row in sub.iterrows():\n            a = np.asarray(row['option_a_ratings'], dtype=float)\n            b = np.asarray(row['option_b_ratings'], dtype=float)\n            n = a.size\n            if n < 2:\n                continue\n            half = n // 2\n            diff = a - b\n            high = float(np.sum(diff[:half]))\n            low = float(np.sum(diff[half:]))\n            d = high + low\n            s = 0.25 * (high - low)\n            y = 1.0 - float(row['response'])\n            d_vals.append(d)\n            s_vals.append(s)\n            y_vals.append(y)\n        if len(y_vals) < 10:\n            continue\n        X = np.column_stack([\n            np.ones(len(y_vals)),\n            np.asarray(d_vals, dtype=float),\n            np.asarray(s_vals, dtype=float)\n        ])\n        y = np.asarray(y_vals, dtype=float)\n        beta = np.linalg.lstsq(X, y, rcond=None)[0]\n        beta = np.asarray(beta, dtype=float).ravel()\n        p_hat = X @ beta\n        p_hat = np.clip(p_hat, 0.005, 0.995)\n        resid = y - p_hat\n        XtX = X.T @ X\n        try:\n            bread = np.linalg.inv(XtX)\n        except np.linalg.LinAlgError:\n            bread = np.linalg.pinv(XtX)\n        meat = (X * (resid ** 2)[:, None]).T @ X\n        cov = bread @ meat @ bread\n        se_d = float(np.sqrt(max(cov[1, 1], 1e-12)))\n        z_d = float(beta[1]) / se_d if se_d > 0 else 0.0\n        z_scores.append(z_d)\n    if len(z_scores) == 0:\n        return 0.0\n    if len(subjects) > 1:\n        return float(np.var(np.asarray(z_scores, dtype=float), ddof=0))\n    return float(z_scores[0])",
  "rationale": "For each subject I fit an ordinary least squares model of the A-choice indicator on two design-derived regressors: d = signed tally advantage of A, and S = centered-validity composition score. I then standardize the fitted d coefficient by its heteroskedasticity-robust standard error, obtaining z_d for that subject. The pooled metric is the population variance of z_d across subjects. Under the advocated context-gated tallying model the tie gate stays closed in this zero-tie design and every subject operates in an unweighted tallying mode with beta_tally bounded in [4.0, 20.0], so d-sensitivity is uniformly strong and the between-subject variance of z_d is moderate. Under validity-graded integration, beta is drawn from [0.1, 20.0], producing a heavy left tail of weakly d-sensitive subjects, especially when kappa and lapse further blur the d contrast. This makes the dispersion of the standardized d-sensitivity z_d substantially larger than under the pure tallying theory. The metric is defined for pooled data as the variance of these subject z-scores, and for a single subject it returns that subject's z_d, which is exactly the unit whose population variance the pooled metric reports."
}
```

## Usage

```json
{
  "input_tokens": 6289,
  "output_tokens": 69907,
  "reasoning_tokens": 69013,
  "provider": "Baidu"
}
```
