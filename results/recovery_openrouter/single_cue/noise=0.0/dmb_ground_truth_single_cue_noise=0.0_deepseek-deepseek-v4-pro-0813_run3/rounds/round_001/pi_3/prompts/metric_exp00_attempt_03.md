# metric_exp00_attempt_03

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
**Validities (n_features=8):** [0.5, 0.52, 0.54, 0.56, 0.94, 0.96, 0.98, 1.0]

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 12: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** This design isolates the validity-weighting term of the advocated theory against pure Tallying. Eight experts have validities in two clusters (0.50-0.56 and 0.94-1.00), so the centered validity sums are -0.88 for the low cluster and +0.88 for the high cluster. The critical trials are eight equal-tally (d=0) pairs: A and B each win on exactly four experts, so Tallying is forced to predict p(A)=0.5 on every one of them. Across these pairs I vary which four experts A wins on, producing a validity-weighted advantage D = kappa*S with S = +1.76, +1.00, +0.84, +0.16, -0.16, -0.84, -1.00, -1.76. Validity-graded integration therefore predicts a monotone psychometric curve across S (p(A) > 0.5 for positive S when kappa > 0, reversed when kappa < 0), whereas Tallying predicts a flat line at 0.5. Four additional d=+/-1 pairs hold the tally difference constant while flipping whether A's wins are on high- or low-validity experts (S=+/-1.57); Tallying predicts identical p(A) within each d, while the advocated theory predicts a validity-composition shift. The 12 unique pairs are repeated 8 times for 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
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
[0] rationale: This metric estimates, separately for each subject, the linear sensitivity of choosing A to the validity-weighted advantage of A over B, S = sum_j centered_validity_j * (A_j - B_j), using all 96 trials. The design balances S within each tally-difference level, so Tallying predicts zero expected slope regardless of beta or epsilon and contributes only binomial noise. The advocated validity-graded rule predicts a slope whose sign follows kappa and whose magnitude increases with beta and kappa. Taking the absolute slope per subject removes sign cancellation from the symmetric kappa prior while preserving a positive contrast against the near-zero random slopes under Tallying, then averaging across subjects gives a stable scalar.
metric_source:
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.52, 0.54, 0.56, 0.94, 0.96, 0.98, 1.0], dtype=float)
    centered = validities - validities.mean()
    subj_abs_slopes = []
    for subj_id, subj in data.groupby('subject_id'):
        a = np.asarray(subj['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(subj['option_b_ratings'].tolist(), dtype=float)
        s = np.sum((a - b) * centered[None, :], axis=1)
        y = (subj['response'].to_numpy(dtype=float) == 0).astype(float)
        denom = float(np.sum(s * s))
        if denom <= 1e-12:
            continue
        slope = float(np.sum(s * (y - y.mean())) / denom)
        subj_abs_slopes.append(abs(slope))
    return float(np.mean(subj_abs_slopes)) if subj_abs_slopes else 0.0

outcome: self_sim=0.0439 (var=0.0011) adversary_sim=0.0237 (var=0.0004) welch_t=+2.642 p=0.01188 (N=25, alpha=0.01) -> reject

[1] rationale: This metric estimates, separately for each subject, a regularized logistic coefficient on the validity-weighted signed advantage S = sum_j (v_j - mean(v)) * (A_j - B_j), while simultaneously controlling for the raw tally difference d. The logistic link matches the softmax core of both theories and avoids the attenuation of a linear probability slope. Under Tallying, the true coefficient on S is exactly zero within every level of d, so fitted slopes are pure estimation noise. Under the advocated validity-graded rule, the coefficient is proportional to beta * kappa (on the log-odds scale), with sign determined by kappa. Taking the absolute value per subject removes sign cancellation from the symmetric kappa prior while preserving the validity-graded effect; averaging across subjects gives a stable scalar. The small ridge penalty stabilizes subjects with near-deterministic choices.
metric_source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.52, 0.54, 0.56, 0.94, 0.96, 0.98, 1.0], dtype=float)
    centered = validities - validities.mean()

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    def _ridge_logit_s_slope(y, s, d, lam=1.0):
        n = len(y)
        zs = (s - s.mean()) / s.std()
        zd = (d - d.mean()) / d.std()
        X = np.column_stack([np.ones(n), zd, zs])
        coef = np.zeros(3)
        pen = np.diag(np.array([0.0, lam, lam]))
        for _ in range(60):
            eta = X @ coef
            p = _sigmoid(eta)
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            w = p * (1.0 - p)
            grad = X.T @ (y - p) - pen @ coef
            hess = X.T @ (w[:, None] * X) + pen
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            coef = coef - delta
            if np.max(np.abs(delta)) < 1e-8:
                break
        return coef[2]

    abs_slopes = []
    for _, subj in data.groupby('subject_id'):
        a = np.asarray(subj['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(subj['option_b_ratings'].tolist(), dtype=float)
        diff = a - b
        s = np.sum(diff * centered[None, :], axis=1)
        d = np.sum(a > b, axis=1) - np.sum(b > a, axis=1)
        y = (subj['response'].to_numpy(dtype=float) == 0.0).astype(float)
        if s.std() == 0.0 or d.std() == 0.0:
            continue
        est = _ridge_logit_s_slope(y, s, d)
        if np.isfinite(est):
            abs_slopes.append(abs(float(est)))
    return float(np.mean(abs_slopes)) if abs_slopes else 0.0

outcome: self_sim=3540934683199359488.0000 (var=3941517168122683246349315084512133120.0000) adversary_sim=2467428897510780928.0000 (var=1935726660661475785368942747219656704.0000) welch_t=+2.214 p=0.03218 (N=25, alpha=0.01) -> reject

[2] rationale: This metric uses the absolute t-statistic for the validity-weighted signed advantage S = sum_j (v_j - mean(v)) * (A_j - B_j), from a within-subject linear probability regression that also controls the raw tally d and d^2. Controlling d removes Tallying's main deterministic signal and shrinks the residual variance used to standardize the S coefficient. Under Tallying the true S coefficient is zero, so the absolute t-statistic is pure noise concentrated around its null expected value of approximately 0.8. Under validity-graded integration the S coefficient carries a true noncentrality proportional to beta*kappa, so abs(t) is systematically larger. Capping abs(t) at 3 prevents occasional near-separable subjects from dominating the between-subject variance while preserving the central-location contrast. The pooled metric is the mean of this capped subject-level statistic.
metric_source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.5, 0.52, 0.54, 0.56, 0.94, 0.96, 0.98, 1.0], dtype=float)
    centered = validities - validities.mean()

    def _subj_metric(sub):
        a = np.asarray(sub['option_a_ratings'].tolist(), dtype=float)
        b = np.asarray(sub['option_b_ratings'].tolist(), dtype=float)
        diff = a - b
        d = (a > b).sum(axis=1).astype(float) - (b > a).sum(axis=1).astype(float)
        s = diff @ centered
        y = (sub['response'].to_numpy(dtype=float) == 0.0).astype(float)

        n = len(y)
        if n < 8:
            return 0.0

        d_c = d - d.mean()
        d2_c = (d * d) - np.mean(d * d)
        s_c = s - s.mean()
        X = np.column_stack([np.ones(n), d_c, d2_c, s_c])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0

        yhat = X @ coef
        resid = y - yhat
        ss = float(resid @ resid)
        df = n - 4
        if df <= 0:
            return 0.0

        sigma2 = ss / df
        if not np.isfinite(sigma2) or sigma2 <= 1e-12:
            return 0.0

        try:
            cov = sigma2 * np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            return 0.0

        var_s = float(cov[3, 3])
        if not np.isfinite(var_s) or var_s <= 1e-12:
            return 0.0

        t_s = float(coef[3] / np.sqrt(var_s))
        return float(min(abs(t_s), 3.0))

    values = []
    for _, sub in data.groupby('subject_id'):
        values.append(_subj_metric(sub))

    if not values:
        return 0.0
    return float(np.mean(values))
outcome: self_sim=1.1906 (var=0.7168) adversary_sim=0.6560 (var=0.2793) welch_t=+2.678 p=0.01066 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    validities = np.array([0.5, 0.52, 0.54, 0.56, 0.94, 0.96, 0.98, 1.0], dtype=float)\n    centered = validities - validities.mean()\n    values = []\n\n    for _, subj in data.groupby('subject_id'):\n        a = np.asarray(subj['option_a_ratings'].tolist(), dtype=float)\n        b = np.asarray(subj['option_b_ratings'].tolist(), dtype=float)\n        diff = a - b\n        d = (diff > 0).sum(axis=1).astype(float) - (diff < 0).sum(axis=1).astype(float)\n        s = np.sum(diff * centered[None, :], axis=1)\n        y = (subj['response'].to_numpy(dtype=float) == 0.0).astype(float)\n        n = len(y)\n\n        if n < 8 or np.var(s) < 1e-10:\n            continue\n\n        d_c = d - d.mean()\n        d2_c = (d * d) - np.mean(d * d)\n        s_c = s - s.mean()\n        X = np.column_stack([np.ones(n), d_c, d2_c, s_c])\n\n        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)\n        slope_s = float(coef[3])\n        if not np.isfinite(slope_s):\n            continue\n\n        resid = y - X @ coef\n        rss = float(resid @ resid)\n        df_error = n - 4\n        if df_error <= 0:\n            continue\n\n        sigma2 = rss / float(df_error)\n        if sigma2 <= 1e-10:\n            sigma2 = 1e-10\n\n        try:\n            cov = sigma2 * np.linalg.pinv(X.T @ X)\n        except np.linalg.LinAlgError:\n            continue\n\n        var_s = float(cov[3, 3])\n        se_s = 0.05 if var_s <= 1e-12 else float(np.sqrt(var_s))\n\n        raw_sensitivity = min(abs(slope_s), 0.25)\n        evidence = min(abs(slope_s) / se_s, 2.0)\n\n        values.append(raw_sensitivity + 0.035 * evidence)\n\n    if not values:\n        return 0.0\n    return float(np.mean(values))",
  "rationale": "This metric combines two subject-level summaries of validity-graded responding. The first is the raw linear probability slope of choosing A on the validity-weighted advantage S = sum_j (v_j - mean(v)) * (A_j - B_j), after controlling the raw tally difference d and d^2. Under Tallying the true S slope is zero, while under validity-graded integration it is proportional to beta * kappa, with sign depending on kappa; taking the absolute value avoids sign cancellation across the symmetric kappa prior. Raw absolute slopes are stable across subjects but the mean contrast was only modest. The second component is a bounded evidence statistic, abs(slope) divided by its within-subject standard error and capped at 2, which retains information about how detectable the S effect is relative to response noise. A near-separable subject cannot dominate because of the cap; the small weight 0.035 places the evidence correction on approximately the same scale as the raw slope. The composite therefore carries both the stable probability-scale effect and a regularized signal-to-noise contribution, giving a larger and tighter separation between the advocated validity-graded theory and pure Tallying."
}
```

## Usage

```json
{
  "input_tokens": 6583,
  "output_tokens": 76975,
  "reasoning_tokens": 76093,
  "provider": "Baidu"
}
```
