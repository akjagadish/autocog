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
**Validities (n_features=6):** [0.96, 0.94, 0.9, 0.8, 0.6, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** At gamma = 2.0 the six validity weights v^gamma are [0.9216, 0.8836, 0.81, 0.64, 0.36, 0.25]. The first eight rows use positive D = sum_j v_j^gamma * (A_j - B_j) values of 0.038, 0.0736, 0.1116, 0.17, 0.20, 0.2436, 0.2816, and 0.3116; the last eight rows are exact A/B mirrors. The staircase is built to bracket the TAVU threshold tau in [0.16, 0.19]: the first three D values lie in the TAVU dead zone, 0.17 sits at threshold, 0.20 just above it, and the remaining values are clearly above threshold. For the negative-weight majority, TAVU-tn predicts p(A) near 0.50 for the first three positive pairs, a sharp drop to about 0.26 at D = 0.20, and then a plateau near 0.20 to 0.21; the mirrored side gaps are about 0.00, 0.00, 0.01, 0.26, 0.47, 0.59, 0.59, and 0.59. CNVW-tau has no dead zone and predicts a smooth continuous decline in p(A) from about 0.42 at D = 0.038 to about 0.24 at D = 0.31, with mirrored side gaps growing monotonically from about 0.16 to 0.53. The critical dissociation is dead-zone flatness below threshold plus a threshold jump and plateau versus a smooth saturation curve, with the two side-gap functions crossing near D = 0.20. This quantitative interaction cannot be removed by beta, lambda, bias, or lapse within the stated ranges.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Thresholded anti-validity utility with Gaussian threshold noise and a small validity-consistent soft leak (TAVU-tn). Subjects compute a validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j). A subject-level latent sign dominates: most subjects use negative weighting and make the anti-validity choice when |D| exceeds a subject-level threshold tau_i, but threshold placement is noisy across trials, so the aggregate threshold boundary is graded rather than a step. Within the dead zone and below the sign-following ceiling, subjects fall back on a subject-level bias. A small plain-D sigmoidal leak in the validity-consistent direction weakly damps extreme anti choices at large |D|. A uniform lapse adds residual noise. The dominant process remains thresholded sign, not continuous magnitude accumulation.

**Parameters:**
- gamma: [1.8, 2.2]
- tau: [0.16, 0.19]
- tau_sd: [0.02, 0.04]
- p_negative: [0.88, 0.94]
- sign_draw: [0.0, 1.0]
- bias: [0.46, 0.54]
- lapse: [0.12, 0.16]
- soft_weight: [0.00, 0.03]
- soft_beta: [0.7, 1.1]
- follow_max: [0.66, 0.74]
- validities: validities

**`predict source code`:**
```python
import numpy as np
import math


def predict(parameters, state, history):
    """Thresholded anti-validity utility with Gaussian threshold noise (TAVU-tn).

    D = sum_j v_j^gamma * (A_j - B_j). A subject-level sign dominates:
    most subjects apply a negative validity weighting and choose the
    anti-validity option when |D| exceeds a subject-level threshold.
    Threshold placement is noisy across trials, implemented analytically
    as p_above = Phi((|D| - tau) / tau_sd). Inside the dead zone, and
    below the sign-following ceiling, choices fall back on a subject-level
    bias. A small plain-D sigmoidal leak weakly opposes the anti-validity
    sign at large |D|, and a uniform lapse adds residual noise.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "TAVU-tn expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    tau_sd = float(parameters["tau_sd"])
    p_negative = float(parameters["p_negative"])
    sign_draw = float(parameters["sign_draw"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    w_soft = float(parameters["soft_weight"])
    beta_soft = float(parameters["soft_beta"])
    follow_max = float(parameters["follow_max"])

    # Latent validity-sign mixture: most subjects are negative weighters.
    sign = -1.0 if sign_draw < p_negative else 1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Analytic Gaussian threshold noise: probability that |D| crosses tau
    # when the trial-level threshold is perturbed.
    z_th = (abs(d) - tau) / tau_sd
    p_above = 0.5 * (1.0 + math.erf(z_th / math.sqrt(2.0)))
    p_above = float(np.clip(p_above, 0.0, 1.0))

    # Thresholded anti-validity target for p(A).
    if sign < 0.0:
        if d > 0.0:
            target_a = 0.0
        elif d < 0.0:
            target_a = 1.0
        else:
            target_a = bias
    else:
        if d > 0.0:
            target_a = 1.0
        elif d < 0.0:
            target_a = 0.0
        else:
            target_a = bias

    # Dominant thresholded sign rule, with a follow ceiling that keeps
    # some residual bias even after threshold crossing.
    core_threshold = (
        follow_max * (p_above * target_a + (1.0 - p_above) * bias)
        + (1.0 - follow_max) * bias
    )

    # Small validity-consistent soft leak. This is deliberately very weak
    # below |D| = 0.2 and only mildly damps extreme anti choices.
    z = beta_soft * d
    if z >= 0.0:
        soft_a = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        soft_a = ez / (1.0 + ez)

    core_a = (1.0 - w_soft) * core_threshold + w_soft * soft_a
    p_a = (1.0 - lapse) * core_a + lapse / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

**Parameters:**
- gamma: [1.5, 2.5]
- tau: [0.10, 0.25]
- lambda: [0.8, 2.0]
- beta: [0.8, 2.0]
- bias: [-0.25, 0.25]
- epsilon: [0.03, 0.10]
- p_positive: [0.03, 0.10]
- sign_draw: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Compressive negative validity-weighted integration with subject-level tau.

    D = sum_j v_j^gamma * (A_j - B_j)
    psi(D) = D / (tau + abs(D))
    S = sign * lambda * psi(D) + bias
    p(A) = lapse/2 + (1-lapse) * sigmoid(beta * S)

    Most subjects use sign = -1; a minority draw sign = +1.
    history is intentionally ignored because there is no feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CNVW-tau expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    lam = float(parameters["lambda"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    tau = float(parameters["tau"])

    # Latent sign mixture: most subjects use negative-valence weighting;
    # a minority trust positive-valence evidence and get a positive sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Subject-level strongly saturating bounded transform. The small
    # half-saturation constant tau forces most non-negligible validity-
    # weighted differences toward the sign bound, so sign and not
    # continuous magnitude drives the main choice contrast.
    psi = d / (tau + abs(d))

    score = sign * lam * psi + bias
    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    validities = np.array([0.96, 0.94, 0.9, 0.8, 0.6, 0.6], dtype=float)\n\n    def subject_contrast(group: pd.DataFrame) -> float:\n        rows = []\n        for _, row in group.iterrows():\n            a = tuple(int(x) for x in row['option_a_ratings'])\n            b = tuple(int(x) for x in row['option_b_ratings'])\n            a_arr = np.array(a, dtype=float)\n            b_arr = np.array(b, dtype=float)\n            d_ref = float(np.dot(validities ** 2.0, a_arr - b_arr))\n            pair = tuple(sorted((a, b)))\n            orient = 0 if a < b else 1\n            rows.append((pair, orient, abs(d_ref), 1.0 - float(row['response'])))\n\n        recs = pd.DataFrame(rows, columns=['pair', 'orient', 'd_ref_abs', 'p_a'])\n        means = recs.groupby(['pair', 'orient'])['p_a'].mean().reset_index()\n\n        def gap_fn(s):\n            vals = list(s)\n            if len(vals) < 2:\n                return float('nan')\n            return abs(vals[0] - vals[1])\n\n        gaps = means.groupby('pair')['p_a'].apply(gap_fn)\n        d_ref = recs.groupby('pair')['d_ref_abs'].first()\n\n        low = gaps[d_ref < 0.15]\n        high = gaps[d_ref > 0.22]\n        if low.empty or high.empty:\n            return float('nan')\n        return float(high.mean() - 2.0 * low.mean())\n\n    per_subject = data.groupby('subject_id').apply(subject_contrast)\n    return float(per_subject.mean())\n",
  "rationale": "For each unordered stimulus pair, the metric computes a side gap: the absolute difference in mean A-choice rate between the two mirror orientations. This difference cancels subject-level bias and lapse because those add equally to both orientations, leaving the evidence-dependent choice contrast. Using the fixed validities with the designed D values, low-evidence pairs are selected with D_ref < 0.15 and high-evidence pairs with D_ref > 0.22. TAVU-tn predicts a dead zone where low-D side gaps are near zero and high-D side gaps jump to a high plateau, so high_gap - 2*low_gap is strongly positive. CNVW-tau has no dead zone: low-D side gaps are already sizable, and its concave smooth side-gap curve makes twice the low gap roughly cancel the high gap, producing values near zero or negative. Averaging per subject keeps the metric aligned with between-subject variability and avoids sign-mixture cancellation."
}
```

## Usage

```json
{
  "input_tokens": 5487,
  "output_tokens": 33132,
  "reasoning_tokens": 32452,
  "provider": "Baidu"
}
```
