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
**Validities (n_features=14):** [0.96, 0.96, 0.96, 0.96, 0.96, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 12: A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Rationale:** The fourteen validities form two groups: five 0.96 experts and nine 0.50 experts. All 14 features discriminate on every unique pair, so the raw inverse-clean tally D = (number of A=0/B=1 positions) - (number of A=1/B=0 positions) is +2 for pairs 1-6 and -2 for the swapped mirrors 7-12. Within the high group, x = high-clean advantage for A over B takes the values -5, -3, -1, +1, +3, +5; the low group is arranged to give y = D - x. This holds the raw tally constant while moving the high-validity cues from all favoring B to all favoring A. The advocated theory uses saturating logit-tanh weights that are normalized and bounded; with this validity mix the high/low weight ratio is only about 1.2 at the median of sat_m, so in the x=-5 pair the seven low-validity A-clean cues outweigh the five high-validity B-clean cues and p(A) is about 0.65. The competing theory includes the 0.96 cues with probability about 0.96 and the 0.50 cues with probability about 0.50, then tallies included cues equally, giving expected included evidence about 7*0.50 - 5*0.96 = -1.31 and p(A) about 0.12 for that same pair. The next x condition and its mirror also disagree in direction, and the two models show different crossover locations across x: the competing model crosses to A around x=-1, while the advocated model remains A-favoring for all D=+2 pairs. The D=-2 mirrors reverse the signs and provide the complementary quantitative dissociation. Twelve unique pairs repeated K=8 gives 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** All subjects treat a binary expert rating of 0 as clean/favorable and 1 as a defect. On each trial every discriminating expert contributes continuous inverse-valence evidence weighted by a saturating, near-tally-like transform of communicated validity: w_j = 1 + m * tanh(2 * [logit(v_j) - mean(logit(v))] / half_range), with the weight vector normalized to mean one. The subject-level saturation m is small and capped, so high-validity experts are attended more but cannot generate extreme top-to-bottom weight ratios: a three-cue high-validity coalition can win small conflicts yet is overridden by a five-cue low-validity majority. When the weighted evidence is inconclusive, or when the raw inverse-clean tally is exactly balanced, subjects fall back to inspecting changed experts in descending validity order with tied highest-validity experts randomly permuted per trial; the precision of that fallback grows with the validity gap separating the best expert from the next-best expert. Choice is a softmax over net evidence with subject-specific gain and lapse. There is no valence-orientation mixture and no separate top-cue attention bonus.

**Parameters:**
- slope_a: [0.02, 0.30]
- gain: [0.10, 1.50]
- epsilon: [0.03, 0.16]
- tie_gain: [0.30, 1.30]
- tie_tol: [0.02, 0.15]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a_row = stim[0]
    b_row = stim[1]
    n_features = stim.shape[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Subject-level parameter shaping -----------------------------------
    # Saturation amplitude for the validity-to-weight map.
    # Kept deliberately small: this is the main fix for the unbounded
    # exponential weights that let three high-validity cues destroy the
    # five-cue low-validity majority in Experiment 8.
    u_m = float(np.clip((float(parameters['slope_a']) - 0.02) / 0.28, 0.0, 1.0))
    sat_m = float(np.exp(_trunc_qnorm(
        u_m, np.log(0.10), 0.45, np.log(0.03), np.log(0.18))))

    # Main choice gain: restore the accepted base's moderate calibration.
    u_g = float(np.clip((float(parameters['gain']) - 0.10) / 1.40, 0.0, 1.0))
    gain = float(np.exp(_trunc_qnorm(
        u_g, np.log(0.40), 0.50, np.log(0.10), np.log(1.50))))

    lo_logit = np.log(0.03 / 0.97)
    hi_logit = np.log(0.16 / 0.84)
    mu_logit = np.log(0.08 / 0.92)
    u_e = float(np.clip((float(parameters['epsilon']) - 0.03) / 0.13, 0.0, 1.0))
    z_eps = _trunc_qnorm(u_e, mu_logit, 0.35, lo_logit, hi_logit)
    epsilon = float(_inv_logit(z_eps))

    # Base tie-break gain is moderate, but is amplified by the validity gap
    # between the selected best expert and the next-best changed expert.
    u_tg = float(np.clip((float(parameters['tie_gain']) - 0.30) / 1.00, 0.0, 1.0))
    tie_gain = float(np.exp(_trunc_qnorm(
        u_tg, np.log(0.45), 0.45, np.log(0.30), np.log(0.70))))

    # Tightened inconclusive-evidence threshold.
    u_tt = float(np.clip((float(parameters['tie_tol']) - 0.02) / 0.13, 0.0, 1.0))
    tie_tol = float(_trunc_qnorm(u_tt, 0.08, 0.030, 0.02, 0.15))

    # ---- Saturating continuous inverse-valence weights ---------------------
    # 0 is clean/favorable, 1 is a defect.
    v = np.clip(val, 0.50001, 0.9999)
    logit_v = np.log(v / (1.0 - v))
    center = float(np.mean(logit_v))

    if n_features > 1:
        spread = float(np.max(logit_v) - np.min(logit_v))
        if spread <= 1e-8:
            z = np.zeros(n_features, dtype=float)
        else:
            half_spread = 0.5 * spread
            z = np.tanh(2.0 * (logit_v - center) / half_spread)
        w = 1.0 + sat_m * z
        w = w / np.mean(w)
    else:
        w = np.ones(1, dtype=float)

    changed_mask = a_row != b_row
    if not np.any(changed_mask):
        return np.ones(2) / 2.0

    # A=0/B=1 gives +w_j; A=1/B=0 gives -w_j.
    diff = (b_row > a_row).astype(float) - (a_row > b_row).astype(float)
    raw_tally = float(np.sum(diff))
    net_evidence = float(np.dot(w, diff))

    if abs(net_evidence) <= tie_tol or abs(raw_tally) <= 0.5:
        # Inconclusive evidence or an exactly balanced raw inverse-clean
        # tally: inspect changed experts in descending validity order.
        # Ties at the maximum validity are exchangeable and resolved by a
        # random permutation on each trial.
        disc = np.flatnonzero(changed_mask)
        top_val = float(np.max(val[disc]))
        tied = disc[np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        if tied.size == 0:
            tied = disc
        idx = int(tied[np.random.randint(tied.size)])

        # Validity-gap-dependent tie-break precision.  This is not a top-cue
        # bonus inside the tally; it is how strongly subjects rely on the
        # lexical fallback only when they are already in the fallback regime.
        non_top = disc[~np.isclose(val[disc], top_val, rtol=1e-12, atol=1e-12)]
        second_val = float(np.max(val[non_top])) if non_top.size > 0 else top_val
        gap = top_val - second_val
        gap_boost = 1.0 + 2.5 * np.tanh(gap / 0.15)

        if a_row[idx] == 0.0 and b_row[idx] == 1.0:
            sign = 1.0   # clean cue favors A
        elif a_row[idx] == 1.0 and b_row[idx] == 0.0:
            sign = -1.0  # clean cue favors B
        else:
            return np.ones(2) / 2.0

        scores = np.array([sign, -sign], dtype=float)
        beta_eff = tie_gain * gap_boost
    else:
        scores = np.array([net_evidence, -net_evidence], dtype=float)
        beta_eff = gain

    logits = beta_eff * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Parameters:**
- alpha: [5.0, 9.0]
- gamma_logit: [-2.50, 2.50]
- beta_tally_log: [-0.30, 0.50]
- beta_tiebreak_log: [0.40, 1.10]
- epsilon_logit: [-3.00, -0.40]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    a = np.array([list(row) for row in data['option_a_ratings']], dtype=float)\n    b = np.array([list(row) for row in data['option_b_ratings']], dtype=float)\n\n    clean_a = (a == 0) & (b == 1)\n    clean_b = (a == 1) & (b == 0)\n\n    D = clean_a.sum(axis=1) - clean_b.sum(axis=1)\n    x = clean_a[:, :5].sum(axis=1) - clean_b[:, :5].sum(axis=1)\n\n    chose_a = (data['response'].to_numpy(dtype=float) == 0)\n\n    mask_pos = (D == 2) & (x == -3)\n    mask_mirror = (D == -2) & (x == 3)\n\n    if mask_pos.sum() == 0 or mask_mirror.sum() == 0:\n        return 0.0\n\n    p_pos = float(chose_a[mask_pos].mean())\n    p_mirror = float(chose_a[mask_mirror].mean())\n    return p_pos - p_mirror\n",
  "rationale": "This metric targets the x=-3 conflict pair and its mirror. In the D=+2, x=-3 pair, the advocated bounded-weight model has weighted evidence dominated by the five low-validity A-clean cues over three high-validity B-clean cues, so p(A) is clearly above 0.5. The competing inclusion-tally model has expected included evidence 5*0.50 - 3*0.96 = -0.38, so p(A) is below 0.5. The mirror pair reverses the sign. Subtracting mirror p(A) therefore yields a positive value under the advocated theory and a negative value under the competing theory. I use x=-3 rather than x=-5 because the x=-5 pair can enter the advocated fallback regime for high saturation subjects, adding between-subject variance. This contrast averages 16 critical trials per subject, giving a stable per-subject estimate."
}
```

## Usage

```json
{
  "input_tokens": 6495,
  "output_tokens": 43238,
  "reasoning_tokens": 42736,
  "provider": "Baidu"
}
```
