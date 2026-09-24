# metric_exp00_attempt_01

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
**Validities (n_features=11):** [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.62, 0.6, 0.7, 0.61, 0.61]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** All six pairs have a raw inverse tally of +2 or -2, so the competing theory is forced to use its mean-centered validity-weighted tally rather than a raw tie-break. The validities are deliberately skewed: features 0-5 are high (1.00 to 0.95), but features 3-5 are tied on every trial and act as high-validity anchors that raise the validity mean to about 0.795. The five low-validity features 6-10 (0.62, 0.60, 0.58, 0.56, 0.54) actually do the discriminating work. At median advocated parameters alpha=7 and gamma=-3.5, inclusion probabilities are about 0.97 for the high cues and 0.70, 0.67, 0.64, 0.60, 0.57 for the low cues. In the competing theory the tied high anchors lower the mean-centered weights of the low cues to about 0.72 to 0.59, while high-cue weights are about 1.33. For the raw +2 pairs, a low-validity majority of 3, 4, or 5 A-clean cues is opposed by 1, 2, or 3 high-validity B-clean cues. The competing weighted score goes from +0.73 through -0.08 to -0.66, so p(A) drops from about 0.64 to 0.49 to 0.37. The advocated expected included tally stays positive at about 1.03, 0.54, and 0.27, so advised p(A) stays at about 0.84, 0.73, and 0.62. The raw -2 mirrors reverse the signs and create a crossover: at the 4- and 5-low-cue levels the two theories make opposite directional predictions. The key manipulation is the tied high anchors, which change the competing mean-centered weights without changing the advocated inclusion tally. Six unique pairs repeated K=16 gives 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

**Parameters:**
- beta_tally: [0.22, 0.85]
- beta_tiebreak: [0.50, 1.70]
- epsilon: [0.04, 0.18]
- validity_weight: [0.60, 2.40]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    """Map a (0,1) draw to a normal quantile truncated to [lo, hi]."""
    u = float(np.clip(u, 0.0, 1.0))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


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

    # --- Subject-level parameter transforms ---------------------------------
    # Raw declared ranges are mapped through truncated normal/lognormal
    # distributions so the intended medians and dispersions are realized
    # per subject rather than sampled uniformly.
    beta_tally_raw = float(parameters['beta_tally'])
    beta_tiebreak_raw = float(parameters['beta_tiebreak'])
    epsilon_raw = float(parameters['epsilon'])
    validity_weight_raw = float(parameters['validity_weight'])

    # Lower median inverse temperature to soften the inverse-tally gradient,
    # especially the s = +/-2 separation in Experiment 4, while retaining a
    # moderately wide support for between-subject variance.
    u_beta = (beta_tally_raw - 0.22) / (0.85 - 0.22)
    beta_tally = np.exp(_trunc_qnorm(
        u_beta, np.log(0.45), 0.40, np.log(0.22), np.log(0.85)))
    beta_tally = float(np.clip(beta_tally, 0.22, 0.85))

    # Keep tie-break temperature near 1.0 but widen the subject-level range.
    # The median is nudged to 1.02 to hold Experiment 3 near its observed
    # anti-TTB tie-break rate once the lapse distribution is widened.
    u_tie = (beta_tiebreak_raw - 0.50) / (1.70 - 0.50)
    beta_tiebreak = np.exp(_trunc_qnorm(
        u_tie, np.log(1.02), 0.40, np.log(0.50), np.log(1.70)))
    beta_tiebreak = float(np.clip(beta_tiebreak, 0.50, 1.70))

    # Shift the single skewed validity-weight distribution upward as advised:
    # median near 1.60 and a small upper tail toward 2.40. This preserves a
    # single unimodal distribution with no rejected mixture component.
    u_w = (validity_weight_raw - 0.60) / (2.40 - 0.60)
    z_w = _trunc_qnorm(
        u_w, np.log(1.10), 0.40, np.log(0.10), np.log(1.90))
    validity_weight = 0.50 + np.exp(z_w)
    validity_weight = float(np.clip(validity_weight, 0.60, 2.40))

    # Widen the lapse distribution only modestly, centered near 0.09, so
    # central tendencies in Experiments 3, 5, and 7 are not flattened while
    # enough heterogeneity is added.
    logit_lo = np.log(0.04 / 0.96)
    logit_hi = np.log(0.18 / 0.82)
    logit_mu = np.log(0.09 / 0.91)
    u_eps = (epsilon_raw - 0.04) / (0.18 - 0.04)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.40, logit_lo, logit_hi)
    epsilon = 1.0 / (1.0 + np.exp(-z_eps))
    epsilon = float(np.clip(epsilon, 0.04, 0.18))

    # --- Inverse clean-cue comparison --------------------------------------
    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0.0:
        # Weak-to-moderate, all-cue validity weighting. Mean-centering keeps
        # the weights near one so low-validity majorities can still move the
        # tally when present.
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)

        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        # Exact raw tally tie: consult features in descending validity order
        # and choose the alternative with a clean 0 on the first discriminator.
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break

        if winner is None:
            return np.ones(2) / 2.0

        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
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
[0] rationale: The metric splits trials by the raw inverse clean-cue tally: s = count where A has 0 and B has 1 minus count where A has 1 and B has 0. In this design the s > 0 trials are the low-validity clean-cue majority pairs and the s < 0 trials are their mirror-image pairs. It then computes P(choose A | s > 0) minus P(choose A | s < 0). The advocated inclusion-tally theory treats included clean cues as roughly equal-strength evidence, so it produces a large positive contrast: A is usually chosen when A has more inverse-clean cues and avoided on the mirrors. The competing validity-weighted theory mean-centers the validities, which shrinks or reverses the low-validity majority, especially at the 5-vs-3 cue level, so its same contrast is close to zero or even negative. Because each subject has 48 trials of each sign, the per-subject binomial noise is modest, making the between-subject variance small enough for the contrast to be highly diagnostic.
metric_source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    pos = []
    neg = []
    for row in data.itertuples(index=False):
        a = row.option_a_ratings
        b = row.option_b_ratings
        s = 0
        for x, y in zip(a, b):
            if x == 0 and y == 1:
                s += 1
            elif x == 1 and y == 0:
                s -= 1
        if s > 0:
            pos.append(row.response)
        elif s < 0:
            neg.append(row.response)
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_a_pos = 1.0 - float(np.mean(pos))
    p_a_neg = 1.0 - float(np.mean(neg))
    return float(p_a_pos - p_a_neg)
outcome: self_sim=0.2733 (var=0.1024) adversary_sim=0.1179 (var=0.0363) welch_t=+2.086 p=0.04353 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    t1a = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)\n    t1b = (0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0)\n    t4a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0)\n    t4b = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)\n    t2a = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)\n    t2b = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)\n    t5a = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1)\n    t5b = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)\n    t3a = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)\n    t3b = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)\n    t6a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1)\n    t6b = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)\n\n    p_adv = (0.746, 0.646, 0.563)\n    p_comp = (0.591, 0.529, 0.472)\n\n    prob_map = {\n        (t1a, t1b): (p_adv[0], p_comp[0]),\n        (t4a, t4b): (1.0 - p_adv[0], 1.0 - p_comp[0]),\n        (t2a, t2b): (p_adv[1], p_comp[1]),\n        (t5a, t5b): (1.0 - p_adv[1], 1.0 - p_comp[1]),\n        (t3a, t3b): (p_adv[2], p_comp[2]),\n        (t6a, t6b): (1.0 - p_adv[2], 1.0 - p_comp[2]),\n    }\n\n    total = 0.0\n    n = 0\n    for row in data.itertuples(index=False):\n        key = (tuple(row.option_a_ratings), tuple(row.option_b_ratings))\n        entry = prob_map.get(key)\n        if entry is None:\n            continue\n        pa, pc = entry\n        pa = float(np.clip(pa, 1e-6, 1.0 - 1e-6))\n        pc = float(np.clip(pc, 1e-6, 1.0 - 1e-6))\n        y = 1.0 if int(row.response) == 0 else 0.0\n        total += y * np.log(pa / pc) + (1.0 - y) * np.log((1.0 - pa) / (1.0 - pc))\n        n += 1\n\n    if n == 0:\n        return 0.0\n    return float(total / n)\n",
  "rationale": "All six displayed pairs are raw inverse-cue tallies of +2 or -2, so the previously attempted aggregate majority contrast mixes trial types that are informative with trial types that dilute the distinction. This metric instead performs a trial-level Bernoulli model comparison. For the three positive trial types, the advocated inclusion-tally model at median parameters predicts P(choose A) of 0.746, 0.646, and 0.563; the competing mean-centered validity-weighting model predicts 0.591, 0.529, and 0.472. Mirror trials use one minus these probabilities. For every trial the metric computes the log-likelihood ratio of the observed A or B response under the advocated prediction versus the competing prediction, then returns the per-trial average. The per-trial average keeps pooled and per-subject values on the same scale, and averaging across 96 trials per subject controls between-subject variance. Advocated data should produce a clearly positive score while competing data should be negative or near zero, because the two models place probability mass on opposite sides of 0.5 for the most conflicted pairs."
}
```

## Usage

```json
{
  "input_tokens": 6046,
  "output_tokens": 46686,
  "reasoning_tokens": 45502,
  "provider": "Baidu"
}
```
